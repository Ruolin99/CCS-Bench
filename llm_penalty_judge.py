"""
===============================================================================
 Autonomous Driving Traffic Rules Knowledge Graph Engine — Phase 3: LLM Judge Penalty Severity Rating & Result Backfilling Engine
 Phase 5: LLM Judge Penalty Severity Rating & Result Backfilling Engine
===============================================================================
 Core Functionality:
   Resolves the homogeneity collapse issue in the legal severity dimension of original datasets.
   Reads global violation penalty catalogues (违规扣分情况.xlsx), partitions them by jurisdiction,
   and dispatches LLMs (e.g., Qwen-Plus/Max) as AI traffic judges to evaluate micro-topological
   violations against actual penalties on a severity scale (0.0–1.0).
   Establishes thread-safe local checkpointing (penalty_cache.json) and backfills severity
   ratings into Detail tables, Aggregated tables, and Macro-scenario summary tables.

 Usage / Execution:
   Local direct execution:
     python llm_penalty_judge.py

 Input Data Dependencies:
   - Penalty catalogue reference table: 违规扣分情况.xlsx
   - Intermediate evaluation tables for each jurisdiction (Detail & Aggregated).

 Generated Output Results:
   - Updated Excel tables with 'A_官方严厉度_LLM修正' and matched penalty clauses.
   - Incremental local evaluation cache: penalty_cache.json.
   - Summary comparison file: penalty_comparison_results.json.
===============================================================================
"""
import os
import json
import re
import time
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from openai import OpenAI
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════  Configuration  ══════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(os.getenv("BASE_DIR", SCRIPT_DIR))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "4 EWM评价结果"))
VIS_DIR = OUTPUT_DIR / "Visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = OUTPUT_DIR / "penalty_cache.json"

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", os.getenv("SILICONFLOW_API_KEY", "YOUR_API_KEY")),
    base_url=os.getenv("SILICONFLOW_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
MODEL_NAME = os.getenv("RECONSTRUCT_MODEL", "qwen-plus")
MODEL_NAME_USER = os.getenv("RECONSTRUCT_MODEL", "qwen-plus")

MAX_WORKERS = 3  # Lower concurrency for stability
MAX_RETRIES = 3
TIMEOUT_SECONDS = 120

# Thread lock for safe concurrent cache updates
cache_lock = threading.Lock()

# Physical Risk Index (PRI) defaults
PRI_Dict = {
    '左转': 0.85, 
    '右转(LHT高危大弯)': 0.85, 
    '直行交叉': 0.7, 
    '变道与汇入': 0.5, 
    '同向跟车': 0.3, 
    '停车与起步': 0.1,
    '右转': 0.5,
    '左转(LHT安全小弯)': 0.5
}

# Configure matplotlib fonts
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False 

# ══════════════════════════════  Utility Functions  ════════════════════════════════════
def load_penalty_contexts():
    penalty_path = BASE_DIR / "违规扣分情况.xlsx"
    if not penalty_path.exists():
        print("⚠️ 违规扣分情况.xlsx not found. Using empty penalty context.")
        return {}
    
    df = pd.read_excel(penalty_path)
    from collections import defaultdict
    contexts = defaultdict(list)
    for _, row in df.iterrows():
        rec = {}
        if pd.notna(row.get('动作行为/违反行为')):
            rec['违规行为'] = str(row['动作行为/违反行为'])
        if pd.notna(row.get('处罚')):
            rec['处罚'] = str(row['处罚'])
            
        jur_raw = str(row.get('国家/地区', '通用')).strip()
        if rec:
            contexts[jur_raw].append(rec)
            
    return {k: json.dumps(v, ensure_ascii=False) for k, v in contexts.items()}

def parse_llm_json(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except: pass
    return None

def assign_penalty_score(rule_text, scenario_desc, penalty_context, cache):
    cache_key = f"{rule_text}_{scenario_desc}"
    import hashlib
    cache_id = hashlib.md5(cache_key.encode()).hexdigest()
    
    if cache_id in cache:
        cached_res = cache[cache_id]
        if "matched_penalty" in cached_res and "matched_penalties" not in cached_res:
            cached_res["matched_penalties"] = [cached_res.pop("matched_penalty")]
        return cached_res

    prompt = f"""You are a chief judge in a traffic court. Your task is to review the given [Traffic Regulation Article & Micro-Scenario Description] against the [Global Violation Penalty Catalogue] to identify ALL potentially applicable penalty clauses (as a single violation scenario may violate multiple rules simultaneously), and assign a severity_score between 0.0 and 1.0. Please assign the severity_score based on the HIGHEST penalty tier among all applicable clauses.

[Scoring Benchmarks (Strict Constraints)]:
1.0: Extremely severe violation involving arrest, license revocation, privilege revocation, criminal detention, or direct threat to life.
0.8: Severe violation involving license suspension, vehicle impoundment, or high penalty point deduction (e.g., >= 6 points).
0.5: Moderate violation involving fines or minor penalty point deduction (e.g., 1-3 points).
0.2: Minor violation involving warnings or required driving safety courses.
0.0: No explicit penalty provision, or advisory recommendation only.

[Input Data]:
Penalty Catalogue Reference:
{penalty_context}

Regulation to Judge:
Scenario & Maneuver: {scenario_desc}
Raw Regulation Text: {rule_text}

[Output Format]:
Must output a valid JSON object containing the following three fields:
{{
  "matched_penalties": ["Clause 1 Name (and penalty points)", "Clause 2 Name (and penalty points)"],
  "severity_score": 0.8,
  "reason": "Short legal explanation for this score, indicating which highest penalty rule determined the score"
}}"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME_USER,
                messages=[
                    {"role": "system", "content": "You are an AI traffic judge strictly returning JSON formatted results."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = parse_llm_json(content)
            
            if result and "severity_score" in result:
                result["severity_score"] = float(result.get("severity_score", 0.0))
                with cache_lock:
                    cache[cache_id] = result
                return result
            else:
                raise ValueError("JSON missing required fields or parsing failed")
                
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"❌ LLM request failed after {MAX_RETRIES} retries: {str(e)}")
                return {"matched_penalties": ["Processing Failed"], "severity_score": 1.0, "reason": str(e)}
            time.sleep(1)

# ══════════════════════════════  Main Pipeline  ════════════════════════════════════
def process_micro_topology():
    print("="*60)
    print("  🚦 Phase 3: LLM Penalty Rating & Visualization Pipeline")
    print("="*60)
    
    print("⏳ Constructing jurisdiction-grouped penalty context dictionary...")
    penalty_contexts = load_penalty_contexts()
    
    cache = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except json.JSONDecodeError as e:
            bak_path = CACHE_FILE.with_suffix('.json.bak')
            import shutil
            shutil.move(str(CACHE_FILE), str(bak_path))
            print(f"⚠️ Cache file corrupt ({e}), backed up to {bak_path.name}. Starting from clean cache.")

    print(f"📖 Loaded cache records: {len(cache)}")
    
    detail_files = list(OUTPUT_DIR.glob("*_MicroTopology_Detail.xlsx"))
    
    total_processed = 0
    new_cache_count = 0
    comparison_summary = []
    
    for file_idx, file in enumerate(detail_files, 1):
        jur = file.name.replace("_MicroTopology_Detail.xlsx", "")
        
        SKIP_JURS = ["中国", "德国", "新加坡"]
        if any(s in jur for s in SKIP_JURS):
            print(f"⏭️ Skipped [{file_idx}/{len(detail_files)}] {jur} (Already completed)")
            continue
            
        print(f"\n🏛️ Processing [{file_idx}/{len(detail_files)}] {jur} ...")
        
        matched_keys = [k for k in penalty_contexts.keys() if k in jur or jur in k]
        if matched_keys:
            penalty_context = penalty_contexts[matched_keys[0]]
        else:
            all_recs = []
            for v in penalty_contexts.values():
                try: all_recs.extend(json.loads(v))
                except: pass
            penalty_context = json.dumps(all_recs, ensure_ascii=False) if all_recs else "No external penalty catalogue"
            
        df = pd.read_excel(file)
        
        tasks = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for idx, row in df.iterrows():
                scene = f"Macro Scenario: {row.get('宏观场景', '/')}, Interactive Entity: {row.get('交互对象(条件)', '/')}, Action: {row.get('法定约束动作', '/')}"
                rule_text = row.get('原始法规文本', row.get('中文翻译', 'Unknown Regulation'))
                
                future = executor.submit(assign_penalty_score, rule_text, scene, penalty_context, cache)
                tasks.append((idx, future, rule_text, scene))
                
            scores, reasons, matched = [], [], []
            
            for task_i, (idx, future, r_text, s_desc) in enumerate(tasks, 1):
                try:
                    res = future.result(timeout=120)
                except Exception as fut_err:
                    print(f"⚠️ Task {task_i}/{len(tasks)} timed out or failed: {fut_err}")
                    res = {"matched_penalties": ["Timeout"], "severity_score": 1.0, "reason": str(fut_err)}
                
                if task_i % 10 == 0 or task_i == len(tasks):
                    print(f"   📊 Progress: {task_i}/{len(tasks)} ({task_i*100//len(tasks)}%)")
                
                s_score = res.get("severity_score", 1.0)
                s_reason = res.get("reason", "")
                
                if "matched_penalties" in res:
                    pens = res["matched_penalties"]
                    s_match = " | ".join(pens) if isinstance(pens, list) else str(pens)
                else:
                    s_match = res.get("matched_penalty", "None")
                
                scores.append(s_score)
                reasons.append(s_reason)
                matched.append(s_match)
                
                comparison_summary.append({
                    "jurisdiction": jur,
                    "scenario": s_desc,
                    "original_rule": r_text,
                    "matched_penalty": s_match,
                    "severity_score": s_score,
                    "reason": s_reason
                })
                
                new_cache_count += 1
                if new_cache_count % 20 == 0:
                    temp_file = CACHE_FILE.with_suffix('.json.tmp')
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(cache, f, ensure_ascii=False, indent=2)
                    os.replace(temp_file, CACHE_FILE)
                        
        df["A_官方严厉度_LLM修正"] = scores
        df["A_官方严厉度_解释"] = reasons
        df["A_官方严厉度_匹配条款"] = matched
        
        df.to_excel(file, index=False)
        print(f"✅ {jur} update completed ({len(df)} records)")
        
        print(f"📊 Aggregating macro features for {jur}...")
        macro_file = OUTPUT_DIR / f"{jur}_MacroScenario_RCI.xlsx"
        if macro_file.exists():
            macro_df = pd.read_excel(macro_file)
            
            agg_df = df.groupby("宏观场景")["A_官方严厉度_LLM修正"].mean().reset_index()
            agg_df.rename(columns={'A_官方严厉度_LLM修正': 'A_官方严厉度_LLM修正_均值'}, inplace=True)
            
            drop_cols = [c for c in agg_df.columns if c in macro_df.columns and c != "宏观场景"]
            if drop_cols:
                macro_df = macro_df.drop(columns=drop_cols)
            
            macro_df = macro_df.merge(agg_df, on="宏观场景", how="left")
            macro_df.to_excel(macro_file, index=False)
            
        agg_file = OUTPUT_DIR / f"{jur}_MicroTopology_Aggregated.xlsx"
        if agg_file.exists():
            print(f"📊 Updating micro-aggregated features for {jur}...")
            
            def agg_penalties(grp):
                pens = [str(x) for x in grp["A_官方严厉度_匹配条款"].dropna().unique() if str(x).strip() and str(x) != "无"]
                reasons = [str(x) for x in grp["A_官方严厉度_解释"].dropna().unique() if str(x).strip()]
                return pd.Series({
                    "包含的惩罚严厉度_均值": grp["A_官方严厉度_LLM修正"].mean(),
                    "包含的惩罚条款": " | ".join(sorted(pens)) if pens else "无",
                    "包含的惩罚解释": " | ".join(sorted(reasons)) if reasons else "无"
                })
                
            std_keys = ["宏观场景", "自车动作(拓扑)", "交互对象(条件)", "他车动作(拓扑)", "信控类型"]
            for k in std_keys: df[k] = df[k].fillna("/")
            std_agg = df.groupby(std_keys).apply(agg_penalties).reset_index()
            
            no_ego_keys = ["宏观场景", "交互对象(条件)", "他车动作(拓扑)", "信控类型"]
            no_ego_agg = df.groupby(no_ego_keys).apply(agg_penalties).reset_index()
            
            try:
                std_df = pd.read_excel(agg_file, sheet_name="标准拓扑聚合")
                no_ego_df = pd.read_excel(agg_file, sheet_name="无自车动作聚合")
                
                for col in ["包含的惩罚严厉度_均值", "包含的惩罚条款", "包含的惩罚解释"]:
                    if col in std_df.columns: std_df.drop(columns=[col], inplace=True)
                    if col in no_ego_df.columns: no_ego_df.drop(columns=[col], inplace=True)
                    
                std_df = std_df.merge(std_agg, on=std_keys, how="left")
                no_ego_df = no_ego_df.merge(no_ego_agg, on=no_ego_keys, how="left")
                
                with pd.ExcelWriter(agg_file) as writer:
                    std_df.to_excel(writer, sheet_name="标准拓扑聚合", index=False)
                    no_ego_df.to_excel(writer, sheet_name="无自车动作聚合", index=False)
            except Exception as e:
                print(f"⚠️ Failed to update {agg_file.name}: {e}")
        
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        
    summary_path = OUTPUT_DIR / "penalty_comparison_results.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_summary, f, ensure_ascii=False, indent=2)
        
    print(f"\n📂 Penalty comparison summary exported to: {summary_path.name}")
    print("\n🎉 LLM Penalty evaluation completed for all jurisdictions!")
    return


if __name__ == "__main__":
    process_micro_topology()

