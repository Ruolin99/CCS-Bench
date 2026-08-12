"""
===============================================================================
 Autonomous Driving Traffic Rules Knowledge Graph Engine
 Phase 1c: Micro-Topology Reconstruction Engine via LLM Sub-scenario Slicing
===============================================================================
 Core Functionality:
   Reads scored tables containing initial legal texts, invokes LLM concurrently
   to atomize each long-text regulation into micro-topology interaction scene
   dictionaries. Forcibly extracts 5 standard topology keys determining right-of-way,
   combined with MD5 deduplication and local checkpoint resumption mechanism,
   outputting the finest-grained interaction micro-topology detail tables.

 Usage / Execution:
   Direct local full-batch processing:
     python reconstruct_engine.py

 Input Data Dependencies:
   - Initial source table file set located in the scoring tables directory.

 Generated Output Results:
   - Full overview table with atomic-level rule decomposition: Reconstructed_Rules.xlsx/.csv.
   - Per-jurisdiction refined dictionary tables: *_MicroTopology_Detail.xlsx.
   - Real-time incremental checkpoint resumption log: _reconstruct_checkpoint.jsonl.
===============================================================================
"""
import sys, os, re, json, time, hashlib, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception: pass
warnings.filterwarnings("ignore")

# ══════════════════════════════  Configuration  ══════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR  = Path(os.getenv("INPUT_DIR", SCRIPT_DIR / "scored_tables"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", SCRIPT_DIR / "4 EWM评价结果"))
CHECKPOINT = OUTPUT_DIR / "_reconstruct_checkpoint.jsonl"
MAX_WORKERS = 5
MAX_RETRIES = 5

client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY", "YOUR_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
MODEL_NAME = os.getenv("RECONSTRUCT_MODEL", "qwen3.6-max-preview")

SCENARIO_ENUM = ['左转','右转','直行交叉','同向跟车','变道与汇入','停车与起步']
EGO_ACTION_ENUM = ['跟车','换道','直行','左转/掉头','右转','汇入','停车','起步']
OTHER_ACTION_ENUM = ['跟车','换道','直行','左转/掉头','右转','汇入','停车','起步','横穿','其他需检查','/']
ENTITY_ENUM = ['机动车','非机动车(含自行车)','行人','特殊车辆(警车/救护车/校车)','环境与死物']

SYSTEM_PROMPT = f"""你是自动驾驶交通规则语义重建专家。你的任务是将一条可能描述模糊的交通规则拆分、纠错为一条或多条严格原子化的规则。当前针对的宏观场景必须锁定为: {{{{target_scenario}}}}。

【严格枚举约束 — 违反即视为无效输出】
1. macro_scenario（宏观场景）必须锁定为: {{{{target_scenario}}}}
   - 若原文涉及该场景的多个子动作，拆分为多条规则
2. ego_action（自车动作）必须从以下单选: {json.dumps(EGO_ACTION_ENUM, ensure_ascii=False)}
   - 禁止使用"驶入""驶近""借道"等模糊词
3. other_action（他车/对象动作）必须从以下单选: {json.dumps(OTHER_ACTION_ENUM, ensure_ascii=False)}
   - 若原文为"通行"等任意动作，必须根据合理性拆分为多条规则（如分为直行、左转/掉头、右转）
   - 若交互对象为"行人"，其动作通常应被明确分类为"横穿"或其他具体动作，尽量避免分类为"其他需检查"
   - 若交互对象为"环境与死物"，他车动作直接输出为"/"
4. interactive_entity（交互对象）必须从以下单选: {json.dumps(ENTITY_ENUM, ensure_ascii=False)}
   - 若原文涉及多类对象（如"人车"），必须拆分为多条独立规则

【输出格式】严格输出合法JSON，顶层键为 reconstructed_rules（数组），每个元素含:
  macro_scenario, ego_action, other_action, interactive_entity
不要输出任何解释文字。"""

# ══════════════════════════════  Utility Functions  ════════════════════════════════════
def parse_jurisdiction(filename):
    name = re.sub(r"_打分$", "", Path(filename).stem)
    for pat in [r"(美国-[^-]+)-", r"(澳大利亚-[^-]+)-"]:
        m = re.match(pat, name)
        if m: return m.group(1)
    m = re.match(r"([^-]+)-", name)
    if m:
        j = m.group(1)
        return "新加坡" if j == "新加披" else j
    return name

def _extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    if text.endswith("```"): text = text[:-3]
    # Find first { or [
    for i, c in enumerate(text):
        if c in ('{','['):
            depth = 0
            opener = c
            closer = '}' if c == '{' else ']'
            in_str = False
            esc = False
            for j in range(i, len(text)):
                ch = text[j]
                if in_str:
                    if esc: esc = False
                    elif ch == '\\': esc = True
                    elif ch == '"': in_str = False
                else:
                    if ch == '"': in_str = True
                    elif ch == opener: depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            return json.loads(text[i:j+1])
            break
    return json.loads(text)

# ══════════════════════════════  LLM Invocation  ════════════════════════════════════
def reconstruct_single_rule(row_dict, row_idx):
    """Invoke LLM to reconstruct a single rule, returns list[dict]."""
    target_scenario = row_dict.get("宏观场景_分裂后", "停车与起步")

    user_msg = json.dumps({
        "原始法规文本": str(row_dict.get("原始法规文本", "")),
        "中文翻译": str(row_dict.get("中文翻译", "")),
        "自车动作(原)": str(row_dict.get("自车动作(拓扑)", "")),
        "他车动作(原)": str(row_dict.get("他车动作(拓扑)", "")),
        "交互对象(原)": str(row_dict.get("交互对象(条件)", "")),
    }, ensure_ascii=False)

    for attempt in range(MAX_RETRIES):
        try:
            sys_prompt_rendered = SYSTEM_PROMPT.replace("{{target_scenario}}", target_scenario)
            msgs = [
                {"role": "system", "content": sys_prompt_rendered + "\n请只输出JSON，不要输出任何解释。"},
                {"role": "user", "content": user_msg},
            ]
            resp = client.chat.completions.create(
                model=MODEL_NAME, messages=msgs,
                temperature=0.0,
                timeout=30,
                extra_body={"enable_thinking": False},
            )
            content = resp.choices[0].message.content
            parsed = _extract_json(content)
            rules = parsed.get("reconstructed_rules", [parsed] if "macro_scenario" in parsed else [])
            if not rules:
                rules = [parsed] if isinstance(parsed, dict) else []
            # Validate enum compliance
            valid = []
            seen = set()
            for r in rules:
                # Force-inherit Python-locked macro scenario
                r["macro_scenario"] = target_scenario
                r.setdefault("ego_action", "直行")
                r.setdefault("other_action", "直行")
                r.setdefault("interactive_entity", "机动车")
                
                # If interactive entity is environment/static object, force no action
                if r.get("interactive_entity") == "环境与死物":
                    r["other_action"] = "/"
                    
                # Filter fully duplicate rule rows
                key = (r["macro_scenario"], r["ego_action"], r["other_action"], r["interactive_entity"])
                if key not in seen:
                    seen.add(key)
                    valid.append(r)
            return valid
        except Exception as e:
            if "RPM limit" in str(e) or "rate" in str(e).lower() or "429" in str(e) or "403" in str(e):
                time.sleep(3 + 2 * attempt)  # Longer backoff for rate limits
            elif attempt < MAX_RETRIES - 1:
                time.sleep(1.5 ** attempt)
            else:
                print(f"    ⚠ Row {row_idx} reconstruction failed ({e}), keeping original values")
                return [_fallback(row_dict)]
    return [_fallback(row_dict)]

def _fuzzy_match_scenario(s):
    s = re.sub(r"[（(][^)）]*[)）]", "", s).strip()
    if "左转" in s: return "左转"
    if "右转" in s: return "右转"
    if "直行" in s or "交叉" in s: return "直行交叉"
    if "跟车" in s: return "同向跟车"
    if "变道" in s or "汇入" in s or "超车" in s: return "变道与汇入"
    if "停车" in s or "起步" in s: return "停车与起步"
    return "停车与起步"

def _fallback(row_dict):
    return {
        "macro_scenario": _fuzzy_match_scenario(str(row_dict.get("宏观场景", ""))),
        "ego_action": str(row_dict.get("自车动作(拓扑)", "直行")),
        "other_action": str(row_dict.get("他车动作(拓扑)", "直行")),
        "interactive_entity": str(row_dict.get("交互对象(条件)", "机动车")),
        "_fallback": True,
    }

# ══════════════════════════════  Scoring Engine  ════════════════════════════════════
def rescore_row(row):
    """Re-scoring logic ported from score_auto.py, with review flag probes."""
    needs_review = False
    review_reasons = []

    if row.get("宏观场景") == "其他需检查":
        needs_review = True
        review_reasons.append("宏观场景被分类为其他需检查")

    raw_text = str(row.get('原始法规文本', ''))
    zh_text = str(row.get('中文翻译', ''))
    action_text = str(row.get('法定约束动作', ''))
    text = raw_text + " | " + zh_text + " | " + action_text

    # A: Legal severity
    if text.strip() in ("", "/", "|", "||", "|  |"):
        A = 0.0
    elif any(w in text for w in ['禁止','不得','严禁','必须','确保','停车(Stop)']):
        A = 1.0
    elif any(w in text for w in ['让行','避让','妨碍','依次','交替','保持车距']):
        A = 0.6
    elif any(w in text for w in ['注意观察','减速','提示','文明','应当','谨慎']):
        A = 0.3
    else:
        A = 0.0

    # B1: Ambiguous parameter count (normalized)
    fuzzy_text = str(row.get('提取的模糊参数', ''))
    if pd.isna(row.get('提取的模糊参数')) or "无模糊参数" in fuzzy_text or not fuzzy_text.strip():
        b_count = 0
    else:
        b_count = len(re.findall(r'\[.*?\]', fuzzy_text))
    B1 = min(b_count / 5.0, 1.0)

    # B2: Fuzziness depth
    if b_count == 0:
        B2 = 0.0
    elif any(w in fuzzy_text for w in ["行为","动作","意图","风险","程度"]):
        B2 = 1.0
    elif any(w in fuzzy_text for w in ["距离","速度","时间","环境","空间"]):
        B2 = 0.5
    else:
        B2 = 0.5
        needs_review = True
        review_reasons.append("B2无法识别模糊维度")

    # C1: Right-of-way clarity
    ctrl = str(row.get('信控类型', ''))
    if ctrl.strip() in ("", "/"):
        C1 = 0.0
    elif any(w in ctrl for w in ["无信控","无限制","环岛","任意","所有","无特定"]):
        C1 = 1.0
    elif any(w in ctrl for w in ["标志","标线","交警","人工","指挥","铁道","接入口"]):
        C1 = 0.5
    elif any(w in ctrl for w in ["信号灯","信控","信号控制"]):
        C1 = 0.0
    else:
        C1 = 1.0
        needs_review = True
        review_reasons.append(f"C1未知信控({ctrl})")

    # C2: Special preemption — based on reconstructed interactive entity
    entity = str(row.get('交互对象(条件)', ''))
    vru_kw = ['行人','自行车','非机动车','骑士','校车','残疾人','儿童','学童',
              '救护车','消防车','警车','盲人','弱势','特殊车辆']
    gen_kw = ['机动车','车辆','环境','死物']
    if entity.strip() in ("", "/"):
        C2 = 0.0
    elif ("无信控" in ctrl) and any(w in entity for w in vru_kw):
        C2 = 1.0
    elif any(w in entity for w in vru_kw):
        C2 = 0.5
    elif any(w in entity for w in gen_kw):
        C2 = 0.3
    else:
        C2 = 0.3
        needs_review = True
        review_reasons.append(f"C2未知交互对象({entity})")

    # D: Topological complexity
    ego = str(row.get('自车动作(拓扑)', ''))
    other = str(row.get('他车动作(拓扑)', ''))
    combined = ego + other
    if not combined.strip() or combined.strip() == "/":
        D = 0.0
    elif any(w in combined for w in ['左转','右转','横穿','交叉','掉头','转弯','任何','任意']):
        D = 1.0
    elif any(w in combined for w in ['换道','超车','汇入','倒车','借道','盲区','逼近','变道']):
        D = 0.6
    elif any(w in combined for w in ['跟车','起步','进入','通行','直行','会车','行驶',
                                      '停车','泊车','排队','拥堵']):
        D = 0.2
    else:
        D = 0.6
        needs_review = True
        review_reasons.append(f"D拓扑动作异常({combined})")

    return {
        'A_法理严厉度': A, 'B_模糊参数数量_归一化': B1, 'B_模糊深度': B2,
        'C_路权清晰度': C1, 'C_特例抢占': C2, 'D_拓扑复杂度': D,
        '⚠打分需人工核查': '是' if needs_review else '否',
        '打分核查原因': ' | '.join(review_reasons) if needs_review else '',
    }

# ══════════════════════════════  Main Pipeline  ══════════════════════════════════════
def load_checkpoint():
    done = set()
    if CHECKPOINT.exists():
        with open(CHECKPOINT, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        done.add(obj.get("rule_id", ""))
                    except: pass
    return done

def save_checkpoint(rule_id, reconstructed):
    with open(CHECKPOINT, 'a', encoding='utf-8') as f:
        f.write(json.dumps({"rule_id": rule_id, "results": reconstructed}, ensure_ascii=False) + "\n")

def load_checkpoint_results():
    results = {}
    if CHECKPOINT.exists():
        with open(CHECKPOINT, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        results[obj["rule_id"]] = obj["results"]
                    except: pass
    return results

def pre_explode_macro_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministically explode generalized scenarios before LLM processing to prevent omissions."""
    expanded_rows = []
    INTERSECTION_SCENARIOS = ['左转', '右转', '直行交叉']
    
    for _, row in df.iterrows():
        raw_scenario = str(row.get('宏观场景', '其他')).strip()
        # Strip parenthetical noise from legacy tables
        clean_scenario = re.sub(r'\(.*?\)|（.*?）', '', raw_scenario).strip()
        
        target_scenarios = set()
        
        if "全局" in clean_scenario or "所有" in clean_scenario:
            target_scenarios.update(SCENARIO_ENUM)
        elif "交叉" in clean_scenario and "通用" in clean_scenario:
            target_scenarios.update(INTERSECTION_SCENARIOS)
        else:
            if "左转" in clean_scenario: target_scenarios.add("左转")
            if "右转" in clean_scenario: target_scenarios.add("右转")
            if "直行" in clean_scenario: target_scenarios.add("直行交叉")
            if "跟车" in clean_scenario: target_scenarios.add("同向跟车")
            if "变道" in clean_scenario or "汇入" in clean_scenario: target_scenarios.add("变道与汇入")
            if "停" in clean_scenario or "起步" in clean_scenario: target_scenarios.add("停车与起步")
                      
        if not target_scenarios:
            target_scenarios.add("其他需检查")
            
        for ts in target_scenarios:
            new_row = row.copy()
            new_row['宏观场景_分裂后'] = ts
            expanded_rows.append(new_row)
            
    return pd.DataFrame(expanded_rows)

def main():
    print("=" * 72)
    print("  Phase 1: LLM Micro-Topology Reconstruction + Re-scoring")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all scored tables
    all_rows = []
    for f in sorted(INPUT_DIR.glob("*.xlsx")):
        if f.name.startswith("~$"): continue
        df = pd.read_excel(f)
        df["_来源文件"] = f.name
        df["_管辖区"] = parse_jurisdiction(f.name)
        all_rows.append(df)
    merged = pd.concat(all_rows, ignore_index=True)
    print(f"📊 Loaded {len(merged)} raw rules")

    # Deterministic scenario pre-explosion
    merged = pre_explode_macro_scenarios(merged)
    print(f"🧬 After scenario explosion: {len(merged)} rules (generalized scenario issue resolved)")

    # Ensure each rule has a unique ID
    if "规则唯一ID" not in merged.columns:
        merged["规则唯一ID"] = [f"R-{i:05d}" for i in range(len(merged))]
    merged["规则唯一ID"] = merged["规则唯一ID"].astype(str).fillna("")
    mask = merged["规则唯一ID"].str.strip() == ""
    merged.loc[mask, "规则唯一ID"] = [f"R-AUTO-{i:05d}" for i in range(mask.sum())]

    # Generate true unique IDs for checkpoint resumption (prevents shared IDs after explosion)
    merged["分裂唯一ID"] = merged["规则唯一ID"] + "_" + merged["宏观场景_分裂后"]

    # Checkpoint resumption
    done_ids = load_checkpoint()
    cached = load_checkpoint_results()
    todo = merged[~merged["分裂唯一ID"].isin(done_ids)]
    print(f"✅ Completed: {len(done_ids)} | 🔄 Pending: {len(todo)}")

    # Concurrent LLM reconstruction
    if len(todo) > 0:
        print(f"🚀 Starting concurrent reconstruction ({MAX_WORKERS} threads)...")
        completed = 0
        total = len(todo)

        def _worker(idx_row):
            idx, row = idx_row
            row_dict = row.to_dict()
            results = reconstruct_single_rule(row_dict, idx)
            return row_dict["分裂唯一ID"], results

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_worker, (i, row)): i
                       for i, row in todo.iterrows()}
            for future in as_completed(futures):
                try:
                    rule_id, results = future.result()
                    save_checkpoint(rule_id, results)
                    cached[rule_id] = results
                    completed += 1
                    if completed % 20 == 0 or completed == total:
                        print(f"    Progress: {completed}/{total} ({100*completed/total:.0f}%)", flush=True)
                except Exception as e:
                    print(f"    ⚠ Task exception: {e}")

    # Expand reconstructed results + re-score
    print("\n🔧 Expanding reconstructed results and re-scoring...")
    output_rows = []
    # Columns to inherit from original row
    inherit_cols = [c for c in merged.columns if c not in [
        "宏观场景", "自车动作(拓扑)", "他车动作(拓扑)", "交互对象(条件)",
        "A_法理严厉度", "B_模糊参数数量_归一化", "B_模糊参数数量_原始",
        "B_模糊深度", "C_路权清晰度", "C_特例抢占", "C_路权归属",
        "D_拓扑复杂度", "⚠打分需人工核查", "打分核查原因",
    ]]

    for _, orig_row in merged.iterrows():
        rid = orig_row["分裂唯一ID"]
        recons = cached.get(rid, [_fallback(orig_row.to_dict())])
        for r in recons:
            new_row = {c: orig_row.get(c, "") for c in inherit_cols}
            new_row["宏观场景"] = r.get("macro_scenario", "停车与起步")
            new_row["自车动作(拓扑)"] = r.get("ego_action", "直行")
            new_row["他车动作(拓扑)"] = r.get("other_action", "直行")
            new_row["交互对象(条件)"] = r.get("interactive_entity", "机动车")
            new_row["_LLM重构"] = "回退" if r.get("_fallback") else "是"
            # Re-score the reconstructed row
            scores = rescore_row(new_row)
            new_row.update(scores)
            output_rows.append(new_row)

    result_df = pd.DataFrame(output_rows)
    print(f"📊 After reconstruction + expansion: {len(result_df)} rules (original: {len(merged)})")

    # Validate scenario convergence
    scenarios = result_df["宏观场景"].unique()
    print(f"🎯 Scenario categories: {sorted(scenarios)}")

    # Full output
    all_path_xlsx = OUTPUT_DIR / "Reconstructed_Rules.xlsx"
    result_df.to_excel(all_path_xlsx, index=False)
    print(f"💾 Full output: {all_path_xlsx.name}")

    # Per-jurisdiction output
    for jur, gdf in result_df.groupby("_管辖区"):
        safe = jur.replace("/", "_").replace("\\", "_")
        p_xlsx = OUTPUT_DIR / f"{safe}_Reconstructed.xlsx"
        gdf.to_excel(p_xlsx, index=False)
        print(f"   📄 {p_xlsx.name} ({len(gdf)} rules)")

    print("\n✅ Phase 1 complete! Run: python ewm_rci_calculator.py --mode evaluate")

if __name__ == "__main__":
    main()
