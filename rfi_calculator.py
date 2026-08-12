"""
===============================================================================
 Autonomous Driving Traffic Rules Knowledge Graph Engine — Phase 2d: Rule Friction Index (RFI) Calculator Pipeline
 Phase 3: Rule Friction Index (RFI) Evaluation via Graph Density & Entropy
===============================================================================
 Core Functionality:
   Quantifies game-theoretic congestion and friction probabilities between interactive entities
   and spatial ontologies within the same macro scenario. Abstracting micro rules as a NetworkX
   undirected graph to calculate rule coupling density (RFI2), priority entropy based on right-of-way
   distribution (RFI3), and absolute rule volume (RFI1), executing Entropy Weight Method (EWM) per
   jurisdiction to generate the composite Rule Friction Index (RFI).

 Usage / Execution:
   Standalone execution (default loads reconstructed rules CSV):
     python rfi_calculator.py
   Specify custom input table path:
     python rfi_calculator.py "<custom_table_path>"
   Imported as module by the main pipeline:
     from rfi_calculator import run_rfi_pipeline

 Input Data Dependencies:
   - Micro rule table containing macro scenario, topology action, spatial context, and interactive entity columns.

 Generated Output Results:
   - Scenario_RFI_Weights.csv containing jurisdiction-level EWM weights.
   - Scenario_RFI_Evaluation.csv containing normalized scores and composite friction ratings.
===============================================================================
"""
import pandas as pd
import numpy as np
import networkx as nx
import math
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception: pass

# ══════════════════════════════  Configuration  ══════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(os.getenv("BASE_DIR", SCRIPT_DIR))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "4 EWM评价结果"))

# Weighted Coupling Parameters
ALPHA_ENTITY = 0.35
ALPHA_ACTION = 0.35
ALPHA_SPATIAL = 0.30
# ════════════════════════════════════════════════════════════════════════════

# 3. Spatial Ontology Dictionary
SPATIAL_ONTOLOGY = {
    "INTERSECTION": ["交叉路口", "十字路口", "四向交叉口", "交叉口", "路口", "交界处", "交差点", "错列", "盲区路口"],
    "T_JUNCTION": ["丁字路口", "T型", "终止道路", "Y形路口", "分岔", "汇流"],
    "ROUNDABOUT": ["环形路口", "环岛", "交通圈", "圆环", "roundabout"],
    "RAIL_CROSSING": ["平交道口", "铁路", "轨道", "踏切", "level crossing"],
    "OVERPASS_UNDERPASS": ["立交桥", "高架", "菱形立交", "地下通道", "桥梁", "拱桥", "窄桥", "隧道"],
    "ROAD_TYPES": ["高速", "快速路", "主干道", "主路", "辅路", "支路", "侧街", "单行道", "单向", "双向", "双程", "多车道", "单车道", "窄路"],
    "TOPOGRAPHY": ["坡路", "陡坡", "上坡", "下坡", "坡顶", "山顶", "弯道", "急弯"],
    "LANES": ["车道", "通行带", "左侧车道", "右侧车道", "中间车道", "专用道", "公交", "有轨电车", "左转专用", "右转专用", "导向车道", "双向左转", "可变车道", "滑行道", "分流车道", "导流车道", "加减速车道", "转向湾"],
    "DIVIDERS": ["隔离带", "分隔带", "中心隔离", "分隔带开口", "中心线", "双黄实线", "双白线", "路缘", "路边", "路侧带", "路肩", "草地边缘", "安全岛", "导流岛"],
    "RESTRICTED": ["网格区", "网状线", "黄色方格", "黄格", "禁入区", "禁停区", "减速带", "冰雪", "积水", "泥泞"],
    "VRU_ZONES": ["人行横道", "斑马线", "行人穿越", "过街处", "人行通道", "步道", "人行道", "盲道", "非机动车道", "自行车道", "自行车街", "共享区域", "边缘区域", "停车位", "私人车道", "相邻土地"],
    "SIGN_ZONE": ["标志", "禁止", "仅限", "必须", "允许掉头", "掉头", "U-turn"]
}

# VRU & Motor Vehicle Sets (for entity similarity fallback)
VRU_SET = {"行人", "非机动车(含自行车)", "VRU"}
MV_SET   = {"机动车", "特殊车辆(警车/救护车/校车)", "特殊车辆(警车/救护车)", "特殊车辆"}

# ── Action Similarity LUT ──────────────────────────────────────────────────
ACTION_SCORE = {
    "直行":       0.85,
    "横穿":       0.90,
    "左转/掉头":  0.80,
    "右转":       0.75,
    "换道":       0.75,
    "汇入":       0.80,
    "跟车":       0.70,
    "停车":       0.20,
    "起步":       0.20,
    "其他需检查": 0.10,
    "/":          0.00,
}

_ACTION_LUT = {
    "直行":       {"直行":0.90, "横穿":0.95, "左转/掉头":0.85, "右转":0.80, "换道":0.80, "汇入":0.85, "跟车":0.85, "停车":0.15, "起步":0.15, "其他需检查":0.10, "/":0.00},
    "横穿":       {"直行":0.95, "横穿":0.90, "左转/掉头":0.90, "右转":0.90, "换道":0.85, "汇入":0.85, "跟车":0.85, "停车":0.15, "起步":0.15, "其他需检查":0.10, "/":0.00},
    "左转/掉头":  {"直行":0.85, "横穿":0.90, "左转/掉头":0.80, "右转":0.70, "换道":0.70, "汇入":0.75, "跟车":0.75, "停车":0.15, "起步":0.15, "其他需检查":0.10, "/":0.00},
    "右转":       {"直行":0.80, "横穿":0.90, "左转/掉头":0.70, "右转":0.75, "换道":0.75, "汇入":0.80, "跟车":0.75, "停车":0.15, "起步":0.15, "其他需检查":0.10, "/":0.00},
    "换道":       {"直行":0.80, "横穿":0.85, "左转/掉头":0.70, "右转":0.75, "换道":0.75, "汇入":0.80, "跟车":0.80, "停车":0.15, "起步":0.15, "其他需检查":0.10, "/":0.00},
    "汇入":       {"直行":0.85, "横穿":0.85, "左转/掉头":0.75, "右转":0.80, "换道":0.80, "汇入":0.80, "跟车":0.80, "停车":0.15, "起步":0.15, "其他需检查":0.10, "/":0.00},
    "跟车":       {"直行":0.85, "横穿":0.85, "左转/掉头":0.75, "右转":0.75, "换道":0.80, "汇入":0.80, "跟车":0.70, "停车":0.20, "起步":0.20, "其他需检查":0.10, "/":0.00},
    "停车":       {"直行":0.15, "横穿":0.15, "左转/掉头":0.15, "右转":0.15, "换道":0.15, "汇入":0.15, "跟车":0.20, "停车":0.20, "起步":0.50, "其他需检查":0.05, "/":0.00},
    "起步":       {"直行":0.15, "横穿":0.15, "左转/掉头":0.15, "右转":0.15, "换道":0.15, "汇入":0.15, "跟车":0.20, "停车":0.50, "起步":0.20, "其他需检查":0.05, "/":0.00},
    "其他需检查": {"直行":0.10, "横穿":0.10, "左转/掉头":0.10, "右转":0.10, "换道":0.10, "汇入":0.10, "跟车":0.10, "停车":0.05, "起步":0.05, "其他需检查":0.10, "/":0.00},
    "/":          {"直行":0.00, "横穿":0.00, "左转/掉头":0.00, "右转":0.00, "换道":0.00, "汇入":0.00, "跟车":0.00, "停车":0.00, "起步":0.00, "其他需检查":0.00, "/":0.00},
}

INVALID_ACTIONS = {"/", "", "不适用", "其他需检查"}
INVALID_TARGETS = {"/", "", "无具体对象", "道路环境/设施", "环境与死物"}

# ── Entity Similarity ─────────────────────────────────────────────────────────
def get_entity_similarity(r1, r2):
    """
    Returns entity coupling weight between two rules.
    Exact match -> 1.0; Same category (VRU↔VRU / MV↔MV) -> 0.8; Else -> 0.0
    """
    invalid = {"", "无具体对象", "道路环境/设施", "环境与死物"}
    t1 = str(r1.get("交互对象(条件)", "")).strip()
    t2 = str(r2.get("交互对象(条件)", "")).strip()

    if not t1 or not t2 or t1 in invalid or t2 in invalid:
        return 0.0
    if pd.isna(r1.get("交互对象(条件)")) or pd.isna(r2.get("交互对象(条件)")):
        return 0.0

    if t1 == t2:
        return 1.0

    t1_vru = t1 in VRU_SET
    t2_vru = t2 in VRU_SET
    t1_mv  = t1 in MV_SET
    t2_mv  = t2 in MV_SET
    if (t1_vru and t2_vru) or (t1_mv and t2_mv):
        return 0.8
    return 0.0


# ── Action Similarity (LUT) ─────────────────────────────────────────────────
def get_action_similarity(r1, r2):
    """
    Looks up action coupling weight between two rules.
    Trajectory conflict actions -> high value (0.70–0.95); Static rules -> low value (0.0–0.2)
    """
    a1 = str(r1.get("他车动作(拓扑)", "")).strip()
    a2 = str(r2.get("他车动作(拓扑)", "")).strip()

    if a1 in INVALID_ACTIONS or a2 in INVALID_ACTIONS:
        return 0.0
    if pd.isna(r1.get("他车动作(拓扑)")) or pd.isna(r2.get("他车动作(拓扑)")):
        return 0.0

    row = _ACTION_LUT.get(a1, {})
    return row.get(a2, 0.0)


# ── Spatial Jaccard Similarity ───────────────────────────────────────────────
def get_spatial_clusters(context_str):
    if pd.isna(context_str) or not isinstance(context_str, str):
        return set()
    active_clusters = set()
    for cluster, keywords in SPATIAL_ONTOLOGY.items():
        if any(kw in context_str for kw in keywords):
            active_clusters.add(cluster)
    return active_clusters


def get_spatial_jaccard(r1, r2):
    """
    Jaccard(Ω(r_u), Ω(r_v)) = |∩| / |∪|
    """
    s1 = get_spatial_clusters(r1.get("空间上下文(条件)", ""))
    s2 = get_spatial_clusters(r2.get("空间上下文(条件)", ""))
    if not s1 or not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


# ── Core: Weighted RFI2 Density ──────────────────────────────────────────────
def calculate_rfi2_density(group):
    """
    Weighted rule coupling density:
    W_uv = α1·Sim_entity + α2·Sim_action + α3·Jaccard(Ω)
    RFI2 = Σ_{u<v} W_uv / [N(N-1)/2]  ∈ [0, 1]
    """
    N = len(group)
    if N <= 1:
        return 0.0

    rules = group.to_dict('records')
    num_pairs = N * (N - 1) // 2
    total_weight = 0.0

    for i in range(N):
        for j in range(i + 1, N):
            r1, r2 = rules[i], rules[j]
            w_entity  = get_entity_similarity(r1, r2)
            w_action  = get_action_similarity(r1, r2)
            w_spatial = get_spatial_jaccard(r1, r2)
            w_uv = ALPHA_ENTITY * w_entity + ALPHA_ACTION * w_action + ALPHA_SPATIAL * w_spatial
            total_weight += w_uv

    return total_weight / num_pairs

def calculate_shannon_entropy(labels):
    """
    Computes priority entropy (RFI3_Raw).
    """
    valid_labels = [l for l in labels if pd.notna(l) and str(l).strip() != '/']
    
    if len(valid_labels) <= 1:
        return 0.0
    
    counts = pd.Series(valid_labels).value_counts()
    probs = counts / len(valid_labels)
    entropy = -sum(p * math.log(p, 2) for p in probs)
    return entropy

def entropy_weight_method(df, cols):
    """
    Entropy Weight Method (EWM) for weighting indicators.
    """
    Y = df[cols].values
    n, m = Y.shape
    
    sum_Y = Y.sum(axis=0)
    P = Y / sum_Y
    
    ln_n = math.log(n) if n > 1 else 1.0
    E = - (1 / ln_n) * np.sum(P * np.log(P + 1e-12), axis=0)
    
    D = 1 - E
    W = D / np.sum(D)
    return W

def run_rfi_pipeline(input_data):
    """
    Main RFI pipeline: calculates metrics, EWM weights, and composite RFI score per jurisdiction.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🚀 Starting RFI calculation pipeline (per jurisdiction)...")
    
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        if not os.path.exists(input_data):
            print(f"❌ Error: File {input_data} does not exist.")
            return
        if str(input_data).endswith('.xlsx'):
            df = pd.read_excel(input_data)
        else:
            df = pd.read_csv(input_data)
            
    print(f"📊 Dataset loaded: {len(df)} rules")
    
    if "_管辖区" not in df.columns:
        if "国家" in df.columns:
            df["_管辖区"] = df["国家"]
        elif "管辖区" in df.columns:
            df["_管辖区"] = df["管辖区"]
        elif "_来源文件" in df.columns:
            def extract_jur(path):
                name = os.path.basename(str(path))
                return name.split('_')[0].split('-')[0].split(' ')[0]
            df["_管辖区"] = df["_来源文件"].apply(extract_jur)
        else:
            df["_管辖区"] = "Unknown"
    
    mask = (
        df["宏观场景"].notna() & (df["宏观场景"].astype(str).str.strip() != "") & (df["宏观场景"].astype(str).str.strip() != "/") &
        df["自车动作(拓扑)"].notna() & (df["自车动作(拓扑)"].astype(str).str.strip() != "") & (df["自车动作(拓扑)"].astype(str).str.strip() != "/")
    )
    df_clean = df[mask].copy()
    
    all_results = []
    all_weights = []
    
    for jur, jur_df in df_clean.groupby("_管辖区"):
        print(f"\n{'─'*60}\n🏛️  Processing jurisdiction: {jur} ({len(jur_df)} rules)")
        
        group_key = ["宏观场景", "自车动作(拓扑)"]
        jur_results = []
        
        for name, group in jur_df.groupby(group_key):
            N = len(group)
            if N < 2:
                continue
                
            rfi1_raw = float(N)
            rfi2_raw = calculate_rfi2_density(group)
            rfi3_raw = calculate_shannon_entropy(group["路权归属"]) if "路权归属" in group.columns else (
                calculate_shannon_entropy(group["C_路权清晰度"]) if "C_路权清晰度" in group.columns else 0.0
            )
            
            jur_results.append({
                "管辖区": jur,
                "宏观场景": name[0],
                "自车动作(拓扑)": name[1],
                "N_Rules": N,
                "RFI1_Raw": rfi1_raw,
                "RFI2_Raw": rfi2_raw,
                "RFI3_Raw": rfi3_raw
            })
        
        if not jur_results:
            print(f"⚠️  Jurisdiction {jur} has no groups matching condition (N >= 2). Skipping.")
            continue
            
        res_df = pd.DataFrame(jur_results)
        
        raw_cols = ["RFI1_Raw", "RFI2_Raw", "RFI3_Raw"]
        norm_cols = ["RFI1_Norm", "RFI2_Norm", "RFI3_Norm"]
        
        for raw, norm in zip(raw_cols, norm_cols):
            c_min = res_df[raw].min()
            c_max = res_df[raw].max()
            if c_max == c_min:
                res_df[norm] = 0.5
            else:
                res_df[norm] = 0.001 + (res_df[raw] - c_min) / (c_max - c_min) * (0.999 - 0.001)
                
        weights = entropy_weight_method(res_df, norm_cols)
        alpha, beta, gamma = weights
        
        print(f"   ⚖️  EWM Weights: Alpha={alpha:.4f}, Beta={beta:.4f}, Gamma={gamma:.4f}")
        
        all_weights.append({
            "管辖区": jur,
            "W_Alpha(Volume)": alpha,
            "W_Beta(Density)": beta,
            "W_Gamma(Entropy)": gamma
        })
        
        res_df["W_Alpha"] = alpha
        res_df["W_Beta"] = beta
        res_df["W_Gamma"] = gamma
        res_df["RFI_Score"] = alpha * res_df["RFI1_Norm"] + beta * res_df["RFI2_Norm"] + gamma * res_df["RFI3_Norm"]
        all_results.append(res_df)

    if not all_results:
        print("⚠️ Warning: No valid groups found across all jurisdictions.")
        return
    
    final_res_df = pd.concat(all_results, ignore_index=True)
    final_res_df = final_res_df.sort_values(["管辖区", "RFI_Score"], ascending=[True, False]).reset_index(drop=True)
    
    weights_path = OUTPUT_DIR / "Scenario_RFI_Weights.csv"
    pd.DataFrame(all_weights).to_csv(weights_path, index=False, encoding="utf-8-sig")
    
    output_path = OUTPUT_DIR / "Scenario_RFI_Evaluation.csv"
    final_cols = ["管辖区", "宏观场景", "自车动作(拓扑)", "N_Rules", 
                  "RFI1_Norm", "RFI2_Norm", "RFI3_Norm", 
                  "W_Alpha", "W_Beta", "W_Gamma", "RFI_Score"]
    final_res_df[final_cols].to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"\n{'='*60}\n✅ RFI Pipeline Complete!")
    print(f"💾 Score output: {output_path}")
    print(f"💾 Weight breakdown: {weights_path}")
    print(f"📊 Top 5 highest friction scenarios (global):")
    print(final_res_df[["管辖区", "宏观场景", "自车动作(拓扑)", "RFI_Score"]].head(5).to_string(index=False))
    
    return final_res_df

if __name__ == "__main__":
    default_input = OUTPUT_DIR / "Reconstructed_Rules.csv"
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = default_input
        
    run_rfi_pipeline(input_path)
