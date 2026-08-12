"""
===============================================================================
 Autonomous Driving Traffic Rules Knowledge Graph Engine — Phase 2b: Micro-Topology Aggregation & Objective Weighting via EWM-RCI
 Phase 2: Micro-Topology Aggregation & Objective Weighting via EWM-RCI
===============================================================================
 Core Functionality:
   Performs deep topological key folding and pure non-ego context mapping (Aggregated)
   on top of reconstructed rules. Automates state divergence detection across priority
   ownership (Proceed/Stop) to identify potential deadlocks.
   Applies Entropy Weight Method (EWM) to objectively weight multi-indicator metrics,
   generating Rule Challenge Index (RCI) at micro/macro levels, and invoking the RFI
   pipeline to compute joint complexity features.

 Usage / Execution:
   Supports modular execution modes:
     python ewm_rci_calculator.py --mode full          # Phase 1 reconstruction + Phase 2 scoring
     python ewm_rci_calculator.py --mode reconstruct   # Trigger Phase 1 atomic reconstruction only
     python ewm_rci_calculator.py --mode evaluate      # Run EWM-RCI calculation on existing rules

 Input Data Dependencies:
   - Intermediate dataset file generated during reconstruction (e.g., Reconstructed_Rules.csv/.xlsx).

 Generated Output Results:
   - *_MicroTopology_Aggregated.xlsx with standard topology and non-ego aggregation sheets.
   - Macro-scenario level comprehensive index tables *_MacroScenario_RCI.xlsx.
===============================================================================
"""
import sys, os, re, warnings, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from rfi_calculator import run_rfi_pipeline

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception: pass
warnings.filterwarnings("ignore")

# ══════════════════════════════  Configuration  ══════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(os.getenv("BASE_DIR", SCRIPT_DIR))
INPUT_DIR = Path(os.getenv("INPUT_DIR", BASE_DIR / "3 打分表格"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "4 EWM评价结果"))
RECONSTRUCTED_XLSX = OUTPUT_DIR / "Reconstructed_Rules.xlsx"

STANDARD_SCENARIOS = ["左转","右转","直行交叉","同向跟车","变道与汇入","停车与起步"]

TOPO_KEY_COLS = [
    "宏观场景", "自车动作(拓扑)",
    "交互对象(条件)", "他车动作(拓扑)", "信控类型",
]
SCORE_COLS = [
    "A_法理严厉度","B_模糊参数数量_归一化","B_模糊深度",
    "C_路权清晰度","C_特例抢占","D_拓扑复杂度",
]
SCENARIO_AGG = {
    "A_法理严厉度":"max", "B_模糊参数数量_归一化":"mean",
    "B_模糊深度":"max", "C_路权清晰度":"mean",
    "C_特例抢占":"max", "C_状态发散度":"max", "D_拓扑复杂度":"max",
}
EWM_FEATURES = [
    "B_模糊深度","B_模糊参数数量_归一化",
    "C_路权清晰度","C_特例抢占","C_状态发散度","D_拓扑复杂度",
]
LHT_COUNTRIES = ["英国","日本","澳大利亚","新加坡"]

# ══════════════════════════════  LHT Mirroring  ════════════════════════════════════
def apply_lht_mirror(df, jurisdiction):
    base = jurisdiction.split("-")[0]
    is_lht = base in LHT_COUNTRIES
    df["E_通行制式"] = "LHT" if is_lht else "RHT"
    if is_lht:
        df.loc[df["宏观场景"] == "左转", "宏观场景"] = "左转(LHT安全小弯)"
        df.loc[df["宏观场景"] == "右转", "宏观场景"] = "右转(LHT高危大弯)"
    return df

def extract_fuzzy_dims(grp, row_dict):
    import re
    from collections import defaultdict
    fuzzies = grp["提取的模糊参数"].dropna().astype(str)
    dim_map = defaultdict(set)
    for f in fuzzies:
        if not f.strip() or f.strip() in ("/", "nan", "无"): continue
        parts = re.split(r"<br>|\n", f)
        for p in parts:
            match = re.match(r"(.*?)\s*\[(.*?)\]", p.strip())
            if match:
                dim_map[f"模糊维度_{match.group(2).strip()}"].add(match.group(1).strip())
    for dim, vals in dim_map.items():
        row_dict[dim] = " | ".join(sorted(vals))

# ══════════════════════════════  Conflict Resolution  ════════════════════════════════════
def _conflict_score(grp):
    if len(grp) <= 1: return 0.0
    rv = grp["路权归属"].astype(str).str.strip().str.lower()
    has_ego = rv.str.contains("ego_优先", na=False).any()
    has_other = rv.str.contains("other_优先", na=False).any()
    has_shared = rv.str.contains("shared", na=False).any()
    if (has_ego and has_other) or (has_shared and (has_ego or has_other)):
        return 1.0
    acts = grp["法定约束动作"].astype(str).str.strip().str.lower()
    cats = set()
    for a in acts:
        if any(k in a for k in ["禁止","prohibit"]): cats.add("P")
        elif any(k in a for k in ["停车","stop"]): cats.add("S")
        elif any(k in a for k in ["通行","proceed"]): cats.add("G")
    if {"S","G"} <= cats or {"P","G"} <= cats: return 0.5
    return 0.0

def micro_topo_aggregate(df, jurisdiction, out_dir, rfi_df=None):
    for c in TOPO_KEY_COLS:
        if c not in df.columns: df[c] = "/"
        df[c] = df[c].astype(str).fillna("/").str.strip()
    for c in ["法定约束动作","路权归属","提取的模糊参数","空间上下文(条件)","原始法规文本","中文翻译"]:
        if c not in df.columns:
            if c == "中文翻译" and "包含的中文翻译" in df.columns:
                df["中文翻译"] = df["包含的中文翻译"]
            else:
                df[c] = "/"

    results, details = [], []
    for i, (key_vals, grp) in enumerate(df.groupby(TOPO_KEY_COLS, sort=False), 1):
        if not isinstance(key_vals, tuple): key_vals = (key_vals,)
        row = {c: v for c, v in zip(TOPO_KEY_COLS, key_vals)}
        row["管辖区"] = jurisdiction
        row["C_状态发散度"] = _conflict_score(grp)
        for sc in SCORE_COLS:
            row[sc] = grp[sc].max() if sc in grp.columns else 0.0
        
        for rf in ["RFI1_Norm", "RFI2_Norm", "RFI3_Norm", "RFI_Score"]:
            if rf in grp.columns:
                row[rf] = grp[rf].max()
                
        row["_合并规则数"] = len(grp)
        
        ctxs = [x for x in grp["空间上下文(条件)"].dropna().astype(str).unique() if x.strip() and x != "/"]
        row["包含的空间上下文"] = " | ".join(sorted(ctxs)) if ctxs else "/"
        
        txts = [x for x in grp["原始法规文本"].dropna().astype(str).unique() if x.strip() and x != "/"]
        row["包含的法规原文"] = " | ".join(sorted(txts)) if txts else "/"
        
        zh_col = "中文翻译" if "中文翻译" in grp.columns else ("包含的中文翻译" if "包含的中文翻译" in grp.columns else None)
        if zh_col:
            txts = [x for x in grp[zh_col].dropna().astype(str).unique() if x.strip() and x != "/"]
            row["包含的中文翻译"] = " | ".join(sorted(txts)) if txts else "/"
        else:
            row["包含的中文翻译"] = "/"
        
        extract_fuzzy_dims(grp, row)
        
        results.append(row)
        topo_id = f"TOPO-{i:03d}"
        for _, orig in grp.iterrows():
            det = {"管辖区": jurisdiction, "_拓扑编号": topo_id, "_合并规则数": len(grp)}
            for k in TOPO_KEY_COLS:
                det[k] = row[k]
            for keep in ["规则唯一ID","法规来源(条文)","原始法规文本",
                         "中文翻译","空间上下文(条件)","法定约束动作","路权归属","提取的模糊参数","_来源文件"]:
                det[keep] = orig.get(keep, "")
            det["C_状态发散度"] = row["C_状态发散度"]
            for sc in SCORE_COLS: det[sc] = row[sc]
            for rf in ["RFI1_Norm", "RFI2_Norm", "RFI3_Norm", "RFI_Score"]:
                if rf in row: det[rf] = row[rf]
            details.append(det)

    safe = jurisdiction.replace("/","_").replace("\\","_")
    pd.DataFrame(details).to_excel(
        out_dir / f"{safe}_MicroTopology_Detail.xlsx", index=False)
    print(f"  📝 Traceability detail: {safe}_MicroTopology_Detail.xlsx ({len(details)} rows)")
    return pd.DataFrame(results)

# ══════════════════════════════  Macro Aggregation  ════════════════════════════════════
def scenario_aggregate(df):
    for c in SCENARIO_AGG:
        if c not in df.columns: df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    agg = df.groupby("宏观场景", sort=False).agg(SCENARIO_AGG).reset_index()
    for c in SCENARIO_AGG: agg[c] = agg[c].round(4)
    return agg

# ══════════════════════════════  EWM  ════════════════════════════════════════
def ewm_weights(matrix):
    n, m = matrix.shape
    if n <= 1: return np.ones(m) / m
    cmin, cmax = matrix.min(0), matrix.max(0)
    d = cmax - cmin; d[d == 0] = 1.0
    normed = (matrix - cmin) / d + 1e-12
    p = normed / normed.sum(0)
    k = 1.0 / np.log(n + 1)
    e = -k * np.nansum(p * np.log(p + 1e-30), axis=0)
    diff = 1.0 - e; diff[diff < 0] = 0.0
    s = diff.sum()
    return diff / s if s > 0 else np.ones(m) / m

def compute_rci(scenario_df):
    df = scenario_df.copy()
    for c in EWM_FEATURES:
        if c not in df.columns: df[c] = 0.0
    mat = df[EWM_FEATURES].values.astype(float)
    w = ewm_weights(mat)
    cmin, cmax = mat.min(0), mat.max(0)
    d = cmax - cmin; d[d == 0] = 1.0
    normed = (mat - cmin) / d
    rci = normed @ w
    rmin, rmax = rci.min(), rci.max()
    if rmax - rmin > 1e-12: rci = (rci - rmin) / (rmax - rmin)
    df["RCI_规则挑战指数"] = np.round(rci, 4)
    df.attrs["_w"] = dict(zip(EWM_FEATURES, np.round(w, 4)))
    return df.sort_values("RCI_规则挑战指数", ascending=False).reset_index(drop=True)

# ══════════════════════════════  Phase 2 Main Pipeline  ════════════════════════════════
def run_evaluation():
    print("=" * 72)
    print("  Phase 2: EWM-RCI Evaluation Engine")
    print("=" * 72)

    if not RECONSTRUCTED_XLSX.exists():
        print(f"❌ Reconstructed result not found: {RECONSTRUCTED_XLSX}")
        print("   Please run reconstruction first: python reconstruct_engine.py")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(RECONSTRUCTED_XLSX)
    print(f"📊 Loaded reconstructed dataset: {len(df)} records")

    for c in SCORE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "_管辖区" not in df.columns:
        if "国家" in df.columns:
            df["_管辖区"] = df["国家"]
        elif "管辖区" in df.columns:
            df["_管辖区"] = df["管辖区"]
        elif "_来源文件" in df.columns:
            try:
                from reconstruct_engine import parse_jurisdiction
                df["_管辖区"] = df["_来源文件"].apply(lambda x: parse_jurisdiction(str(x)))
            except ImportError:
                df["_管辖区"] = df["_来源文件"].apply(lambda x: os.path.basename(str(x)).split('_')[0].split('-')[0])
        else:
            df["_管辖区"] = "通用"

    print("\n🔍 Computing RFI indicators and syncing to result table...")
    df["管辖区"] = df["_管辖区"]
    rfi_df = run_rfi_pipeline(df)
    rfi_cols = ["管辖区", "宏观场景", "自车动作(拓扑)", "RFI1_Norm", "RFI2_Norm", "RFI3_Norm", "RFI_Score"]
    df = pd.merge(df, rfi_df[rfi_cols], on=["管辖区", "宏观场景", "自车动作(拓扑)"], how="left")
    
    summaries = []
    for jur, gdf in df.groupby("_管辖区"):
        print(f"\n{'─'*60}\n🏛️  {jur} ({len(gdf)} records)\n{'─'*60}")
        gdf = gdf.copy()

        gdf = apply_lht_mirror(gdf, jur)
        side = gdf["E_通行制式"].iloc[0]
        print(f"  🚗 Traffic System: {side}" + (" → LHT mirror applied" if side == "LHT" else ""))

        topo = micro_topo_aggregate(gdf, jur, OUTPUT_DIR, rfi_df=rfi_df)
        print(f"  🔬 Micro-topology resolved: {len(topo)} unique topologies")
        cc = (topo["C_状态发散度"] > 0).sum()
        if cc: print(f"  ⚡ {cc} topologies show state divergence")

        topo_rci = compute_rci(topo)
        
        NO_EGO_KEYS = ["宏观场景", "交互对象(条件)", "他车动作(拓扑)", "信控类型"]
        no_ego_results = []
        for key_vals, grp in gdf.groupby(NO_EGO_KEYS, sort=False):
            if not isinstance(key_vals, tuple): key_vals = (key_vals,)
            row = {c: v for c, v in zip(NO_EGO_KEYS, key_vals)}
            row["C_状态发散度"] = _conflict_score(grp)
            for sc in SCORE_COLS:
                row[sc] = grp[sc].max() if sc in grp.columns else 0.0
            row["_合并规则数"] = len(grp)
            row["包含的自车动作"] = " | ".join(sorted(grp["自车动作(拓扑)"].dropna().astype(str).unique()))
            
            ctxs = [x for x in grp["空间上下文(条件)"].dropna().astype(str).unique() if x.strip() and x != "/"]
            row["包含的空间上下文"] = " | ".join(sorted(ctxs)) if ctxs else "/"
            
            txts = [x for x in grp["原始法规文本"].dropna().astype(str).unique() if x.strip() and x != "/"]
            row["包含的法规原文"] = " | ".join(sorted(txts)) if txts else "/"
            zh_col = "中文翻译" if "中文翻译" in grp.columns else ("包含的中文翻译" if "包含的中文翻译" in grp.columns else None)
            if zh_col:
                txts = [x for x in grp[zh_col].dropna().astype(str).unique() if x.strip() and x != "/"]
                row["包含的中文翻译"] = " | ".join(sorted(txts)) if txts else "/"
            else:
                row["包含的中文翻译"] = "/"
            
            for rf in ["RFI1_Norm", "RFI2_Norm", "RFI3_Norm", "RFI_Score"]:
                if rf in grp.columns:
                    row[rf] = grp[rf].max()
            
            extract_fuzzy_dims(grp, row)
            no_ego_results.append(row)
            
        no_ego_df = pd.DataFrame(no_ego_results)
        no_ego_rci = compute_rci(no_ego_df)
        
        safe = jur.replace("/","_").replace("\\","_")
        agg_path = OUTPUT_DIR / f"{safe}_MicroTopology_Aggregated.xlsx"
        with pd.ExcelWriter(agg_path) as writer:
            topo_rci.to_excel(writer, sheet_name="标准拓扑聚合", index=False)
            no_ego_rci.to_excel(writer, sheet_name="无自车动作聚合", index=False)
        print(f"  📝 Aggregated topology exported: {agg_path.name} (2 sheets)")

        sc = scenario_aggregate(topo)
        print(f"  🎯 Macro scenarios: {len(sc)}")

        result = compute_rci(sc)
        w = result.attrs.get("_w", {})
        if w:
            print("  ⚖️  EWM Weights:")
            for f, v in w.items(): print(f"      {f}: {v:.4f}")

        out_cols = ["宏观场景","A_法理严厉度"] + EWM_FEATURES + ["RCI_规则挑战指数"]
        out_cols = [c for c in out_cols if c in result.columns]
        result = result[out_cols]

        safe = jur.replace("/","_").replace("\\","_")
        rci_path_xlsx = OUTPUT_DIR / f"{safe}_MacroScenario_RCI.xlsx"
        result.to_excel(rci_path_xlsx, index=False)
        print(f"  💾 Saved: {rci_path_xlsx.name}")
        print("  📋 Results:")
        for ln in result.to_string(index=False).split("\n"):
            print(f"      {ln}")

        summaries.append({"Jurisdiction": jur, "System": side, "Scenarios": len(result)})

    print(f"\n{'='*72}\n📊 Evaluation Complete\n{'='*72}")
    print(pd.DataFrame(summaries).to_string(index=False))

# ══════════════════════════════  CLI Entry Point  ════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Autonomous Driving Traffic Rule Evaluation Engine V4")
    parser.add_argument("--mode", choices=["reconstruct","evaluate","full"],
                        default="full", help="Execution mode: reconstruct | evaluate | full")
    args = parser.parse_args()

    if args.mode in ("reconstruct", "full"):
        from reconstruct_engine import main as reconstruct_main
        reconstruct_main()

    if args.mode in ("evaluate", "full"):
        run_evaluation()

if __name__ == "__main__":
    main()
