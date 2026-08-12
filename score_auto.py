"""
===============================================================================
 Autonomous Driving Traffic Rules Knowledge Graph Engine — Phase 2a: Automated Scoring
 Phase 2a: Automated Indicator Scoring across Semantic and Topology Dimensions
===============================================================================
 Core Functionality:
   As the core quantitative module of the evaluation framework, this script parses reconstructed
   rule base files and evaluates each rule across semantic fuzziness depth (B), right-of-way
   clarity and preemption (C), and topological complexity (D) using regular expressions and keyword
   dictionaries. Includes fallback probe flags for manual auditing when edge conditions occur.

 Usage / Execution:
   Imported as module by external scripts:
     from score_auto import score_rules
     score_rules("Input_Rules.xlsx", "Scored_Rules.xlsx")

 Input Data Dependencies:
   - Rule dataset matrix reconstructed in Phase 1 (e.g., Reconstructed_Rules.xlsx).

 Generated Output Results:
   - Scored rule files with detailed indicator scores and audit probe flags.
===============================================================================
"""
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

def score_rules(input_csv: str, output_csv: str):
    print(f"🔄 Reading rule database: {input_csv}")
    df = pd.read_excel(input_csv)
    
    scored_data = []

    for index, row in df.iterrows():
        needs_review = False
        review_reasons = []
        
        # ---------------------------------------------------------
        # Dimension A: Legal Severity (A_Legal_Severity)
        # ---------------------------------------------------------
        raw_text = str(row.get('原始法规文本', ''))
        zh_text = str(row.get('中文翻译', ''))
        action_text = str(row.get('法定约束动作', ''))
        
        text = raw_text + " | " + zh_text + " | " + action_text
        
        A_score = 0.0
        
        if text.strip() in ("", "/", "|", "||", "|  |"):
            A_score = 0.0
        elif any(w in text for w in ['禁止', '不得', '严禁', '必须', '确保', '停车(Stop)']):
            A_score = 1.0
        elif any(w in text for w in ['让行', '避让', '妨碍', '依次', '交替', '保持车距']):
            A_score = 0.6
        elif any(w in text for w in ['注意观察', '减速', '提示', '文明', '应当', '谨慎']):
            A_score = 0.3
        else:
            A_score = 0.0  

        # ---------------------------------------------------------
        # Dimension B1: Fuzziness Parameter Count (B_Fuzziness_Count_Norm)
        # ---------------------------------------------------------
        fuzzy_text = str(row.get('提取的模糊参数', ''))
        if pd.isna(row.get('提取的模糊参数')) or "无模糊参数" in fuzzy_text or fuzzy_text.strip() == "":
            b_count_raw = 0
        else:
            b_count_raw = len(re.findall(r'\[.*?\]', fuzzy_text))
            
        B1_score = min(b_count_raw / 5.0, 1.0)

        # ---------------------------------------------------------
        # Dimension B2: Fuzziness Depth (B_Fuzziness_Depth)
        # ---------------------------------------------------------
        if b_count_raw == 0:
            B2_score = 0.0
        elif any(w in fuzzy_text for w in ["行为", "动作", "意图", "风险", "安全","程度"]):
            B2_score = 1.0
        elif any(w in fuzzy_text for w in ["距离", "速度", "时间", "环境", "空间"]):
            B2_score = 0.5
        else:
            B2_score = 0.5
            needs_review = True
            review_reasons.append("Unidentified fuzzy dimension in B2")

        # ---------------------------------------------------------
        # Dimension C1: Right-of-Way Clarity (C_Right_of_Way_Clarity)
        # ---------------------------------------------------------
        control_text = str(row.get('信控类型', ''))
        if control_text.strip() in ("", "/"):
            C1_score = 0.0
        elif any(w in control_text for w in ["无信控", "无限制", "环岛", "任意", "所有"]):
            C1_score = 1.0
        elif any(w in control_text for w in ["标志", "标线", "交警","人工","指挥","铁道"]):
            C1_score = 0.5
        elif any(w in control_text for w in ["信号灯", "信控","信号控制"]):
            C1_score = 0.0
        else:
            C1_score = 1.0
            needs_review = True
            review_reasons.append(f"Unknown signal control in C1 ({control_text})")

        # ---------------------------------------------------------
        # Dimension C2: Special Preemption & Heterogeneity (C_Exception_Preemption)
        # ---------------------------------------------------------
        entity_text = str(row.get('交互对象(条件)', ''))
        
        vru_special_keywords = ['优先通行的一方','行人', '自行车', '脚踏车','骑士','非机动车', '警车','残疾人','儿童','学童', '救护车', '消防车', '校车', '弱势群体', '特殊车辆', '动物', '盲人', '手推车', '畜力车', '残疾人机动轮椅车', '乘客','骑行']
        
        general_veh_keywords = [
            '机动车', '对向来车', '其他车辆', '各方交通流', '前车', '后车', '被超车辆', '公交车','巴士',
            '来车', '电车','车队','车辆', '行驶', '无具体对象', '地形', '地貌', '障碍物', '设施', '环境',
            '路肩', '路界', '护栏', '隔离带', '信号','潜在对象','阻塞车流','其他车辆','铁路', '列车', '有轨',
            '交通流', '车流', '使用者', '驾驶员', '驾驶人', '迎面', '驶来', '侧向', '后方', '潜在', '目标车道',
            '边界', '禁区', '实线', '停止线', '拓扑', '中心线', '中线', '边缘','警员','交通指挥','交警'
        ]
        
        if entity_text.strip() in ("", "/"):
            C2_score = 0.0
        elif ("无信控" in control_text) and any(w in entity_text for w in vru_special_keywords):
            C2_score = 1.0
        elif any(w in entity_text for w in vru_special_keywords):
            C2_score = 0.5
        elif any(w in entity_text for w in general_veh_keywords):
            C2_score = 0.3
        else:
            C2_score = 0.3
            needs_review = True
            review_reasons.append(f"Unknown interactive entity in C2 ({entity_text})")

        # ---------------------------------------------------------
        # Dimension D: Topological Complexity (D_Topological_Complexity)
        # ---------------------------------------------------------
        ego = str(row.get('自车动作(拓扑)', ''))
        other = str(row.get('他车动作(拓扑)', ''))
        combined = ego + other
        
        if not combined.strip() or combined.strip() == "/":
            D_score = 0.0
        elif any(w in combined for w in ['左转', '右转', '横穿', '交叉', '掉头', '转弯','任何','任意']):
            D_score = 1.0
        elif any(w in combined for w in ['变道', '超车', '汇入', '倒车', '借道', '盲区', '逼近']):
            D_score = 0.6
        elif any(w in combined for w in ['跟车', '起步', '进入','通行','直行', '会车', '行驶','驶入','避让','通过', '停车', '滑行', '失控', '不适用', '任意', '鸣喇叭', '灯光', '描述性', '泊车', '排队', '拥堵', '停放', '停用']):
            D_score = 0.2
        else:
            D_score = 0.6
            needs_review = True
            review_reasons.append(f"Abnormal topological action in D ({combined})")
            
        # ---------------------------------------------------------
        # C1 Sub-dimension: Right-of-Way Ownership Uncertainty
        # ---------------------------------------------------------
        priority_owner_text = str(row.get('路权归属', '')).lower()
        C1_unclear_score = 1.0 if "unclear" in priority_owner_text else 0.0

        scored_data.append({
            'A_法理严厉度': A_score,
            'B_模糊参数数量_原始': b_count_raw,
            'B_模糊参数数量_归一化': B1_score,
            'B_模糊深度': B2_score,
            'C_路权清晰度': C1_score,
            'C_路权归属': C1_unclear_score,
            'C_特例抢占': C2_score,
            'D_拓扑复杂度': D_score,
            '⚠打分需人工核查': "是" if needs_review else "否",
            '打分核查原因': " | ".join(review_reasons) if needs_review else ""
        })

    score_df = pd.DataFrame(scored_data)
    final_df = pd.concat([df, score_df], axis=1)
    
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    final_df.to_excel(output_csv, index=False)
    print(f"✅ Feature scoring completed! Successfully processed {len(final_df)} rules.")
    print(f"⚠️ Found {len(final_df[final_df['⚠打分需人工核查'] == '是'])} rules triggering fallback probes requiring review.")
    print(f"💾 Saved feature matrix data table to: {output_csv}\n")
    
    cols_to_show = [
        '规则唯一ID', '宏观场景', '原始法规文本', 
        'A_法理严厉度', 'B_模糊参数数量_归一化', 'B_模糊深度', 
        'C_路权清晰度', 'C_路权归属', 'C_特例抢占', 'D_拓扑复杂度', 
        '⚠打分需人工核查', '打分核查原因'
    ]
    
    valid_cols = [c for c in cols_to_show if c in final_df.columns]
    
    print("🔍 Data Preview (First 3 rows):")
    print(final_df[valid_cols].head(3))


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    BASE_DIR = Path(os.getenv("BASE_DIR", SCRIPT_DIR))
    input_path = Path(os.getenv("INPUT_FILE", BASE_DIR / "2 校对后的评估表格" / "美国-亚利桑那州-code.xlsx"))
    output_path = Path(os.getenv("OUTPUT_FILE", BASE_DIR / "3 打分表格" / "美国-亚利桑那州-code_打分.xlsx"))
    
    score_rules(str(input_path), str(output_path))
