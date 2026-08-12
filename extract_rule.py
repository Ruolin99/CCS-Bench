"""
===============================================================================
 Autonomous Driving Traffic Rules Knowledge Graph Engine — Phase 1: Batch Atomic Rule Extraction via LLM
 Phase 1: Atomic Rule Extraction via SiliconFlow LLM Services
===============================================================================
 Core Functionality:
   Batch process unstructured traffic regulation Markdown files from various jurisdictions,
   using hybrid chunking strategies to feed text into LLM services deployed on SiliconFlow (e.g., Qwen3-8B).
   Enforces Pydantic structured output of standardized rule JSON dictionaries, covering macro scenario
   classification, interaction topology, and extracted ambiguous variables.

 Usage / Execution:
   Local direct execution (defaults to reading 'traffic_rules_md' directory):
     python extract_rule.py
   Specify custom text path or rule output directory:
     python extract_rule.py --md-dir <path> --rules-dir <path>

 Input Data Dependencies:
   - Markdown file set containing raw sectioned traffic regulation articles (e.g., California_Vehicle_Code.md).

 Generated Output Results:
   - Corresponding *_compiled_semantic_rules_db.json structured rule sets.
   - Synchronously exported Excel (.xlsx) and CSV (.csv) evaluation matrices for human review.
===============================================================================
"""
import os
import sys
import glob
import json
import re
import hashlib
import argparse
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any, Tuple
try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    except ImportError:
        MarkdownHeaderTextSplitter = None
        RecursiveCharacterTextSplitter = None
from openai import OpenAI
import pandas as pd # For table exports

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception: pass

# ==========================================
# 0. Initialization
# ==========================================
# SiliconFlow: base_url requires /v1, OpenAI SDK appends /chat/completions automatically.
# Set SILICONFLOW_API_KEY environment variable in your system.
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY", "YOUR_SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
)
MODEL_NAME = os.getenv("SILICONFLOW_MODEL", "qwen3.6-plus")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MD_DIR = os.path.join(SCRIPT_DIR, "法规文件的markdown文件")
DEFAULT_DEBUG_ROOT = os.path.join(SCRIPT_DIR, "compile_debug")
DEFAULT_RULES_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "compiled_rules")
DEBUG_DIR = DEFAULT_DEBUG_ROOT
FOCUS_INTERSECTION_ONLY = True


def _safe_output_stem(filename: str) -> str:
    """Generate safe output filename stem (stripping extension and illegal path characters)."""
    base = os.path.splitext(os.path.basename(filename))[0]
    base = re.sub(r'[<>:"/\\|?*]', "_", base)
    return base.strip() or "unknown"

# ==========================================
# 1. Interaction Topology Structuring
# ==========================================
class InteractionTopology(BaseModel):
    ego_entity: Literal[
        '小型汽车(含轿车/SUV)', '大型车辆(货车/客车/公交)', '摩托车',
        '非机动车(自行车/电动车)', '行人', '通用机动车(未区分车型)',
        '特殊车辆(警车/救护车)', '其他/未明确'
    ] = Field(default='通用机动车(未区分车型)', description="Entity type of ego vehicle. Determine from raw text and section title; if uncertain, set default and set ego_entity_needs_review to True.")
    ego_maneuver: str = Field(description="Ego vehicle maneuver, e.g., 'Lane change', 'Left turn / U-turn', 'Right turn', 'Stop', 'Straight', 'Start', 'Follow', 'Merge', 'Yield', 'Other'")
    other_maneuver: str = Field(description="Interactive entity maneuver, e.g., 'Straight', 'Stop', 'Cross', 'Unrestricted', 'Right turn', 'Left turn / U-turn', 'Follow', 'Merge', 'Lane change', 'Start', 'Other'")
    control_type: Literal['无限制', '无信控', '信号灯控制', '交通标志控制', '标线控制', '交警指挥', '环岛', '铁道路口', '信号灯故障', '其他/未明确']

# ==========================================
# 2. Ambiguous Variable Semantic Modeling
# ==========================================
class AmbiguousVariable(BaseModel):
    name: str = Field(description="Extracted fuzzy word or implicit requirement (e.g., 'yielding spatial threshold', 'without impeding others')")
    
    semantic_type: Literal[
        "距离维度", "速度/加速度维度", "时间维度", "行为与动作程度","安全与风险维度", "其他/未明确"
    ] = Field(description="Physical or behavioral dimension of the fuzzy judgement")
    
    related_entities: str = Field(description="Target entities under constraint, e.g., 'ego-lead vehicle', 'ego-pedestrian'")
    applicable_scope: str = Field(description="Interaction scope where this variable applies, e.g., 'left turn - oncoming straight'")
    intended_use: str = Field(description="Algorithm purpose, e.g., 'determine yield threshold', 'evaluate risk level'")

# ==========================================
# 3. Conditions & Actions
# ==========================================
class SemanticCondition(BaseModel):
    spatial_context: str
    ego_state: str
    interactive_entity: Literal[
        '机动车', '非机动车(含自行车)', '行人', '特殊车辆', 
        '无具体对象', '警员或交通指挥人员', '其他车辆', '环境与死物', '轨道车辆'
    ]
    other_state: str

class SemanticAction(BaseModel):
    action_type: Literal[
        '停车(Stop)', '让行(Yield)', '减速(Decelerate)', '保持车距(Maintain_Distance)', 
        '不得妨碍(Do_Not_Impede)', '注意观察(Observe/Caution)', '通行(Proceed)', '其他(Other)'
    ]
    action_target: str

# ==========================================
# 4. Structured Rule Schema
# ==========================================

class StructuredSemanticRule(BaseModel):
    rule_id: str
    macro_scenario: Literal[
        '左转', '右转', '直行交叉', 
        '同向跟车', '变道与汇入', '停车与起步', '其他'
    ]
    structured_topology: InteractionTopology
    rule_category: Literal['强制性规范(Mandatory)', '倡导性建议(Advisory)']
    trigger_condition: SemanticCondition
    rule_action: SemanticAction
    ambiguous_variables: List[AmbiguousVariable]
    priority_owner: Literal['ego_优先(自车)', 'other_优先(他车/弱势群体)', 'Shared', 'unclear_法规未明确']
    original_text: str
    
    # Ego entity review flag: Set to True when LLM cannot determine ego_entity with certainty
    ego_entity_needs_review: bool = Field(default=False, description="If True, ego entity type was inferred or uncertain, requiring manual review")
    
    # Default fallbacks for tail fields
    translated_text_zh: str = Field(default="暂无翻译")
    source_path: str = Field(default="未知来源")
    is_overridden_by_global: bool = Field(default=False)
    global_exceptions_applied: Optional[str] = Field(default=None)
    
    # Debug fields: Trace back to source chunk and raw model output
    debug_chunk_index: Optional[int] = None
    debug_chunk_metadata: Optional[Dict[str, Any]] = None
    debug_raw_rule_output: Optional[Dict[str, Any]] = None

class RuleCompilationResult(BaseModel):
    semantic_rules: List[StructuredSemanticRule]

# ==========================================
# Utility Functions: Robust Normalization & Fallbacks
# ==========================================
def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def _normalize_rule_value(raw_obj: Any) -> Dict[str, Any]:
    """Perform robust normalization on model output to clean format hallucinations and generated terms."""
    if isinstance(raw_obj, list): raw_obj = {"semantic_rules": raw_obj}
    if not isinstance(raw_obj, dict): return {"semantic_rules": []}

    rules = raw_obj.get("semantic_rules", [])
    if not isinstance(rules, list): rules = []
    raw_obj["semantic_rules"] = rules
    
    def clean_literal(s: str) -> str:
        if not isinstance(s, str): return s
        s = s.replace("（", "(").replace("）", ")").replace("：", ":").replace("／", "/").replace("，", ",")
        s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        return re.sub(r"\s+", "", s)

    for rule in rules:
        # 1. Macro scenario normalization (Map LLM English keys back to dataset Chinese standard values)
        macro = clean_literal(rule.get("macro_scenario", ""))
        if "left-turn" in macro.lower():
            rule["macro_scenario"] = "左转"
        elif "right-turn" in macro.lower():
            rule["macro_scenario"] = "右转"
        elif "straight" in macro.lower():
            rule["macro_scenario"] = "直行交叉"
        elif "following" in macro.lower():
            rule["macro_scenario"] = "同向跟车"
        elif "lane" in macro.lower() or "merge" in macro.lower():
            rule["macro_scenario"] = "变道与汇入"
        elif "stop" in macro.lower() or "start" in macro.lower():
            rule["macro_scenario"] = "停车与起步"
        else:
            rule["macro_scenario"] = "其他"

        # 2. Interactive entity normalization
        trigger = rule.get("trigger_condition", {})
        if not isinstance(trigger, dict): trigger = {}
        entity = clean_literal(trigger.get("interactive_entity", ""))
        if entity in ("警员或交通指挥人员", "交通警员", "交警", "警察"): entity = "警员或交通指挥人员"
        elif "脚踏车" in entity or "自行车" in entity or "非机动" in entity: entity = "非机动车(含自行车)"
        elif "对向" in entity or "汽车" in entity or "机动车" in entity: entity = "机动车"
        elif "信号灯" in entity or "障碍" in entity or "设施" in entity or "标志" in entity or "环境" in entity or "死物" in entity: entity = "环境与死物"
        elif "轨道" in entity or "火车" in entity or "电车" in entity or "train" in entity.lower() or "tram" in entity.lower(): entity = "轨道车辆"
        elif "特殊" in entity or "警车" in entity or "救护" in entity or "消防" in entity or "校车" in entity: entity = "特殊车辆"
        elif "人" in entity: entity = "行人"
        elif entity not in ['机动车', '非机动车(含自行车)', '行人', '特殊车辆', '无具体对象', '警员或交通指挥人员', '其他车辆', '环境与死物', '轨道车辆']:
            entity = "无具体对象"
        trigger["interactive_entity"] = entity
        rule["trigger_condition"] = trigger

        # 3. Ambiguous variable normalization
        amb_vars = rule.get("ambiguous_variables", [])
        if not isinstance(amb_vars, list): amb_vars = []
        if isinstance(amb_vars, list):
            for var in amb_vars:
                if not isinstance(var, dict): continue
                st = clean_literal(var.get("semantic_type", ""))
                if "距" in st or "空间" in st or "间隙" in st or "distance" in st.lower() or "spatial" in st.lower(): 
                    var["semantic_type"] = "距离维度"
                elif "速" in st or "加" in st or "减" in st or "慢" in st or "快" in st or "speed" in st.lower(): 
                    var["semantic_type"] = "速度/加速度维度"
                elif "时" in st or "秒" in st or "time" in st.lower(): 
                    var["semantic_type"] = "时间维度"
                elif "行为" in st or "动作" in st or "程度" in st or "妨碍" in st or "小心" in st or "观察" in st or "behavior" in st.lower(): 
                    var["semantic_type"] = "行为与动作程度"
                else: 
                    var["semantic_type"] = "其他/未明确"
        rule["ambiguous_variables"] = amb_vars

        # 4. Priority owner normalization
        po = clean_literal(rule.get("priority_owner", "")).lower()
        if "ego" in po: 
            rule["priority_owner"] = "ego_优先(自车)"
        elif "other" in po: 
            rule["priority_owner"] = "other_优先(他车/弱势群体)"
        elif "shared" in po: 
            rule["priority_owner"] = "Shared"
        else: 
            rule["priority_owner"] = "unclear_法规未明确"

        # 5. Rule category normalization
        rc = clean_literal(rule.get("rule_category", ""))
        if "强制" in rc or "mandatory" in rc.lower():
            rule["rule_category"] = "强制性规范(Mandatory)"
        else:
            rule["rule_category"] = "倡导性建议(Advisory)"

        # 6. Action type normalization
        action = rule.get("rule_action", {})
        if not isinstance(action, dict): action = {}
        at = clean_literal(action.get("action_type", ""))
        if "停车" in at or "stop" in at.lower():
            action["action_type"] = "停车(Stop)"
        elif "让行" in at or "yield" in at.lower():
            action["action_type"] = "让行(Yield)"
        elif "减速" in at or "decelerate" in at.lower():
            action["action_type"] = "减速(Decelerate)"
        elif "保持车距" in at or "distance" in at.lower():
            action["action_type"] = "保持车距(Maintain_Distance)"
        elif "不得妨碍" in at or "impede" in at.lower():
            action["action_type"] = "不得妨碍(Do_Not_Impede)"
        elif "观察" in at or "注意" in at or "caution" in at.lower() or "observe" in at.lower():
            action["action_type"] = "注意观察(Observe/Caution)"
        elif "通行" in at or "proceed" in at.lower():
            action["action_type"] = "通行(Proceed)"
        else:
            action["action_type"] = "其他(Other)"
        rule["rule_action"] = action

        # 7. Control type normalization
        topo = rule.get("structured_topology", {})
        if not isinstance(topo, dict): topo = {}
        ct = clean_literal(topo.get("control_type", ""))
        if "无限制" in ct or "unrestricted" in ct.lower():
            topo["control_type"] = "无限制"
        elif "无信控" in ct or "unsignalized" in ct.lower():
            topo["control_type"] = "无信控"
        elif "信号灯控制" in ct or "signalized" in ct.lower():
            topo["control_type"] = "信号灯控制"
        elif "信号灯故障" in ct or "failure" in ct.lower():
            topo["control_type"] = "信号灯故障"
        elif "交通标志" in ct or "yield/stop" in ct.lower() or "标志" in ct:
            topo["control_type"] = "交通标志控制"
        elif "交警" in ct or "police" in ct.lower():
            topo["control_type"] = "交警指挥"
        elif "标线" in ct:
            topo["control_type"] = "标线控制"
        elif "环岛" in ct or "roundabout" in ct.lower():
            topo["control_type"] = "环岛"
        elif "铁道" in ct or "railway" in ct.lower() or "train" in ct.lower():
            topo["control_type"] = "铁道路口"
        else:
            topo["control_type"] = "其他/未明确"

        # 8. Ego entity normalization
        ego_e = clean_literal(topo.get("ego_entity", "")).replace("_", "")
        if "小型" in ego_e or "轿车" in ego_e or "suv" in ego_e.lower() or "乘用" in ego_e:
            topo["ego_entity"] = "小型汽车(含轿车/SUV)"
        elif "大型" in ego_e or "货车" in ego_e or "客车" in ego_e or "公交" in ego_e or "heavy" in ego_e.lower() or "truck" in ego_e.lower():
            topo["ego_entity"] = "大型车辆(货车/客车/公交)"
        elif "摩托" in ego_e or "motorcycle" in ego_e.lower():
            topo["ego_entity"] = "摩托车"
        elif "非机动" in ego_e or "自行车" in ego_e or "电动车" in ego_e or "bicycle" in ego_e.lower():
            topo["ego_entity"] = "非机动车(自行车/电动车)"
        elif "行人" in ego_e or "pedestrian" in ego_e.lower():
            topo["ego_entity"] = "行人"
        elif "特殊" in ego_e or "警车" in ego_e or "救护" in ego_e or "emergency" in ego_e.lower():
            topo["ego_entity"] = "特殊车辆(警车/救护车)"
        elif "通用" in ego_e or "未区分" in ego_e or "机动车" in ego_e or ego_e == "":
            topo["ego_entity"] = "通用机动车(未区分车型)"
        else:
            topo["ego_entity"] = "其他/未明确"

        rule["structured_topology"] = topo

    return raw_obj

def _strip_markdown_json_fence(s: str) -> str:
    """Remove ``` / ```json markdown fence while retaining interior text."""
    s = s.strip()
    if not s.startswith("```"):
        return s
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    if s.rstrip().endswith("```"):
        s = s.rstrip()[:-3].rstrip()
    return s


def _extract_balanced_json_object(s: str) -> Optional[str]:
    """Extract first bracket-balanced JSON object/array substring from text."""
    obj_pos = s.find("{")
    arr_pos = s.find("[")

    candidates = []
    if obj_pos >= 0:
        candidates.append((obj_pos, "{", "}"))
    if arr_pos >= 0:
        candidates.append((arr_pos, "[", "]"))
    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])

    for start, opener, closer in candidates:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]
    return None


def _parse_json_from_llm(raw: Optional[str]) -> Tuple[Any, str]:
    """Parse JSON returned from LLM supporting markdown fences and surrounding commentary."""
    if raw is None or not str(raw).strip():
        raise ValueError("Model returned empty content, unable to parse JSON")

    text = str(raw).strip()
    candidates: List[str] = []
    for c in (
        text,
        _strip_markdown_json_fence(text),
        _extract_balanced_json_object(text) or "",
        _strip_markdown_json_fence(_extract_balanced_json_object(text) or ""),
    ):
        if c and c not in candidates:
            candidates.append(c)

    last_err: Optional[Exception] = None
    for cand in candidates:
        try:
            return json.loads(cand), cand
        except json.JSONDecodeError as e:
            last_err = e
            continue

    preview = text[:1200] + ("…" if len(text) > 1200 else "")
    raise ValueError(f"JSON parsing failed ({last_err}). Raw output preview:\n{preview}") from last_err


def _is_intersection_rule(rule: StructuredSemanticRule) -> bool:
    if rule.macro_scenario in {"左转", "右转", "直行交叉", "交叉口通用"}:
        return True
    text = f"{rule.original_text} {rule.translated_text_zh} {rule.trigger_condition.spatial_context}".lower()
    multilingual_keywords = ("交叉口", "路口", "十字", "丁字", "左转", "右转", "直行", "让行", "优先通行", "会车", 
                             "intersection", "junction", "crossroad", "yield", "right-of-way", "kreuzung")
    return any(k in text for k in multilingual_keywords)

# ==========================================
# Core LLM Compilation (Pass2 Chunk Compilation)
# ==========================================
def pass2_compile_semantic_rules(
    chunk_content,
    chunk_metadata,
    chunk_idx: int,
    file_ego_hint: str = "通用",
) -> Tuple[RuleCompilationResult, Dict[str, Any], str]:
    metadata_str = json.dumps(chunk_metadata, ensure_ascii=False)
    schema_str = json.dumps(RuleCompilationResult.model_json_schema(), ensure_ascii=False)

    system_prompt = f"""
    You are an expert in autonomous driving traffic rule semantic modeling.
    【Current Chunk Metadata】: {metadata_str}
    【File-level Ego Vehicle Hint】: Target vehicle type for this file is "{file_ego_hint}".

    【Compilation Rules】:
    1. ✂️ [Atomic Splitting & VRU Decoupling]: Multi-object rules must be split. Note: Pedestrians/non-motorized vehicles (VRU) cannot be used as independent macro scenarios; they must be attached as interactive entities (other_entity) under specific ego vehicle maneuvers (e.g., left turn, right turn, straight).
    2. 🔄 [Turn Generalization Expansion]: When traffic rules use the general term "turning", it must be forcefully copied and instantiated separately into two independent micro topologies: "Left Turn" and "Right Turn".
    3. 🗂️ [Macro Scenario Closure & Global Generalization]: macro_scenario must be strictly chosen from these 6 categories: ["left-turn", "right-turn", "straight-through", "car-following", "lane-change/merge", "stop/start"].
       - ⭐ Critical Operation: If the extracted rule belongs to a global clause (e.g., "ensure safe passage", "drive safely"), copy and map it into all applicable macro scenario categories.
    4. ⚖️ [4-State Right-of-Way Allocation]: Right-of-way priority allocation is adversarial. Strictly choose from ["ego (自车优先)", "other (他车/弱势群体优先)", "shared (共享路权)", "unclear (未明确/死锁)"].
    5. 🚥 [Implicit Control & High-Risk Recovery]: Identify control types (control_type) accurately:
       - Signs (Stop/Yield) -> '交通标志控制'
       - Roundabout -> '环岛'
       - Railway crossing -> '铁道路口'
       - Traffic officer -> '交警指挥'
       - Signal failure -> '信号灯故障'
       - Unrestricted -> '无限制'
       - Unsignalized intersection -> '无信控'
    6. 🔍 [Fuzzy Variables Mining]: semantic_type must be strictly chosen from ["距离维度", "速度/加速度维度", "时间维度", "行为与动作程度", "安全与风险维度", "其他/未明确"].

    Output a valid JSON result adhering strictly to this JSON Schema:
    {schema_str}
    """

    def _call_pass2(msgs: List[Dict[str, str]]):
        return client.chat.completions.create(
            model=MODEL_NAME,
            messages=msgs,
            response_format={"type": "json_object"},
            temperature=0.0,
        )

    messages = [
        {"role": "system", "content": system_prompt + "\n\nOutput only one JSON object without markdown code blocks or explanatory text."},
        {"role": "user", "content": f"Please output valid json only.\n\n{chunk_content}"},
    ]
    response = _call_pass2(messages)
    if not response.choices:
        raise ValueError(f"Pass2 chunk {chunk_idx}: API returned empty choices")
    content = response.choices[0].message.content

    try:
        raw_obj, json_str = _parse_json_from_llm(content)
    except ValueError as first_err:
        print(f"      [Retry] Chunk {chunk_idx} initial JSON parse failed: {first_err}")
        repair_messages = messages + [
            {"role": "assistant", "content": content or ""},
            {
                "role": "user",
                "content": "The previous response was not valid JSON. Output ONLY a valid JSON object with top-level key 'semantic_rules'.",
            },
        ]
        response = _call_pass2(repair_messages)
        if not response.choices:
            raise ValueError(f"Pass2 chunk {chunk_idx}: API returned empty choices after retry") from first_err
        content2 = response.choices[0].message.content
        try:
            raw_obj, json_str = _parse_json_from_llm(content2)
        except ValueError as second_err:
            raise ValueError(f"Pass2 chunk {chunk_idx}: JSON parsing failed after retry") from second_err

    normalized_obj = _normalize_rule_value(raw_obj)
    if "semantic_rules" not in normalized_obj:
        normalized_obj["semantic_rules"] = []
    result = RuleCompilationResult.model_validate(normalized_obj)
    debug_packet = {
        "chunk_index": chunk_idx,
        "chunk_metadata": chunk_metadata,
        "chunk_content": chunk_content,
        "system_prompt": system_prompt,
        "raw_model_output": raw_obj,
        "normalized_output": normalized_obj,
    }
    return result, debug_packet, json_str


# ==========================================
# Hybrid Chunking Engine
# ==========================================
def enhanced_hybrid_chunking(md_text: str):
    if MarkdownHeaderTextSplitter is not None:
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2")])
        ast_docs = markdown_splitter.split_text(md_text)
        recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
        
        final_chunks = []
        for doc in ast_docs:
            sub_docs = recursive_splitter.split_text(doc.page_content)
            for i, sub in enumerate(sub_docs):
                meta = doc.metadata.copy() if doc.metadata else {}
                match = re.search(r"(Article \d+|Section \d+|§ \d+|第[一二三四五六七八九十百千]+条)", sub, re.IGNORECASE)
                if match: meta["pseudo_header"] = match.group(0)
                final_chunks.append({"content": sub, "metadata": meta})
        return final_chunks
    else:
        sections = re.split(r'\n(?=#+ )', md_text)
        final_chunks = []
        for sec in sections:
            if not sec.strip(): continue
            chunks = [sec[i:i+1200] for i in range(0, len(sec), 900)]
            for chunk in chunks:
                meta = {}
                match = re.search(r"(Article \d+|Section \d+|§ \d+|第[一二三四五六七八九十百千]+条)", chunk, re.IGNORECASE)
                if match: meta["pseudo_header"] = match.group(0)
                final_chunks.append({"content": chunk, "metadata": meta})
        return final_chunks

def _sanitize_for_excel(text: str) -> str:
    """Remove control characters and LaTeX residues invalid for Excel worksheets."""
    if not isinstance(text, str):
        return text
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    text = re.sub(r'\$[^$]*\$', '', text)
    text = re.sub(r'\\text\{[^}]*\}', '', text)
    text = re.sub(r'\\[A-Za-z]+', '', text)
    if len(text) > 32000:
        text = text[:32000] + '...(Truncated)'
    return text.strip()


# ==========================================
# Table Export Utilities
# ==========================================
def export_to_excel_and_csv(rules: List[StructuredSemanticRule], base_filename: str):
    rows = []
    for r in rules:
        amb_strs = [f"{a.name} [{a.semantic_type}]" for a in r.ambiguous_variables]
        amb_joined = " \n ".join(amb_strs)
        
        source_clause = r.debug_chunk_metadata.get("pseudo_header", r.source_path) if r.debug_chunk_metadata else r.source_path

        row = {
            "规则唯一ID": r.rule_id,
            "法规来源(条文)": _sanitize_for_excel(source_clause),
            "宏观场景": r.macro_scenario,
            "自车类型(拓扑)": r.structured_topology.ego_entity,
            "自车动作(拓扑)": _sanitize_for_excel(r.structured_topology.ego_maneuver),
            "他车动作(拓扑)": _sanitize_for_excel(r.structured_topology.other_maneuver),
            "信控类型": r.structured_topology.control_type,
            "空间上下文(条件)": _sanitize_for_excel(r.trigger_condition.spatial_context),
            "交互对象(条件)": r.trigger_condition.interactive_entity,
            "法定约束动作": r.rule_action.action_type,
            "路权归属": r.priority_owner,
            "提取的模糊参数": _sanitize_for_excel(amb_joined),
            "规范类别": r.rule_category,
            "受全局特例豁免": "是" if r.is_overridden_by_global else "否",
            "需人工核查(ego类型)": "是" if r.ego_entity_needs_review else "否",
            "原始法规文本": _sanitize_for_excel(r.original_text),
            "中文翻译": _sanitize_for_excel(r.translated_text_zh)
        }
        rows.append(row)

    columns = [
        "规则唯一ID", "法规来源(条文)", "宏观场景", "自车类型(拓扑)", "自车动作(拓扑)", "他车动作(拓扑)", "信控类型",
        "空间上下文(条件)", "交互对象(条件)", "法定约束动作", "路权归属", "提取的模糊参数",
        "规范类别", "受全局特例豁免", "需人工核查(ego类型)", "原始法规文本", "中文翻译",
    ]
    if not rows:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(rows)
        df = df.reindex(columns=columns)
    df.to_excel(f"{base_filename}.xlsx", index=False)
    df.to_csv(f"{base_filename}.csv", index=False, encoding="utf-8-sig")
    print(f"\n📊 Report successfully exported: {base_filename}.xlsx / .csv")

# ==========================================
# Main Processing Pipeline
# ==========================================
def process(file_path: str, *, debug_root: Optional[str] = None) -> List[StructuredSemanticRule]:
    debug_root = debug_root or DEFAULT_DEBUG_ROOT
    stem = _safe_output_stem(file_path)
    per_file_debug = os.path.join(debug_root, stem)
    os.makedirs(per_file_debug, exist_ok=True)

    print(f"\n[{'='*40}]\n Compiling regulation file: {os.path.basename(file_path)}\n[{'='*40}]")
    try:
        with open(file_path, "r", encoding="utf-8") as f: md = f.read()
    except UnicodeDecodeError:
        print(f"   [Encoding] UTF-8 decoding failed, trying GBK...")
        with open(file_path, "r", encoding="gbk", errors="replace") as f: md = f.read()

    # Infer file-level ego vehicle hint from filename
    fname = os.path.basename(file_path)
    file_ego_hint = "通用"
    if "大型车" in fname or "货车" in fname or "heavy" in fname.lower() or "truck" in fname.lower():
        file_ego_hint = "大型车辆(货车/客车/公交)"
    elif "小型车" in fname or "乘用" in fname or "passenger" in fname.lower():
        file_ego_hint = "小型汽车(含轿车/SUV)"
    elif "摩托车" in fname or "motorcycle" in fname.lower():
        file_ego_hint = "摩托车"
    elif "非机动" in fname or "自行车" in fname or "bicycle" in fname.lower():
        file_ego_hint = "非机动车(自行车/电动车)"

    if file_ego_hint == "通用":
        head = md[:500].lower()
        if "大型车" in head or "货车" in head or "重型" in head or "truck" in head or "heavy vehicle" in head:
            file_ego_hint = "大型车辆(货车/客车/公交)"
        elif "小型车" in head or "乘用车" in head or "轿车" in head or "小型汽车" in head:
            file_ego_hint = "小型汽车(含轿车/SUV)"
        elif "摩托车" in head or "motorcycle" in head:
            file_ego_hint = "摩托车"
        elif "非机动车" in head or "自行车" in head or "bicycle" in head:
            file_ego_hint = "非机动车(自行车/电动车)"
    print(f"   -> File-level ego vehicle hint: {file_ego_hint}")

    chunks = enhanced_hybrid_chunking(md)
    print(f"   -> Hybrid chunking complete: {len(chunks)} chunks total. Debug dir: {per_file_debug}")

    trace_file = os.path.join(per_file_debug, "compile_trace.jsonl")
    rule_file = os.path.join(per_file_debug, "compiled_rules_debug.jsonl")

    all_rules = []
    seen_hashes = set()
    processed_chunks = set()

    # 1. Read history trace, resume checkpoint
    if os.path.exists(trace_file):
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    trace_data = json.loads(line)
                    if trace_data.get("status") in ["ok", "filtered_out_non_intersection"]:
                        processed_chunks.add(trace_data.get("chunk_index"))
                except json.JSONDecodeError:
                    continue

    # 2. Read saved rules, restore hash deduplication pool
    if os.path.exists(rule_file):
        with open(rule_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip(): continue
                try:
                    r_dict = json.loads(line)
                    _hist_wrapped = _normalize_rule_value({"semantic_rules": [r_dict]})
                    if _hist_wrapped.get("semantic_rules"):
                        r_dict = _hist_wrapped["semantic_rules"][0]
                    r_obj = StructuredSemanticRule.model_validate(r_dict)
                    all_rules.append(r_obj)
                    
                    topo_str = f"{r_obj.structured_topology.ego_maneuver}-{r_obj.structured_topology.other_maneuver}"
                    path_str = str(r_obj.debug_chunk_metadata.get("pseudo_header", "")) if r_obj.debug_chunk_metadata else ""
                    rule_hash = hashlib.md5((r_obj.original_text + r_obj.macro_scenario + topo_str + path_str).encode('utf-8')).hexdigest()
                    seen_hashes.add(rule_hash)
                except Exception as e:
                    print(f"   [Warning] Historical rule restore failed at line {line_idx+1}, skipped: {e}")
                    continue

    if processed_chunks:
        print(f"   -> 🔄 Historical checkpoint detected! Skipped {len(processed_chunks)} completed chunks.")
        print(f"   -> 📦 Successfully restored {len(all_rules)} atomic rules from local cache.")

    for i, c in enumerate(chunks):
        chunk_idx = i + 1
        if len(c["content"]) < 50: continue
        
        if chunk_idx in processed_chunks:
            continue

        print(f"   -> Compiling chunk {chunk_idx}/{len(chunks)}...")
        try:
            result, debug_packet, _ = pass2_compile_semantic_rules(
                c["content"], c["metadata"], chunk_idx, file_ego_hint=file_ego_hint
            )
            
            _append_jsonl(trace_file, {"status": "ok", **debug_packet})
            
            for idx, r in enumerate(result.semantic_rules):
                if FOCUS_INTERSECTION_ONLY and not _is_intersection_rule(r):
                    _append_jsonl(trace_file, {
                        "status": "filtered_out_non_intersection",
                        "chunk_index": chunk_idx,
                        "rule_preview": {"macro_scenario": r.macro_scenario, "translated": r.translated_text_zh}
                    })
                    continue
                
                topo_str = f"{r.structured_topology.ego_maneuver}-{r.structured_topology.other_maneuver}"
                path_str = str(c["metadata"].get("pseudo_header", ""))
                rule_hash = hashlib.md5((r.original_text + r.macro_scenario + topo_str + path_str).encode('utf-8')).hexdigest()
                if rule_hash in seen_hashes: continue
                
                seen_hashes.add(rule_hash)
                r.rule_id = f"RULE-{rule_hash[:8].upper()}"
                r.debug_chunk_metadata = c["metadata"]
                all_rules.append(r)
                
                _append_jsonl(rule_file, r.model_dump())
                
        except Exception as e:
            print(f"      [Error] Chunk {chunk_idx} compilation failed: {e}")
            _append_jsonl(trace_file, {"status": "error", "chunk_index": chunk_idx, "error": str(e), "content": c["content"]})

    print(f"\n🎉 Yielded {len(all_rules)} unique atomic rules after deduplication.")
    return all_rules


def _write_outputs_for_file(
    file_path: str,
    rules: List[StructuredSemanticRule],
    rules_output_dir: str,
) -> None:
    os.makedirs(rules_output_dir, exist_ok=True)
    stem = _safe_output_stem(file_path)
    json_path = os.path.join(rules_output_dir, f"{stem}_compiled_semantic_rules_db.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in rules], f, ensure_ascii=False, indent=2)
    excel_base = os.path.join(rules_output_dir, f"{stem}_交通规则评价矩阵")
    export_to_excel_and_csv(rules, excel_base)
    print(f"   -> JSON Database: {json_path}")


# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic regulation Markdown compilation: single file or batch mode")
    parser.add_argument(
        "md_paths",
        nargs="*",
        help="Optional: specify one or more .md paths; if omitted, processes all .md files under --input-dir",
    )
    parser.add_argument("--input-dir", default=DEFAULT_MD_DIR, help="Markdown directory in batch mode")
    parser.add_argument("--debug-root", default=DEFAULT_DEBUG_ROOT, help="Debug root directory")
    parser.add_argument("--output-dir", default=DEFAULT_RULES_OUTPUT_DIR, help="Output directory for JSON + Excel/CSV")
    args = parser.parse_args()

    if args.md_paths:
        md_files = [os.path.abspath(p) for p in args.md_paths]
    else:
        md_files = sorted(glob.glob(os.path.join(os.path.abspath(args.input_dir), "*.md")))

    if not md_files:
        print(f"No .md files found. Please check directory: {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.debug_root, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Total {len(md_files)} files to process.\nDebug root: {args.debug_root}\nOutput dir: {args.output_dir}")

    for idx, fp in enumerate(md_files, start=1):
        print(f"\n>>> [{idx}/{len(md_files)}] {fp}")
        try:
            rules = process(fp, debug_root=args.debug_root)
            _write_outputs_for_file(fp, rules, args.output_dir)
        except Exception as e:
            import traceback
            print(f"   [Fatal Error] Skipping file: {e}")
            traceback.print_exc()
            err_dir = os.path.join(args.debug_root, _safe_output_stem(fp))
            os.makedirs(err_dir, exist_ok=True)
            _append_jsonl(
                os.path.join(err_dir, "compile_trace.jsonl"),
                {"status": "fatal_file_error", "file": fp, "error": str(e), "traceback": traceback.format_exc()},
            )