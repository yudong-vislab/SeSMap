import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from services.llm_config import LLM_CONFIG, model_for, get_openai_client

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# OpenAI 客户端改为惰性构造：无 key 时不再在 import 阶段崩溃（也便于 --no-llm）。

def clean_json_text(text: str) -> str:
    # 去掉 markdown 代码块 ```json 或 ```
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def parse_msu_json(text: str):
    """Decode an LLM JSON response, including unescaped LaTex backslashes.

    Models occasionally return valid-looking JSON whose LaTex fragments use a
    single backslash (for example ``\\in``).  That is not valid JSON and used
    to make the complete paragraph disappear from the corpus.  Preserve the
    text by escaping only backslashes that are not already valid JSON escapes.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r'\\(?!(?:["\\\\/bfnrt]|u[0-9a-fA-F]{4}))', r'\\\\', text)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            print("⚠️ JSON 解析失败，返回原始输出：")
            print(text)
            return None
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        print("⚠️ MSU 输出不是对象列表，返回原始输出：")
        print(text)
        return None
    return data

def extract_msu(paragraph: str):
    """
    输入科研论文文本，输出 Minimum Semantic Units (MSUs)
    每个 MSU = {sentence, category, rank}
    """

    prompt = f"""
You extract Minimum Semantic Units (MSUs) from scientific-paper text for SeSMap.

MSU rules:
1. Each MSU must be a single, self-contained scientific statement.
2. Split conjunctions, causal clauses, aims, methods, and results when they contain distinct facts.
3. Preserve important technical terms, variables, datasets, model names, and measurements.
4. Do not include vague author-introduction or navigation text unless it carries scientific content.
5. Classify each MSU as exactly one of: Background, Method, Experiment, Result, Conclusion, Other.
6. Rank importance from 1 to 5. Use 5 for paper-specific contributions, key methods, or central findings.

Output strict JSON only. No markdown, comments, or trailing text.
Schema:
[
  {{"sentence":"...","category":"Background|Method|Experiment|Result|Conclusion|Other","rank":1}}
]

Example input:
We conducted experiments using three datasets to validate the proposed method. The method integrates multimodal features and applies a transformer-based encoder. The results demonstrate significant improvement compared to baseline models.

Example output:
[
  {{"sentence":"Experiments were conducted using three datasets.","category":"Experiment","rank":3}},
  {{"sentence":"The experiments validate the proposed method.","category":"Experiment","rank":3}},
  {{"sentence":"The proposed method integrates multimodal features.","category":"Method","rank":4}},
  {{"sentence":"The proposed method applies a transformer-based encoder.","category":"Method","rank":5}},
  {{"sentence":"The results demonstrate significant improvement compared to baseline models.","category":"Result","rank":4}}
]

Paragraph:
{paragraph}
"""

    client = get_openai_client()
    # 瞬时网络/限流错误重试：单次抖动不应中断整批语料构建
    import time as _t
    _last = None
    for _try in range(5):
        try:
            response = client.chat.completions.create(
                model=model_for("summary"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            break
        except Exception as _e:
            _last = _e
            print(f"    [retry {_try+1}/5] {type(_e).__name__}: {str(_e)[:80]}", flush=True)
            _t.sleep(3 * (_try + 1))
    else:
        print(f"    [skip] 连续 5 次失败，跳过该段: {type(_last).__name__}", flush=True)
        return []

    # 提取结果
    text_output = response.choices[0].message.content.strip()
    text_output = clean_json_text(text_output)
    return parse_msu_json(text_output)

# def batch_process(root_dir: str):
#     root = Path(root_dir)
#     for sub in root.iterdir():
#         if not sub.is_dir():
#             continue
#         name = sub.name
#         json_path = sub/f"{name}.json"
        
if __name__ == "__main__":
    # Lightweight smoke prompt for direct script testing.
    para = "Fig. [7](#page-6-0) presents a comprehensive overview of two distinct combustion stabilization mechanisms in a cavity-floor direct-injection scramjet. Building upon previous studies by Yuan et al. [\\[10](#page-11-0)], it is found that during scram mode, the fuel jet splits the cavity into two distinct regions: a rich-premixed zone and a hot product zone. The resulting flame in the cavity is discontinuous and is stabilized within the shear layers of recirculation zones. The shear layer and fuel jet impede supersonic inflow, creating reflected shocks and a bow shock. On the other hand, during ram mode operation, the flame is stabilized in the jet-wake. Furthermore, the corner recirculation zone plays a vital role in maintaining continuous ignition of the fuel jet. Although there is no primary recirculation zone within the cavity, the hot products that flow downstream have positive effects on the jet-wake flame. This is because pure jet-wake stabilized combustion is not attainable under low inflow stagnation temperatures [\\[9\\]](#page-11-0). Additionally, the shear layer and shock train exhibit violent oscillations due to large-scale vortex shedding. As a result, the flow field is complex and intrinsically unstable.",
    result = extract_msu(para)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("-------------")
    para = "Scrutinizing the local flow regimes is significant for comprehending the underlying mechanisms driving mode transition. Fig. [9](#page-7-0) demonstrates the 1-D supersonic vs. subsonic and upstream vs. downstream mass flow ratio (*q*˙*m,local/q*˙*m,tot*) along the engine length. Notably, since the flow in recirculation zones is primarily subsonic, the supersonic upstream regime is excluded from the analysis. Inspired by the work of Cao et al. [\\[11\\]](#page-11-0), the filter functions used to extract the local mass flow rate of mixture through a cross section are defined as",
    result = extract_msu(para)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("-------------")
