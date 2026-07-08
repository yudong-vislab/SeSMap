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
    response = client.chat.completions.create(
        model=model_for("summary"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    # 提取结果
    text_output = response.choices[0].message.content.strip()
    text_output = clean_json_text(text_output)
    try:
        msus = json.loads(text_output)
    except json.JSONDecodeError:
        print("⚠️ JSON 解析失败，返回原始输出：")
        print(text_output)
        return None
    return msus

# def batch_process(root_dir: str):
#     root = Path(root_dir)
#     for sub in root.iterdir():
#         if not sub.is_dir():
#             continue
#         name = sub.name
#         json_path = sub/f"{name}.json"
        
if __name__ == "__main__":
    # 示例
    # json_path = Path("/home/lxy/case_engine/largeeddy/largeeddy.json")
    # with open(json_path, "r", encoding="utf-8") as f:
    #     data = json.load(f)
    # extractmsu = []
    # for section in data:
    #     title = section.get("title", "No Title")
    #     print(f"\n=== Section: {title} ===")
    #     for para in section.get("paragraphs", []):
    #         para_type = para.get("type")
    #         origin = para.get("origin_text", "").strip()

    #         if para_type == "text":
    #             paragraph = origin
    #             result = extract_msu(paragraph)
    #             print(json.dumps(result, indent=2, ensure_ascii=False))
    #             extractmsu.append({"paragraph":paragraph,"type":"text","resultmsu":result})
    #         elif para_type == "figure":
    #             extractmsu.append({"paragraph":origin,"type":"figure"})
    # with open("/home/lxy/case_engine/largeeddy/largeeddy.json", "w", encoding="utf-8") as f:
    #     json.dump(extractmsu, f, ensure_ascii=False, indent=2)    

    para = "Fig. [7](#page-6-0) presents a comprehensive overview of two distinct combustion stabilization mechanisms in a cavity-floor direct-injection scramjet. Building upon previous studies by Yuan et al. [\\[10](#page-11-0)], it is found that during scram mode, the fuel jet splits the cavity into two distinct regions: a rich-premixed zone and a hot product zone. The resulting flame in the cavity is discontinuous and is stabilized within the shear layers of recirculation zones. The shear layer and fuel jet impede supersonic inflow, creating reflected shocks and a bow shock. On the other hand, during ram mode operation, the flame is stabilized in the jet-wake. Furthermore, the corner recirculation zone plays a vital role in maintaining continuous ignition of the fuel jet. Although there is no primary recirculation zone within the cavity, the hot products that flow downstream have positive effects on the jet-wake flame. This is because pure jet-wake stabilized combustion is not attainable under low inflow stagnation temperatures [\\[9\\]](#page-11-0). Additionally, the shear layer and shock train exhibit violent oscillations due to large-scale vortex shedding. As a result, the flow field is complex and intrinsically unstable.",
    result = extract_msu(para)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("-------------")
    para = "Scrutinizing the local flow regimes is significant for comprehending the underlying mechanisms driving mode transition. Fig. [9](#page-7-0) demonstrates the 1-D supersonic vs. subsonic and upstream vs. downstream mass flow ratio (*q*˙*m,local/q*˙*m,tot*) along the engine length. Notably, since the flow in recirculation zones is primarily subsonic, the supersonic upstream regime is excluded from the analysis. Inspired by the work of Cao et al. [\\[11\\]](#page-11-0), the filter functions used to extract the local mass flow rate of mixture through a cross section are defined as",
    result = extract_msu(para)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("-------------")

