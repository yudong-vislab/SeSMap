import json
import os
import re
from pathlib import Path
from openai import OpenAI  
import base64 
from dotenv import load_dotenv
from services.llm_config import LLM_CONFIG, model_for
# 1. 读取 formdatabase.json

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

client = OpenAI(
    api_key=LLM_CONFIG.api_key,
    base_url=LLM_CONFIG.base_url
)

def ask_llm_with_image_and_text(image_path, text):
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')
    response = client.chat.completions.create(
        model=model_for("summary"),
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64_image}}
            ]}
        ],
        temperature=0
    )
    answer = response.choices[0].message.content.strip()
    return answer
def clean_json_text(text: str) -> str:
    # 去掉 markdown 代码块 ```json 或 ```
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return text.strip()

with open('case_engine/formdatabase.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 按次序编号，添加 MSU_id 属性
for idx, item in enumerate(data):
    item['MSU_id'] = idx

# 3. 处理 figure 的 sentence 属性，并调用大模型
for idx, item in enumerate(data):
    if item.get('type') == 'figure' and idx + 1 < len(data):
        item['sentence'] = data[idx + 1].get('para_info', {})
        item['2d_coord'] = data[idx + 1].get('2d_coord', [0, 0])
        item['para_id'] = data[idx + 1].get('para_id', -1)
        # 调用大模型，假设图片路径在item['image_path']，文本在item['sentence'].get('sentence', '')
        image_path = f"case_engine/images/{item.get('paper_info', '')}/{item.get('para_info', '')}"        
        print(f"Processing figure MSU_id {item['MSU_id']} with image {image_path}")
        text = item['sentence']
        prompt = f"""
You extract one figure-level MSU from a scientific paper image and its nearby text.

Use the image and the provided text as evidence. If the text is not a caption, rely on the visible figure content and say only what is supported.

Output requirements:
- Produce exactly one self-contained sentence describing the figure's scientific content.
- Classify it as one of: Background, Method, Experiment, Result, Conclusion, Other.
- Rank importance from 1 to 5; use higher rank for paper-specific methods, experiments, or findings.
- Output strict JSON only, no markdown:
[
  {{"sentence":"...","category":"Background|Method|Experiment|Result|Conclusion|Other","rank":1}}
]

Nearby text:
{text}
        """
        text_output = ask_llm_with_image_and_text(image_path, prompt)
        text_output = clean_json_text(text_output)
        try:
            msus = json.loads(text_output)
            print(f"解析成功: {msus}")
            item['sentence'] = msus[0].get('sentence', '')
            item['category'] = msus[0].get('category', 'others')
            item['rank'] = msus[0].get('rank', -1)
        except json.JSONDecodeError:
            print("⚠️ JSON 解析失败，返回原始输出：")
            print(text_output)


with open('case_engine/formdatabase.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# # 4. 生成连线
# links = []
# link_idx = 0
# # 按 para_id 分组
# from collections import defaultdict
# groups = defaultdict(list)
# for item in data:
#     groups[item.get('para_id', -1)].append(item)
# # 4.1. 对每组 para_id 内部按 MSU_id 排序后逐个连线
# for para_id, items in groups.items():
#     items_sorted = sorted(items, key=lambda x: x['MSU_id'])
#     for i in range(len(items_sorted) - 1):
#         link = {
#             'type': 'road',
#             'panelIdx': link_idx,
#             'countryFrom': para_id,
#             'countryTo': para_id,
#             'path': [
#                 {'q': items_sorted[i]['MSU_id'], 'r': 0},
#                 {'q': items_sorted[i+1]['MSU_id'], 'r': 0}
#             ]
#         }
#         links.append(link)
#         link_idx += 1
# # 4.2. figure与下一个sentence连线
# for idx, item in enumerate(data):
#     if item.get('type') == 'figure' and idx + 1 < len(data):
#         link = {
#             'type': 'road',
#             'panelIdx': link_idx,
#             'countryFrom': item.get('para_id', -1),
#             'countryTo': data[idx+1].get('para_id', -1),
#             'path': [
#                 {'q': item['MSU_id'], 'r': 0},
#                 {'q': data[idx+1]['MSU_id'], 'r': 0}
#             ]
#         }
#         links.append(link)
#         link_idx += 1

# # 5. 保存所有连线到 line.json
# with open('line.json', 'w', encoding='utf-8') as f:
#     json.dump({'links': links}, f, ensure_ascii=False, indent=2)

# print('已生成 line.json，连线数量:', len(links))
