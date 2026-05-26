import json
import os
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from services.llm_config import LLM_CONFIG, model_for

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# 设置OpenAI API密钥
client = OpenAI(api_key=LLM_CONFIG.api_key, base_url=LLM_CONFIG.base_url)# 加载JSON文件1和文件2
def load_json_files(file1_path, file2_path):
    with open(file1_path, 'r', encoding='utf-8') as f1:
        file1_data = json.load(f1)
    with open(file2_path, 'r', encoding='utf-8') as f2:
        file2_data = json.load(f2)
    return file1_data, file2_data

# 调用大模型API进行总结
def summarize_with_llm(text):
    try:
        response = client.chat.completions.create(
            model=model_for("summary"),
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You summarize clustered MSU sentences for a semantic-map HSU. "
                        "Use only the provided sentences, preserve technical terms, and write in English. "
                        "Identify the shared theme, key evidence, and any obvious limitation in one concise paragraph. "
                        "Do not add outside knowledge or generic filler."
                    )
                },
                {
                    "role": "user", 
                    "content": (
                        "The following MSU sentences are close in the semantic layout and belong to one HSU cluster. "
                        "Write a 45-80 word evidence-grounded English summary of their shared meaning.\n\n"
                        f"MSU sentences:\n{text}"
                    )}
                ],
        )
        return  response.choices[0].message.content.strip()
    except Exception as e:
        print(f"调用API时出错: {e}")
        return None

# 主处理函数
def process_and_summarize(file1_data, file2_data):
    # 为file2_data创建映射：MSU_id -> sentence
    msu_to_sentence = {}
    for item in file2_data:
        msu_id = item["MSU_id"]
        sentence = item["sentence"]
        msu_to_sentence[msu_id] = sentence

    results = []  # 存储每个元素的结果
    for i, element in enumerate(file1_data):
        msu_ids = element["MSU_ids"]
        sentences = []
        for msu_id in msu_ids:
            if msu_id in msu_to_sentence:
                sentences.append(msu_to_sentence[msu_id])
            else:
                print(f"Warning: MSU_id {msu_id} not found in file2_data")
        
        # 组合所有sentence为一个文本
        text_to_summarize = "\n\n".join(sentences)
        
        # 调用大模型进行总结
        print(f"正在处理元素 {i+1}/{len(file1_data)}...")
        print(text_to_summarize)
        summary = summarize_with_llm(text_to_summarize)
        print(f"总结完成: {summary}\n")
        
        # 存储结果
        result = {
            "hex_coord": element.get("hex_coord", []),
            "country": element.get("country", 0),
            "MSU_ids": msu_ids,
            "summary": summary
        }
        results.append(result)
    
    return results

# 示例用法
if __name__ == "__main__":
    # 假设文件路径
    file1_path = "/home/lxy/case_pollution/hexagon_info_0.15.json"
    file2_path = "/home/lxy/case_pollution/formdatabase_v2.0.json"
    
    # 加载数据
    file1_data, file2_data = load_json_files(file1_path, file2_path)
    
    # 处理并获取总结
    results = process_and_summarize(file1_data, file2_data)
    
    # 输出结果
    for i, result in enumerate(results):
        print(f"结果 {i+1}:")
        print(f"坐标: {result['hex_coord']}")
        print(f"国家: {result['country']}")
        print(f"总结: {result['summary']}")
        print("\n" + "="*50 + "\n")
    
    # 可选：将结果保存到文件
    with open("/home/lxy/case_pollution/pollution_summaries.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
