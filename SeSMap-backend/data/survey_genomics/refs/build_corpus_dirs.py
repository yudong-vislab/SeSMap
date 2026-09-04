#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把综述正文 + 81 篇引文摘要落成 01_corpus/<name>/<name>.md，供 build_corpus.py 消费。
结构化摘要按 LABEL 拆成独立段落，便于 MSU 抽取与 discourse-role 归类。"""
import json, re, sys, shutil
from pathlib import Path
BK=Path('/Users/yudong/Desktop/SeSMap/SeSMap-backend')
SG=BK/'data/survey_genomics'
CORP=SG/'stages/01_corpus'

# ---- 泄漏控制：评测集中的论文，其相关内容一律不进训练语料 ----
LEAK_TERMS=[r'\bcircos\b', r'\bbiocircos\b', r'krzywinski']

def slug(s):
    s=re.sub(r'[^\w\s-]','',s); s=re.sub(r'\s+','_',s.strip())
    return s[:80]

def write_md(name, title, paragraphs):
    d=CORP/name; d.mkdir(parents=True, exist_ok=True)
    body=[f"# {title}",""]
    for head,txt in paragraphs:
        if head: body += [f"## {head}",""]
        body += [txt,""]
    (d/f"{name}.md").write_text("\n".join(body), encoding='utf-8')
    return sum(len(t) for _,t in paragraphs)

def split_abstract(ab):
    """结构化摘要 'MOTIVATION: ... RESULTS: ...' -> [(label, text), ...]"""
    parts=re.split(r'(?:^|\s)([A-Z][A-Z /&-]{3,30}):\s', ' '+ab)
    if len(parts)>=3:
        out=[]
        for i in range(1,len(parts)-1,2):
            lbl=parts[i].strip().title(); txt=parts[i+1].strip()
            if len(txt)>=40: out.append((lbl,txt))
        if out: return out
    return [("Abstract", ab.strip())]

def main():
    abs_json=json.load(open(SG/'refs/abstracts.json'))
    n_ok=n_skip=0; chars=0
    for r in abs_json:
        ab=r.get('abstract')
        if not ab: n_skip+=1; continue
        blob=(r['title']+' '+ab).lower()
        if any(re.search(p,blob) for p in LEAK_TERMS):
            print(f"  [泄漏剔除] {r['key']}  {r['title'][:60]}"); n_skip+=1; continue
        name=f"{r['key']}_{slug(r['title'].split(':')[0])}"
        chars+=write_md(name, r['title'], split_abstract(ab)); n_ok+=1
    print(f"\n摘要语料: 写入 {n_ok} 篇, 跳过 {n_skip} 篇, 共 {chars:,} 字符")
    return n_ok

if __name__=='__main__':
    CORP.mkdir(parents=True, exist_ok=True)
    main()
