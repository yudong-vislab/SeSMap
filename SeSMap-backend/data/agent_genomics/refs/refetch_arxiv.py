#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 Crossref/S2/PubMed 未命中的条目补抓 arXiv。"""
import urllib.request, urllib.parse, json, re, time, difflib
import xml.etree.ElementTree as ET
from pathlib import Path
HERE=Path(__file__).resolve().parent
UA={'User-Agent':'SeSMap-academic-research/1.0 (mailto:yudong5018@gmail.com)'}
def get(u,tries=3):
    last=None
    for i in range(tries):
        try: return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40).read()
        except Exception as e: last=e; time.sleep(1.5*(i+1))
    raise last
def norm(s): return re.sub(r'[^a-z0-9]+',' ',(s or '').lower()).strip()
def sim(a,b): return difflib.SequenceMatcher(None,norm(a),norm(b)).ratio()
NS={'a':'http://www.w3.org/2005/Atom'}
def arxiv(title):
    q=urllib.parse.quote(f'all:"{re.sub(chr(34),"",title)}"')
    try: x=ET.fromstring(get(f"http://export.arxiv.org/api/query?search_query={q}&max_results=5"))
    except Exception: return None
    best=None
    for e in x.findall('a:entry',NS):
        t=(e.findtext('a:title',default='',namespaces=NS) or '').strip()
        ab=re.sub(r'\s+',' ',(e.findtext('a:summary',default='',namespaces=NS) or '')).strip()
        s=sim(title,t)
        if ab and (best is None or s>best[0]): best=(s,t,ab)
    if best and best[0]>=0.70:
        return dict(src='arxiv',score=round(best[0],3),matched_title=best[1],abstract=best[2])
    return None
data=json.load(open(HERE/'abstracts.json'))
n=0
for r in data:
    if r.get('abstract'): continue
    got=arxiv(r['title'])
    if got and len(got['abstract'])>=180:
        r.update(got); r.pop('note',None); n+=1
        print(f"  补回 {r['key']:5} sim={got['score']:.2f} len={len(got['abstract']):>4}  {r['title'][:50]}",flush=True)
    time.sleep(1.2)
json.dump(data,open(HERE/'abstracts.json','w'),ensure_ascii=False,indent=1)
ok=sum(1 for r in data if r.get('abstract'))
print(f"\narXiv 补回 {n} 条 -> 总计 {ok}/{len(data)} ({ok/len(data)*100:.0f}%)")
print(f"总摘要字数 {sum(len(r['abstract']) for r in data if r.get('abstract')):,}")
