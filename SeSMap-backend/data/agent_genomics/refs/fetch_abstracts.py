#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 PubMed 抓取引文摘要；失败回退 Crossref。带标题校验，防止抓错论文。"""
import urllib.request, urllib.parse, json, re, time, sys, difflib
import xml.etree.ElementTree as ET
from pathlib import Path
HERE=Path(__file__).resolve().parent
E="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
UA={'User-Agent':'SeSMap-academic-research/1.0 (mailto:yudong5018@gmail.com)'}

def get(url, tries=4):
    last=None
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=40).read()
        except Exception as e:
            last=e; time.sleep(1.5*(i+1))
    raise last

def norm(s): return re.sub(r'[^a-z0-9]+',' ',(s or '').lower()).strip()
def sim(a,b): return difflib.SequenceMatcher(None,norm(a),norm(b)).ratio()

def pubmed(title):
    for term in [re.sub(r'[^\w\s]',' ',title)+"[Title]", title]:
        try:
            r=json.loads(get(f"{E}esearch.fcgi?db=pubmed&retmode=json&retmax=5&term={urllib.parse.quote(term)}"))
            ids=r['esearchresult']['idlist']
        except Exception: continue
        if not ids: continue
        time.sleep(0.4)
        try: x=ET.fromstring(get(f"{E}efetch.fcgi?db=pubmed&retmode=xml&id={','.join(ids)}"))
        except Exception: continue
        best=None
        for art in x.findall('.//PubmedArticle'):
            a=art.find('.//Article'); at=a.findtext('ArticleTitle') or ''
            s=sim(title,at)
            segs=[(e.get('Label'),(e.text or '')) for e in a.findall('.//Abstract/AbstractText')]
            ab=" ".join((f"{l}: {t}" if l else t) for l,t in segs).strip()
            if ab and (best is None or s>best[0]): best=(s,at,ab,art.findtext('.//PMID'))
        if best and best[0]>=0.72: return dict(src='pubmed',score=round(best[0],3),
                                               matched_title=best[1],abstract=best[2],pmid=best[3])
        time.sleep(0.4)
    return None

def semanticscholar(title):
    try:
        r=json.loads(get("https://api.semanticscholar.org/graph/v1/paper/search?limit=3&fields=title,abstract&query="
                         +urllib.parse.quote(title)))
    except Exception: return None
    for it in r.get('data',[]) or []:
        ab=it.get('abstract'); t=it.get('title') or ''
        if ab and sim(title,t)>=0.72:
            return dict(src='s2',score=round(sim(title,t),3),matched_title=t,abstract=re.sub(r'\s+',' ',ab).strip())
    return None

def crossref(title):
    try:
        r=json.loads(get("https://api.crossref.org/works?rows=3&select=title,abstract,DOI&query.bibliographic="
                         +urllib.parse.quote(title)))
    except Exception: return None
    for it in r.get('message',{}).get('items',[]):
        t=(it.get('title') or [''])[0]; ab=it.get('abstract')
        if ab and sim(title,t)>=0.72:
            ab=re.sub(r'<[^>]+>',' ',ab); ab=re.sub(r'\s+',' ',ab).strip()
            ab=re.sub(r'^Abstract\s*','',ab,flags=re.I)
            return dict(src='crossref',score=round(sim(title,t),3),matched_title=t,abstract=ab,doi=it.get('DOI'))
    return None

rows=[l.rstrip('\n').split('\t') for l in open(HERE/'references.tsv') if l.strip()]
rows=[(r[0],'tool',r[1],r[2],r[3]) if len(r)==4 else r for r in rows]
out=[]; ok=fail=0
for i,(key,kind,title,venue,year) in enumerate(rows,1):
    r=crossref(title) or semanticscholar(title) or pubmed(title)
    if r and len(r['abstract'])>=180:
        out.append(dict(key=key,title=title,venue=venue,year=year,**r)); ok+=1
        print(f"[{i:>2}/{len(rows)}] OK   {r['src']:8} sim={r['score']:.2f} len={len(r['abstract']):>4}  {key} {title[:52]}",flush=True)
    else:
        why="太短" if r else "未命中"
        out.append(dict(key=key,title=title,venue=venue,year=year,src=None,abstract=None,note=why)); fail+=1
        print(f"[{i:>2}/{len(rows)}] MISS ({why})                        {key} {title[:52]}",flush=True)
    time.sleep(0.45)
json.dump(out,open(HERE/'abstracts.json','w'),ensure_ascii=False,indent=1)
print(f"\n成功 {ok}/{len(rows)} ({ok/len(rows)*100:.0f}%)   失败 {fail}")
print(f"总摘要字数 {sum(len(o['abstract']) for o in out if o.get('abstract')):,}")
