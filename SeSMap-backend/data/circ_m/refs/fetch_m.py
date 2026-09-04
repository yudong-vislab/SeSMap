#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取 M 语料摘要（环形基因组可视化工具），严格排除专家研究用的 12 篇。"""
import urllib.request, urllib.parse, json, re, time, difflib
import xml.etree.ElementTree as ET
from pathlib import Path
HERE=Path(__file__).resolve().parent
E="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
UA={'User-Agent':'SeSMap-academic-research/1.0 (mailto:yudong5018@gmail.com)'}
def get(u,t=4):
    last=None
    for i in range(t):
        try: return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40).read()
        except Exception as e: last=e; time.sleep(1.5*(i+1))
    raise last
def norm(s): return re.sub(r'[^a-z0-9]+',' ',(s or '').lower()).strip()
def sim(a,b): return difflib.SequenceMatcher(None,norm(a),norm(b)).ratio()

# --- 专家研究的 12 篇：一律排除 ---
STUDY=[
 "AuraGenome An LLM-Powered Framework for On-the-Fly Reusable and Scalable Circular Genome Visualizations",
 "CGView.js Circular Genome Visualization in the Web Browser",
 "Circos an information aesthetic for comparative genomics",
 "Should I make it round Suitability of circular and linear layouts",
 "GenoREC A Recommendation System for Interactive Genomics Data Visualization",
 "GlyphCreator Towards Example-based Automatic Generation of Circular Glyphs",
 "Gosling A Grammar-based Toolkit for Scalable and Interactive Genomics Data Visualization",
 "HERA Interactive Circular Comparison of Plasmids and Other Genomic Sequences",
 "IntelliCircos",
 "NG-Circos Next-generation Circos for Data Visualization and Interpretation",
 "BioCircos.js an interactive Circos JavaScript library for biological data visualization on web applications",
 "interacCircos An R Package Based on JavaScript Libraries for the Generation of Interactive Circos Plots",
]
STUDY_TOK=[{"auragenome"},{"cgview","js"},{"circos","aesthetic"},{"round","suitability"},
           {"genorec"},{"glyphcreator"},{"gosling"},{"hera","plasmids"},{"intellicircos"},
           {"ng","circos"},{"biocircos"},{"interaccircos"}]
def is_study(title):
    t=set(norm(title).split())
    for full,tok in zip(STUDY,STUDY_TOK):
        if sim(title,full)>=0.80: return True
        if tok and tok <= t: return True
    return False

ids=json.load(open(HERE/'pmids.json'))
out=[]; skip_study=[]; skip_short=0
for i in range(0,len(ids),40):
    chunk=ids[i:i+40]
    x=ET.fromstring(get(f"{E}efetch.fcgi?db=pubmed&retmode=xml&id={','.join(chunk)}"))
    for art in x.findall('.//PubmedArticle'):
        a=art.find('.//Article'); ti=a.findtext('ArticleTitle') or ''
        segs=[(e.get('Label'),(e.text or '')) for e in a.findall('.//Abstract/AbstractText')]
        ab=" ".join((f"{l}: {t}" if l else t) for l,t in segs).strip()
        yr=art.findtext('.//PubDate/Year') or ''
        pmid=art.findtext('.//PMID')
        if is_study(ti): skip_study.append(ti[:70]); continue
        if len(ab)<250: skip_short+=1; continue
        out.append(dict(pmid=pmid,title=ti.rstrip('.'),year=yr,abstract=ab))
    time.sleep(0.4)
json.dump(out,open(HERE/'abstracts.json','w'),ensure_ascii=False,indent=1)
print(f"M 语料候选: {len(out)} 篇, 共 {sum(len(o['abstract']) for o in out):,} 字符")
print(f"  排除 12 篇研究论文命中: {len(skip_study)}")
for t in skip_study: print(f"    - {t}")
print(f"  摘要过短剔除: {skip_short}")
