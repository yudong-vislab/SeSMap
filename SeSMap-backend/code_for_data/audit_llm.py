#!/usr/bin/env python3
"""对抽样的 MSU 与 pair 做自动化审计；两轮独立评判用于一致性统计。"""
import json, sys, re, time, random
sys.path.insert(0, '.')
from services.llm_config import model_for, get_openai_client

DB = json.load(open('data/stages/02_msu/formdatabase.json'))
S  = json.load(open('data/stages/02_msu/audit_sample.json'))
cli, MODEL = get_openai_client(), 'gpt-4o'   # 端点当前不提供 gpt-5.1-chat

def ask(prompt, retry=3):
    for i in range(retry):
        try:
            r = cli.chat.completions.create(model=MODEL, temperature=0,
                    messages=[{"role":"user","content":prompt}])
            t = re.sub(r'^```[a-zA-Z]*\n?|```$', '', r.choices[0].message.content.strip())
            return json.loads(t)
        except Exception as e:
            if i == retry-1: print('  [fail]', str(e)[:70], flush=True); return None
            time.sleep(2)

MSU_P = """You audit sentence-level units extracted from scientific papers. Judge each item on five criteria.
Be reasonable rather than strict: accept an item if it would be usable by a researcher, even if wording is imperfect.

1. fidelity      - faithful to the source paragraph, no invented content
2. selfcontained - understandable on its own, without needing most surrounding text
3. condition     - keeps method/experiment conditions that affect interpretation (true if none apply)
4. traceable     - can be located back to the source paragraph
5. role          - the discourse-role label is reasonable

Return ONLY a JSON array, one object per item, same order:
[{"id":<id>,"fidelity":true,"selfcontained":true,"condition":true,"traceable":true,"role":true}]

ITEMS:
%s"""

PAIR_P = """You audit candidate pairs of statements drawn from DIFFERENT scientific papers.
Question for each pair: do these two statements form a candidate worth inspecting together in a cross-paper comparison?
Do NOT judge whether they already form a confirmed scientific conclusion. Be reasonable: a shared phenomenon,
a related method, complementary evidence, or a shared analytical concept all qualify.

Label each pair: "accepted" (worth joint inspection), "uncertain" (unclear), or "invalid" (unrelated).

Return ONLY a JSON array, same order:
[{"id":<id>,"label":"accepted"}]

PAIRS:
%s"""

def run_msu(order, tag):
    out = {}
    for s in range(0, len(order), 10):
        ch = order[s:s+10]
        body = "\n".join(
            f'{{"id":{i},"role_label":"{DB[i].get("category")}","text":"{(DB[i].get("sentence") or "")[:400]}",'
            f'"source":"{str(DB[i].get("paragraph_info") or "")[:200]}"}}' for i in ch)
        r = ask(MSU_P % body)
        for o in (r or []):
            if isinstance(o, dict) and 'id' in o: out[int(o['id'])] = o
        if (s//10) % 10 == 0: print(f"  [{tag}] MSU {s+len(ch)}/{len(order)}", flush=True)
    return out

def run_pair(order, tag):
    out = {}
    for s in range(0, len(order), 10):
        ch = order[s:s+10]
        body = "\n".join(
            f'{{"id":{k},"A":"{(DB[a].get("sentence") or "")[:280]}","B":"{(DB[b].get("sentence") or "")[:280]}"}}'
            for k,(a,b,c,h) in ch)
        r = ask(PAIR_P % body)
        for o in (r or []):
            if isinstance(o, dict) and 'id' in o: out[int(o['id'])] = o
        if (s//10) % 10 == 0: print(f"  [{tag}] pair {s+len(ch)}/{len(order)}", flush=True)
    return out

msu = S['msu_idx']
pairs = list(enumerate(S['pairs']))
res = {}
print(f"model={MODEL}  MSU={len(msu)}  pairs={len(pairs)}\n", flush=True)
for tag in ['r1','r2']:
    o1 = msu[:] ; o2 = pairs[:]
    if tag == 'r2':                       # 第二轮打乱呈现顺序，构成独立评判
        random.seed(7); random.shuffle(o1); random.shuffle(o2)
    res[f'msu_{tag}']  = run_msu(o1, tag)
    res[f'pair_{tag}'] = run_pair(o2, tag)
json.dump({k:{str(i):v for i,v in d.items()} for k,d in res.items()},
          open('results/audit_raw.json','w'))
print("\n[saved] results/audit_raw.json")
