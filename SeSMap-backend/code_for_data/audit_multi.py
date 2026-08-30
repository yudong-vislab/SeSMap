#!/usr/bin/env python3
"""跨语料自动化审计：25 篇合并库，各语料分层抽样 20%，两轮独立判定。"""
import json, os, sys, re, time, random, collections
import numpy as np
sys.path.insert(0, '.')
from services.llm_config import get_openai_client

MODEL = 'gpt-4o'
FRAC  = 0.20
SEED  = 20260827
CORP  = [('A', 'data/stages'), ('B', 'data/general_v11/stages'), ('C', 'data/bio_eval/stages')]
cli   = get_openai_client()

def band(c):
    return '0.55-0.65' if c < .65 else '0.65-0.75' if c < .75 else '0.75-0.85' if c < .85 else '0.85-0.95'

# ---------------- 分层抽样 ----------------
random.seed(SEED)
SAMP = {}
for tag, base in CORP:
    db = json.load(open(f'{base}/02_msu/formdatabase.json'))
    X  = np.load(f'{base}/04_embeddings/emb_corpus.npy').astype(np.float32)
    N  = X.shape[0]
    pf = next(f for f in ['llm_pairs_10k.json', 'llm_pairs.json']
              if os.path.exists(f'{base}/02_msu/{f}'))
    pj = json.load(open(f'{base}/02_msu/{pf}'))

    st = collections.defaultdict(list)
    for i, r in enumerate(db[:N]):
        if len((r.get('sentence') or '').split()) < 3: continue
        st[(r.get('paper_id'), r.get('category') or 'None')].append(i)
    midx = sorted({j for ids in st.values()
                   for j in random.sample(ids, max(1, round(len(ids)*FRAC)))})

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    role = [(r.get('category') or 'None') for r in db[:N]]
    pos = [(int(a), int(b)) for a, b, *_ in pj['train_pos'] + pj['test_pos']
           if int(a) < N and int(b) < N]
    held = {(int(a), int(b)) for a, b, *_ in pj['test_pos']}
    ps = collections.defaultdict(list)
    for a, b in pos:
        c = float(Xn[a] @ Xn[b])
        ps[(band(c), tuple(sorted((role[a], role[b]))))].append((a, b, c))
    psamp = [x for lst in ps.values() for x in random.sample(lst, max(1, round(len(lst)*FRAC)))]

    SAMP[tag] = {'base': base, 'db': db, 'msu': midx, 'pairs': psamp,
                 'held': held, 'n_msu': N, 'n_pair': len(pos)}
    print(f'[{tag}] MSU {len(midx)}/{N} ({len(midx)/N*100:.1f}%)  '
          f'pair {len(psamp)}/{len(pos)} ({len(psamp)/len(pos)*100:.1f}%)', flush=True)

# ---------------- LLM 判定 ----------------
def ask(prompt, retry=3):
    for i in range(retry):
        try:
            r = cli.chat.completions.create(model=MODEL, temperature=0,
                    messages=[{"role": "user", "content": prompt}])
            return json.loads(re.sub(r'^```[a-zA-Z]*\n?|```$', '',
                                     r.choices[0].message.content.strip()))
        except Exception as e:
            if i == retry - 1: return None
            time.sleep(2)

MSU_P = """You audit sentence-level units extracted from scientific papers. Judge each item on five criteria.
Be reasonable rather than strict: accept an item if a researcher could use it, even if wording is imperfect.
1. fidelity      - faithful to the source, no invented content
2. selfcontained - understandable on its own, without most surrounding text
3. condition     - keeps method/experiment conditions affecting interpretation (true if none apply)
4. traceable     - can be located back to its source paragraph
5. role          - the discourse-role label is reasonable
Return ONLY a JSON array, same order:
[{"id":<id>,"fidelity":true,"selfcontained":true,"condition":true,"traceable":true,"role":true}]
ITEMS:
%s"""

PAIR_P = """You audit candidate pairs of statements from DIFFERENT scientific papers.
Question: do these two form a candidate worth inspecting together in a cross-paper comparison?
Do NOT judge whether they already constitute a confirmed scientific conclusion. Be reasonable: a shared
phenomenon, related method, complementary evidence, or shared analytical concept all qualify.
Label each: "accepted", "uncertain", or "invalid".
Return ONLY a JSON array, same order: [{"id":<id>,"label":"accepted"}]
PAIRS:
%s"""

def run(tag, kind, order, rnd):
    S, db, out = SAMP[tag], SAMP[tag]['db'], {}
    for s in range(0, len(order), 10):
        ch = order[s:s+10]
        if kind == 'msu':
            body = "\n".join(json.dumps({"id": i, "role_label": db[i].get("category"),
                    "text": (db[i].get("sentence") or "")[:400],
                    "source": str(db[i].get("paragraph_info") or "")[:200]}, ensure_ascii=False) for i in ch)
            r = ask(MSU_P % body)
        else:
            body = "\n".join(json.dumps({"id": k, "A": (db[a].get("sentence") or "")[:280],
                    "B": (db[b].get("sentence") or "")[:280]}, ensure_ascii=False)
                    for k, (a, b, c) in ch)
            r = ask(PAIR_P % body)
        for o in (r or []):
            if isinstance(o, dict) and 'id' in o: out[int(o['id'])] = o
        if (s // 10) % 15 == 0:
            print(f'  [{tag}/{kind}/{rnd}] {s+len(ch)}/{len(order)}', flush=True)
    return out

RES = {}
for tag, _ in CORP:
    for rnd in ['r1', 'r2']:
        mo = SAMP[tag]['msu'][:]
        po = list(enumerate(SAMP[tag]['pairs']))
        if rnd == 'r2':
            random.seed(7); random.shuffle(mo); random.shuffle(po)
        RES[f'{tag}_msu_{rnd}']  = run(tag, 'msu',  mo, rnd)
        RES[f'{tag}_pair_{rnd}'] = run(tag, 'pair', po, rnd)

json.dump({k: {str(i): v for i, v in d.items()} for k, d in RES.items()},
          open('results/audit_multi_raw.json', 'w'))
json.dump({t: {'n_msu': SAMP[t]['n_msu'], 'n_pair': SAMP[t]['n_pair'],
               'msu_sampled': len(SAMP[t]['msu']), 'pair_sampled': len(SAMP[t]['pairs']),
               'held_in_sample': sum(1 for a, b, c in SAMP[t]['pairs'] if (a, b) in SAMP[t]['held'])}
           for t, _ in CORP}, open('results/audit_multi_meta.json', 'w'))
print('\n[saved] results/audit_multi_raw.json')
