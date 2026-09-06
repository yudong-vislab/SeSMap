#!/usr/bin/env python3
"""Deterministically replay the sampling in code_for_data/audit_multi.py so each
audit label can be traced back to the concrete MSU pair it judged, then map those
pairs from per-corpus row ids into merged13 row ids.

audit_multi.py seeds `random` once and consumes it in a fixed order
(A-msu, A-pair, B-msu, B-pair, C-msu, C-pair), so replaying the same code path
reproduces the exact samples.  We verify the replay against the recorded counts
in results/audit_multi_meta.json before using it.
"""
import json, os, random, collections
from pathlib import Path
import numpy as np

BK = Path('/Users/yudong/Desktop/SeSMap/SeSMap-backend')
HERE = Path(__file__).parent
FRAC, SEED = 0.20, 20260827
CORP = [('A', 'data/stages'), ('B', 'data/general_v11/stages'), ('C', 'data/bio_eval/stages')]


def band(c):
    return '0.55-0.65' if c < .65 else '0.65-0.75' if c < .75 else '0.75-0.85' if c < .85 else '0.85-0.95'


def replay():
    random.seed(SEED)
    SAMP = {}
    for tag, base in CORP:
        b = BK / base
        db = json.load(open(b / '02_msu/formdatabase.json'))
        X = np.load(b / '04_embeddings/emb_corpus.npy').astype(np.float32)
        N = X.shape[0]
        pf = next(f for f in ['llm_pairs_10k.json', 'llm_pairs.json']
                  if os.path.exists(b / '02_msu' / f))
        pj = json.load(open(b / '02_msu' / pf))

        st = collections.defaultdict(list)
        for i, r in enumerate(db[:N]):
            if len((r.get('sentence') or '').split()) < 3:
                continue
            st[(r.get('paper_id'), r.get('category') or 'None')].append(i)
        midx = sorted({j for ids in st.values()
                       for j in random.sample(ids, max(1, round(len(ids) * FRAC)))})

        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        role = [(r.get('category') or 'None') for r in db[:N]]
        pos = [(int(a), int(b)) for a, b, *_ in pj['train_pos'] + pj['test_pos']
               if int(a) < N and int(b) < N]
        held = {(int(a), int(b)) for a, b, *_ in pj['test_pos']}
        ps = collections.defaultdict(list)
        for a, b in pos:
            c = float(Xn[a] @ Xn[b])
            ps[(band(c), tuple(sorted((role[a], role[b]))))].append((a, b, c))
        psamp = [x for lst in ps.values() for x in random.sample(lst, max(1, round(len(lst) * FRAC)))]

        SAMP[tag] = {'msu': midx, 'pairs': psamp, 'held': held, 'n_msu': N,
                     'n_pair': len(pos), 'pairs_file': pf}
        print(f'[{tag}] replay: MSU {len(midx)}/{N}   pair {len(psamp)}/{len(pos)}   '
              f'held_in_sample={sum(1 for a,b,c in psamp if (a,b) in held)}')
    return SAMP


def verify(SAMP):
    meta = json.load(open(BK / 'results/audit_multi_meta.json'))
    ok = True
    for t in ['A', 'B', 'C']:
        m, s = meta[t], SAMP[t]
        got = (s['n_msu'], s['n_pair'], len(s['msu']), len(s['pairs']),
               sum(1 for a, b, c in s['pairs'] if (a, b) in s['held']))
        exp = (m['n_msu'], m['n_pair'], m['msu_sampled'], m['pair_sampled'], m['held_in_sample'])
        match = got == exp
        ok &= match
        print(f'  {t}: replay={got}  recorded={exp}  {"MATCH" if match else "MISMATCH"}')
    return ok


def merged13_map():
    """merged13 row -> (corpus tag, corpus-local row), validated on sentence text."""
    mm = json.load(open(BK / 'data/merged13/stages/02_msu/llm_pairs.json'))['meta']
    nA, excl = mm['n_A'], set(mm.get('excluded_B_rows', []))
    fdb_m = json.load(open(BK / 'data/merged13/stages/02_msu/formdatabase_v2.0.json'))
    fdb_A = json.load(open(BK / 'data/stages/02_msu/formdatabase.json'))
    fdb_B = json.load(open(BK / 'data/general_v11/stages/02_msu/formdatabase.json'))

    fwd = {}
    for i in range(nA):
        fwd[('A', i)] = i
    row = nA
    nB_total = np.load(BK / 'data/general_v11/stages/04_embeddings/emb_corpus.npy').shape[0]
    for j in range(nB_total):
        if j in excl:
            continue
        fwd[('B', j)] = row
        row += 1
    assert row == len(fdb_m), f'merged rows {row} != formdatabase {len(fdb_m)}'

    bad = sum(1 for (tag, loc), mr in fwd.items()
              if (fdb_A if tag == 'A' else fdb_B)[loc].get('sentence') != fdb_m[mr].get('sentence'))
    print(f'  merged13 map: {len(fwd)} rows, sentence mismatches = {bad}')
    return fwd, bad


if __name__ == '__main__':
    S = replay()
    print('\nverify against results/audit_multi_meta.json:')
    ok = verify(S)
    print(f'\nreplay verified: {ok}')
    print('\nbuild merged13 index map:')
    fwd, bad = merged13_map()

    out = {t: {'pairs': [[int(a), int(b), float(c)] for a, b, c in S[t]['pairs']],
               'held': sorted([list(map(int, x)) for x in S[t]['held']])}
           for t in ['A', 'B', 'C']}
    out['_map'] = {f'{t}:{l}': m for (t, l), m in fwd.items()}
    out['_verified'] = bool(ok and bad == 0)
    (HERE / 'audit_replay.json').write_text(json.dumps(out))
    print(f"\nsaved -> {HERE/'audit_replay.json'}   verified={out['_verified']}")
