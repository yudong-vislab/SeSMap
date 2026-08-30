#!/usr/bin/env python3
"""汇总审计结果：一致率、Cohen's kappa、各标准通过率。"""
import json, collections

R = json.load(open('results/audit_raw.json'))
S = json.load(open('data/stages/02_msu/audit_sample.json'))
CRIT = ['fidelity','selfcontained','condition','traceable','role']
NAME = {'fidelity':'semantic fidelity','selfcontained':'self-containedness',
        'condition':'condition preservation','traceable':'source traceability',
        'role':'role accuracy'}

def kappa(a, b):
    """两轮二元判定的 Cohen's kappa。"""
    n = len(a)
    if n == 0: return float('nan')
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c)/n) * (b.count(c)/n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

# ================= MSU =================
m1, m2 = R['msu_r1'], R['msu_r2']
ids = sorted(set(m1) & set(m2), key=int)
print("="*54); print("MSU AUDIT"); print("="*54)
print(f"n = {len(ids)} MSUs")

flat1 = [bool(m1[i].get(c)) for i in ids for c in CRIT]
flat2 = [bool(m2[i].get(c)) for i in ids for c in CRIT]
agree = sum(1 for x, y in zip(flat1, flat2) if x == y) / len(flat1)
print(f"raw agreement = {agree*100:.1f}%")
print(f"Cohen's kappa = {kappa(flat1, flat2):.3f}")
print()
# 采纳两轮均通过为准（保守合取）；两轮不一致时以第一轮为准
def val(i, c):
    return bool(m1[i].get(c)) and bool(m2[i].get(c))
for c in CRIT:
    p = sum(val(i, c) for i in ids) / len(ids)
    print(f"{NAME[c]:24} = {p*100:.1f}%")
allc = sum(all(val(i, c) for c in CRIT) for i in ids) / len(ids)
print(f"{'all-criterion acceptance':24} = {allc*100:.1f}%")

# ================= PAIR =================
p1, p2 = R['pair_r1'], R['pair_r2']
pids = sorted(set(p1) & set(p2), key=int)
print(); print("="*54); print("CORRESPONDENCE AUDIT"); print("="*54)
print(f"n = {len(pids)} pairs")
l1 = [str(p1[i].get('label','uncertain')) for i in pids]
l2 = [str(p2[i].get('label','uncertain')) for i in pids]
ag = sum(1 for x, y in zip(l1, l2) if x == y) / len(l1)
print(f"raw agreement = {ag*100:.1f}%")
print(f"Cohen's kappa = {kappa(l1, l2):.3f}")
print()
# 两轮一致才计入该类；不一致计入 uncertain
final = [a if a == b else 'uncertain' for a, b in zip(l1, l2)]
cnt = collections.Counter(final)
for lab, disp in [('accepted','accepted candidate pairs'),
                  ('uncertain','uncertain pairs'),
                  ('invalid','invalid pairs')]:
    print(f"{disp:26} = {cnt.get(lab,0)/len(final)*100:.1f}%   (n={cnt.get(lab,0)})")
print()
print(f"audit precision = {cnt.get('accepted',0)}/{len(final)} = {cnt.get('accepted',0)/len(final):.3f}")
print("  (uncertain 单独报告，不并入 accepted)")

held = sum(1 for x in S['pairs'] if x[3])
print(f"\n抽样中来自留出集的对数 = {held}")
