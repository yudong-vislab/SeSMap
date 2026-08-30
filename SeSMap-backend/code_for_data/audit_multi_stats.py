#!/usr/bin/env python3
import json, collections
R = json.load(open('results/audit_multi_raw.json'))
M = json.load(open('results/audit_multi_meta.json'))
CRIT = ['fidelity','selfcontained','condition','traceable','role']
NAME = {'fidelity':'semantic fidelity','selfcontained':'self-containedness',
        'condition':'condition preservation','traceable':'source traceability','role':'role accuracy'}
LAB  = {'A':'A  atmospheric / air quality  (6 papers)',
        'B':'B  spatiotemporal + env.      (7 papers)',
        'C':'C  genomics visualization     (12 papers)'}

def kappa(a,b):
    n=len(a)
    if not n: return float('nan')
    po=sum(1 for x,y in zip(a,b) if x==y)/n
    cats=set(a)|set(b)
    pe=sum((a.count(c)/n)*(b.count(c)/n) for c in cats)
    return (po-pe)/(1-pe) if pe<1 else 1.0

print("="*68); print("MSU AUDIT"); print("="*68)
pool1=pool2=[]; pool1,pool2=[],[]; pool_ids=[]
rows=[]
for t in ['A','B','C']:
    m1,m2=R[f'{t}_msu_r1'],R[f'{t}_msu_r2']
    ids=sorted(set(m1)&set(m2),key=int)
    f1=[bool(m1[i].get(c)) for i in ids for c in CRIT]
    f2=[bool(m2[i].get(c)) for i in ids for c in CRIT]
    pool1+=f1; pool2+=f2
    val=lambda i,c: bool(m1[i].get(c)) and bool(m2[i].get(c))
    per={c:sum(val(i,c) for i in ids)/len(ids) for c in CRIT}
    allc=sum(all(val(i,c) for c in CRIT) for i in ids)/len(ids)
    ag=sum(1 for x,y in zip(f1,f2) if x==y)/len(f1)
    rows.append((t,len(ids),ag,kappa(f1,f2),per,allc))
    print(f"\n{LAB[t]}")
    print(f"  n = {len(ids)} MSUs   raw agreement = {ag*100:.1f}%   Cohen's k = {kappa(f1,f2):.3f}")
    for c in CRIT: print(f"    {NAME[c]:24} = {per[c]*100:.1f}%")
    print(f"    {'all-criterion acceptance':24} = {allc*100:.1f}%")
ag=sum(1 for x,y in zip(pool1,pool2) if x==y)/len(pool1)
tn=sum(r[1] for r in rows)
print(f"\n{'POOLED (25 papers)':34}")
print(f"  n = {tn} MSUs   raw agreement = {ag*100:.1f}%   Cohen's k = {kappa(pool1,pool2):.3f}")
for c in CRIT:
    p=sum(r[4][c]*r[1] for r in rows)/tn
    print(f"    {NAME[c]:24} = {p*100:.1f}%")
print(f"    {'all-criterion acceptance':24} = {sum(r[5]*r[1] for r in rows)/tn*100:.1f}%")

print("\n"+"="*68); print("CORRESPONDENCE AUDIT"); print("="*68)
P1,P2=[],[]; prows=[]
for t in ['A','B','C']:
    p1,p2=R[f'{t}_pair_r1'],R[f'{t}_pair_r2']
    ids=sorted(set(p1)&set(p2),key=int)
    l1=[str(p1[i].get('label','uncertain')) for i in ids]
    l2=[str(p2[i].get('label','uncertain')) for i in ids]
    P1+=l1; P2+=l2
    fin=[a if a==b else 'uncertain' for a,b in zip(l1,l2)]
    cnt=collections.Counter(fin); n=len(fin)
    ag=sum(1 for x,y in zip(l1,l2) if x==y)/n
    prows.append((t,n,ag,kappa(l1,l2),cnt))
    print(f"\n{LAB[t]}")
    print(f"  n = {n} pairs   raw agreement = {ag*100:.1f}%   Cohen's k = {kappa(l1,l2):.3f}")
    for k,d in [('accepted','accepted candidate pairs'),('uncertain','uncertain pairs'),('invalid','invalid pairs')]:
        print(f"    {d:26} = {cnt.get(k,0)/n*100:.1f}%   (n={cnt.get(k,0)})")
    print(f"    audit precision            = {cnt.get('accepted',0)/n:.3f}")
fin=[a if a==b else 'uncertain' for a,b in zip(P1,P2)]
cnt=collections.Counter(fin); n=len(fin)
print(f"\n{'POOLED (25 papers)':34}")
print(f"  n = {n} pairs   raw agreement = {sum(1 for x,y in zip(P1,P2) if x==y)/n*100:.1f}%   Cohen's k = {kappa(P1,P2):.3f}")
for k,d in [('accepted','accepted candidate pairs'),('uncertain','uncertain pairs'),('invalid','invalid pairs')]:
    print(f"    {d:26} = {cnt.get(k,0)/n*100:.1f}%   (n={cnt.get(k,0)})")
print(f"    audit precision            = {cnt.get('accepted',0)/n:.3f}")
print(f"\n抽样中来自各留出集的对数: " + ", ".join(f"{t}={M[t]['held_in_sample']}" for t in ['A','B','C']))
