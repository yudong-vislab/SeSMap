#!/usr/bin/env python3
"""A+B（13 篇）的 discourse-role 子空间量化评测。各语料自拟合投影器，逐语料计算后合并。"""
import json, sys, collections
import numpy as np, torch
sys.path.insert(0,'.')
from sklearn.neighbors import NearestNeighbors
from code_for_model.models import ResidualProjectionMapper, evaluate_quick, set_seed, compute_joint_P

K=12
CORP=[('A','data/stages','llm_pairs_10k.json'),('B','data/general_v11/stages','llm_pairs.json')]
ROLES=['Background','Method','Experiment','Result','Conclusion']

def norm_role(r):
    """归一化两套命名：METHOD/Method -> Method, EXPERIMENT/SETUP -> Experiment ..."""
    s=(r or '').strip().upper()
    if s.startswith('BACKGROUND'): return 'Background'
    if s.startswith('METHOD'):     return 'Method'
    if s.startswith('EXPERIMENT'): return 'Experiment'
    if s.startswith('RESULT'):     return 'Result'
    if s.startswith('CONCLUSION'): return 'Conclusion'
    return 'Other'

def coloc(Z,pos,k=K):
    if not pos: return float('nan')
    nn=NearestNeighbors(n_neighbors=min(k,len(Z)-1)+1).fit(Z); _,ind=nn.kneighbors(Z)
    ng=[set(r[1:]) for r in ind]
    return sum(1 for a,b in pos if b in ng[a] or a in ng[b])/len(pos)

def train(X,pairs,lam,iters=1500):
    n=X.shape[0]; set_seed(0)
    x=torch.tensor(X); P=torch.tensor(compute_joint_P(X,30.0)); eye=torch.eye(n,dtype=torch.bool)
    m=ResidualProjectionMapper(X.shape[1],512,4,0.1); opt=torch.optim.Adam(m.parameters(),1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,iters); ns=0.4/(X.shape[1]**0.5); m.train()
    if lam>0 and pairs:
        pa=torch.tensor([a for a,b in pairs]); pb=torch.tensor([b for a,b in pairs])
    for it in range(iters):
        Z=m(x+ns*torch.randn_like(x)); D2=torch.cdist(Z,Z)**2
        num=(1/(1+D2)).masked_fill(eye,0); Q=(num/num.sum().clamp_min(1e-12)).clamp_min(1e-12)
        Pe=P*12 if it<250 else P; Pn=(Pe/Pe.sum()).clamp_min(1e-12)
        loss=(Pn*(Pn.log()-Q.log())).sum()
        if lam>0 and pairs:
            Qr=(num/num.sum(1,keepdim=True).clamp_min(1e-12)).clamp_min(1e-12)
            loss=loss+lam*0.5*(-Qr[pa,pb].log().mean()-Qr[pb,pa].log().mean())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5)
        opt.step(); sch.step()
    m.eval()
    with torch.no_grad(): return m(torch.tensor(X)).cpu().numpy()

ACC=collections.defaultdict(lambda: {'msu':0,'pairs':0,'g_hit':0,'s_hit':0,'tw':[],'ko':[],'w':[]})
SAME={'ours':[0,0],'sup':[0,0]}; CROSS={'ours':[0,0],'sup':[0,0]}
for tag,base,pf in CORP:
    X=np.load(f'{base}/04_embeddings/emb_corpus.npy').astype(np.float32)
    db=json.load(open(f'{base}/02_msu/formdatabase.json'))
    role=np.array([norm_role(r.get('category')) for r in db][:X.shape[0]])
    pj=json.load(open(f'{base}/02_msu/{pf}'))
    tr=[(int(a),int(b)) for a,b,*_ in pj['train_pos']]
    te=[(int(a),int(b)) for a,b,*_ in pj['test_pos']]
    print(f'[{tag}] N={X.shape[0]} train={len(tr)} held-out={len(te)}  训练中...',flush=True)
    Z0=train(X,tr,0.0); Z1=train(X,tr,0.1)
    same=[(a,b) for a,b in te if role[a]==role[b]]
    cross=[(a,b) for a,b in te if role[a]!=role[b]]
    for key,ps in [('ours',same)]: pass
    SAME['ours'][0]+=coloc(Z0,same)*len(same); SAME['ours'][1]+=len(same)
    SAME['sup'][0] +=coloc(Z1,same)*len(same); SAME['sup'][1] +=len(same)
    CROSS['ours'][0]+=coloc(Z0,cross)*len(cross); CROSS['ours'][1]+=len(cross)
    CROSS['sup'][0] +=coloc(Z1,cross)*len(cross); CROSS['sup'][1] +=len(cross)
    for r in ROLES:
        m=np.where(role==r)[0]
        ps=[(a,b) for a,b in same if role[a]==r]
        if len(m)<60: continue
        tw,ko=evaluate_quick(X[m],Z1[m],K)
        A=ACC[r]; A['msu']+=len(m); A['tw'].append(tw); A['ko'].append(ko); A['w'].append(len(m))
        if ps:
            idx={g:i for i,g in enumerate(m)}
            A['pairs']+=len(ps)
            A['g_hit']+=coloc(Z1,ps)*len(ps)
            A['s_hit']+=coloc(Z1[m],[(idx[a],idx[b]) for a,b in ps])*len(ps)
    print(f'[{tag}] 完成',flush=True)

print('\n'+'='*74); print('A+B (13 papers) — discourse-role subspaces'); print('='*74)
print(f"{'Subspace':14}{'MSUs':>7}{'held-out':>10}{'Trust':>9}{'knnOv':>9}{'global':>9}{'in-sub':>9}{'ratio':>8}")
for r in ROLES:
    A=ACC[r]
    if not A['w']: continue
    tw=np.average(A['tw'],weights=A['w']); ko=np.average(A['ko'],weights=A['w'])
    if A['pairs']>=10:
        g=A['g_hit']/A['pairs']; s=A['s_hit']/A['pairs']
        print(f"{r:14}{A['msu']:>7}{A['pairs']:>10}{tw:>9.3f}{ko:>9.3f}{g:>9.3f}{s:>9.3f}{s/g if g>0 else float('nan'):>8.2f}")
    else:
        print(f"{r:14}{A['msu']:>7}{A['pairs']:>10}{tw:>9.3f}{ko:>9.3f}{'--':>9}{'--':>9}{'--':>8}")
print('\n'+'-'*74); print('同角色 vs 跨角色'); print('-'*74)
print(f"{'Pair type':14}{'n':>7}{'Ours':>9}{'+sup':>9}{'gain':>9}")
for nm,D in [('Same-role',SAME),('Cross-role',CROSS)]:
    o=D['ours'][0]/D['ours'][1]; s=D['sup'][0]/D['sup'][1]
    print(f"{nm:14}{D['ours'][1]:>7}{o:>9.3f}{s:>9.3f}{(s/o-1)*100:>8.0f}%")
