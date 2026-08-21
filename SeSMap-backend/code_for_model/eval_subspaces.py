import json, numpy as np, sys, collections
sys.path.insert(0,'.')
from sklearn.neighbors import NearestNeighbors
from code_for_model.models import evaluate_quick

X  = np.load('data/stages/04_embeddings/emb_corpus.npy').astype(np.float32)
db = json.load(open('data/stages/02_msu/formdatabase.json'))
role = np.array([(r.get('category') or 'None') for r in db][:X.shape[0]])
pj = json.load(open('data/stages/02_msu/llm_pairs_big.json'))
test = [(int(a),int(b)) for a,b,*_ in pj['test_pos']]
Z0, Z1 = np.load('/tmp/Z_lam0.0.npy'), np.load('/tmp/Z_lam0.1.npy')

def coloc(space, pos, k=12):
    if not pos: return float('nan')
    nn=NearestNeighbors(n_neighbors=k+1).fit(space); _,ind=nn.kneighbors(space)
    ng=[set(r[1:]) for r in ind]
    return sum(1 for a,b in pos if b in ng[a] or a in ng[b])/len(pos)

ROLES=['Background','Method','Experiment','Result','Conclusion']

print("="*78)
print("A. 每个子空间内部的布局质量（只取该角色的 MSU 重算近邻）")
print("="*78)
print(f"{'subspace':14}{'N':>7}{'Trust':>9}{'knnOv@12':>11}")
print(f"{'ALL (global)':14}{X.shape[0]:>7}", end="")
tw,ko=evaluate_quick(X,Z1,12); print(f"{tw:>9.3f}{ko:>11.3f}")
for r in ROLES:
    m=np.where(role==r)[0]
    if len(m)<60: continue
    tw,ko=evaluate_quick(X[m],Z1[m],12)
    print(f"{r:14}{len(m):>7}{tw:>9.3f}{ko:>11.3f}")

print()
print("="*78)
print("B. 同角色 vs 跨角色 的对应关系共位（267 对留出）")
print("="*78)
same=[(a,b) for a,b in test if role[a]==role[b]]
cross=[(a,b) for a,b in test if role[a]!=role[b]]
print(f"{'pairs':22}{'n':>6}{'ours':>10}{'+sup':>10}{'gain':>9}")
for nm,ps in [('same-role',same),('cross-role',cross),('all',test)]:
    c0,c1=coloc(Z0,ps),coloc(Z1,ps)
    print(f"{nm:22}{len(ps):>6}{c0:>10.3f}{c1:>10.3f}{(c1/c0-1)*100:>8.0f}%")

print()
print("="*78)
print("C. 同角色对应能否在“该角色子空间视图内”被找到")
print("   （只在该角色的 MSU 子集里重算 12 近邻，模拟分析者只看一个子空间）")
print("="*78)
print(f"{'subspace':14}{'pairs':>7}{'global view':>13}{'subspace view':>15}")
for r in ROLES:
    ps=[(a,b) for a,b in same if role[a]==r]
    if len(ps)<8: continue
    m=np.where(role==r)[0]; idx={g:i for i,g in enumerate(m)}
    local=[(idx[a],idx[b]) for a,b in ps]
    print(f"{r:14}{len(ps):>7}{coloc(Z1,ps):>13.3f}{coloc(Z1[m],local):>15.3f}")
