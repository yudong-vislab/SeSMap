#!/usr/bin/env python3
"""Retrain the projector on MSU-disjoint splits and re-measure Co-loc@12.

Mirrors train_contrastive_v11.py (same architecture, losses, schedule) and
eval_merged13.py (same hits()/Co-loc definition), changing ONE thing: the
positive pairs used for InfoNCE come from an MSU-disjoint split, so held-out
pairs share no MSU with the training pairs.

NOTE on `_floor`: torch's clamp_min silently fails on large MPS tensors (it
leaves the 6113 masked diagonal entries at 0, so log() returns -inf and the KL
term becomes inf).  torch.where is correct on both devices and identical to
clamp_min on CPU, so it is used throughout.

Completed runs are skipped, so an interrupted sweep can be resumed.
"""
import json, sys, time, argparse
from pathlib import Path
import numpy as np, torch, torch.optim as optim

BK = Path('/Users/yudong/Desktop/SeSMap/SeSMap-backend')
sys.path.insert(0, str(BK))
from code_for_model.models import ResidualProjectionMapper, set_seed
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).parent
EMB = BK / 'data/merged13/stages/04_embeddings/emb_corpus.npy'
PCACHE = HERE / 'P_merged13.npy'
K = 12
EPS = 1e-12


def _floor(t, eps=EPS):
    """clamp_min replacement that is correct on MPS as well as CPU."""
    return torch.where(t < eps, torch.full_like(t, eps), t)


def hits(Z, pairs, k=K):
    """Identical to eval_merged13.hits: symmetric top-k co-location indicator."""
    ng = [set(r) for r in NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)]
    return np.array([1.0 if (b in ng[a] or a in ng[b]) else 0.0 for a, b in pairs])


def train(X, P, train_pos, seed, lam_con, lam_tsne=1.0, iters=1500, device='cpu',
          width=512, blocks=4, dropout=0.1, input_noise=0.4, lr=1e-3,
          exaggeration=12.0, exag_iters=250, log=True):
    set_seed(seed)
    dev = torch.device(device)
    n = X.shape[0]
    x = torch.tensor(X, dtype=torch.float32, device=dev)
    eye = torch.eye(n, dtype=torch.bool, device=dev)
    Pt = torch.tensor(P, dtype=torch.float32, device=dev)
    pa = torch.tensor([a for a, b in train_pos], dtype=torch.long, device=dev)
    pb = torch.tensor([b for a, b in train_pos], dtype=torch.long, device=dev)

    m = ResidualProjectionMapper(embed_dim=X.shape[1], width=width,
                                 num_blocks=blocks, dropout=dropout).to(dev)
    opt = optim.Adam(m.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    noise = input_noise / (X.shape[1] ** 0.5) if input_noise > 0 else 0.0

    m.train()
    for it in range(iters):
        xin = x + noise * torch.randn_like(x) if noise > 0 else x
        Z = m(xin)
        num = (1.0 / (1.0 + torch.cdist(Z, Z) ** 2)).masked_fill(eye, 0.0)
        loss = torch.zeros((), device=dev)
        if lam_tsne > 0:
            Q = _floor(num / _floor(num.sum()))
            Pe = Pt * exaggeration if it < exag_iters else Pt
            Pn = _floor(Pe / Pe.sum())
            loss = loss + lam_tsne * (Pn * (Pn.log() - Q.log())).sum()
        if lam_con > 0 and len(train_pos) > 0:
            Qr = _floor(num / _floor(num.sum(dim=1, keepdim=True)))
            loss = loss + lam_con * 0.5 * (-Qr[pa, pb].log().mean() - Qr[pb, pa].log().mean())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step(); sched.step()
        if log and (it == 0 or (it + 1) % 500 == 0):
            assert torch.isfinite(loss), f'non-finite loss at it {it}'
            print(f'      it {it+1:4d}/{iters} loss={loss.item():.4f}', flush=True)

    m.eval()
    with torch.no_grad():
        return m(x).cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--lams', type=float, nargs='+', default=[0.0, 0.3])
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--tag', default='')
    ap.add_argument('--splits', type=Path, default=HERE / 'splits_disjoint.json')
    ap.add_argument('--out', type=Path, default=HERE / 'disjoint_results.json')
    a = ap.parse_args()

    X = np.load(EMB).astype(np.float32)
    P = np.load(PCACHE)
    S = json.load(open(a.splits))
    res = json.loads(a.out.read_text()) if a.out.exists() else {}

    for s in a.seeds:
        sp = S[str(s)]
        tr = [tuple(x) for x in sp['train']]
        te = [tuple(x) for x in sp['test']]
        res.setdefault(str(s), {})['n_train'] = len(tr)
        res[str(s)]['n_test'] = len(te)
        for lam in a.lams:
            zf = HERE / f'Zdis{a.tag}_s{s}_lam{lam}.npy'
            if zf.exists():
                print(f'[seed {s}] lam_con={lam}  SKIP (exists)', flush=True)
                continue
            t0 = time.time()
            print(f'[seed {s}] lam_con={lam}  train={len(tr)} test={len(te)} dev={a.device}', flush=True)
            Z = train(X, P, tr, seed=s, lam_con=lam, iters=a.iters, device=a.device)
            h = hits(Z, te)
            res[str(s)][f'lam{lam}'] = {'coloc': float(h.mean())}
            np.save(zf, Z)
            np.save(HERE / f'hits{a.tag}_s{s}_lam{lam}.npy', h)
            print(f'      -> Co-loc@12 = {h.mean():.4f}   ({time.time()-t0:.0f}s)', flush=True)
            a.out.write_text(json.dumps(res, indent=2))
    a.out.write_text(json.dumps(res, indent=2))
    print(f'\nsaved -> {a.out}')


if __name__ == '__main__':
    raise SystemExit(main())
