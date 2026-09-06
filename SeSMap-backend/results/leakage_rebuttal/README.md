# Leakage / circularity rebuttal — Co-loc@12

Reviewer objection, in two parts:

1. **Circularity.** Training supervision and the Co-loc test set both come from
   the same LLM correspondence heuristic, so "were these pairs pulled together"
   may only measure that the model learned the heuristic.
2. **MSU-level leakage.** Sec. 5.3 notes that individual MSUs recur across pairs
   in both sets, so held-out pairs are not independent of training pairs.

## What is in here

| file | purpose |
|---|---|
| `build_splits.py` | MSU-disjoint re-splits (stratified by paper, 5 seeds) |
| `run_disjoint.py` | retrain projector on each disjoint split, score Co-loc@12 |
| `replay_audit.py` | deterministically replay `code_for_data/audit_multi.py` sampling so audit labels map back to concrete pairs |
| `eval_audit_subset.py` | Co-loc on audit-verified / leakage-free subsets, published layouts |
| `stats_audit.py` | paired bootstrap on those subsets |
| `aggregate.py` | per-seed + pooled results for the disjoint retraining |
| `check_difficulty.py` | are disjoint test pairs harder? (guards the main claim) |

Run order: `build_splits` → `replay_audit` → `eval_audit_subset` → `stats_audit`
→ `run_disjoint` → `aggregate`. `check_difficulty` is independent.

`run_disjoint.py` skips runs whose `Zdis_*.npy` already exists, so an
interrupted sweep resumes rather than restarting.

## Measured leakage in the released split

| quantity | value |
|---|---|
| test MSUs also appearing in a training pair | 779 / 1042 = **74.8%** |
| test pairs with **both** endpoints seen in training | 485 / 744 = **65.2%** |
| test pairs with neither endpoint seen | 31 / 744 = 4.2% |
| exact duplicate pairs between train and test | **0** |

So the leak is at MSU level, not pair level.

## Caveats that must not be lost

* **The correspondence audit is LLM-based.** `code_for_data/audit_multi.py`
  sets `MODEL = 'gpt-4o'` and runs two passes; the reported "raw agreement" and
  Cohen's kappa are between two LLM passes, not between human annotators. Any
  rebuttal leaning on the audit subset must say so, and Sec. 5.3's
  "expert auditing" / "stratified human auditing" wording needs to match
  whatever was actually done.
* **`audit_all` (n=792) and `audit_accepted` (n=583) are not generalisation
  numbers.** About 80% of those pairs are training pairs; the high Co-loc there
  is memorisation and they are kept only as a contrast row.
* **`clamp_min` is silently wrong on large MPS tensors** (leaves the masked
  diagonal at 0, so `log()` yields `-inf` and the KL term becomes `inf`).
  `run_disjoint.py` uses a `torch.where` floor instead, which matches CPU.
  The published models were trained on CPU and are unaffected.
