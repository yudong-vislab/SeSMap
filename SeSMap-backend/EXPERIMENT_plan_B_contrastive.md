# 方案B 实验方案：LLM 增强跨论文语义对比能否改进 v10

**问题**：在 v10(纯 parametric t-SNE)之上，加入"LLM 挖掘的跨论文语义等价对"作对比监督，
能否**在不损害保真的前提下，把真正的跨论文语义对应更好地放到一起（boundary zone 更强）**？

**核心指标**（held-out test 对，不参与训练）：
- `knnOv@12` / `trust`：保真。**不应显著低于 v10**(0.303 / 0.930)。
- `bz_recall@12 (2D)`：held-out 跨论文语义等价对在 2D 的近邻召回。**这是要赢 v10 的地方。**
- `bz_recall@12 (highD)`：bge 自身的天花板参考。
- `paperSil`：论文分离度。**应保持低**(≈0，不能因为对比而把论文推开)。

---

## 步骤 0 — 生成真实的 LLM 语义等价对（一次）

```bash
python3 code_for_data/augment_pairs_llm.py --max-candidates 3000 --judge-batch 10
# 产出 data/stages/02_msu/llm_pairs.json（含 train_pos / test_pos）
```
- 机制：bge 余弦在**跨论文**候选(cosine∈[0.55,0.95]) → LLM 判定 equivalent/corresponding → 正样本。
- 成本控制：先 `--max-candidates 1500` 看质量/花费；判定 batch=10 约 150 次调用。
- **专家抽查**：可人工过一遍 `llm_pairs.json` 的正样本，剔除误判（就是你说的"专家抽查监督"，可写进方法）。

## 步骤 1 — 训练三个臂

```bash
# A0 基线（复现 v10；也可直接用现成 v10.pt）
python3 code_for_model/train_contrastive_v11.py --pairs data/stages/02_msu/llm_pairs.json \
    --lambda-tsne 1 --lambda-con 0 --out data/stages/05_model/v11_tsne_only.pt

# A1 混合（主臂；扫 lambda-con ∈ {0.1, 0.3, 1.0}）
python3 code_for_model/train_contrastive_v11.py --pairs data/stages/02_msu/llm_pairs.json \
    --lambda-tsne 1 --lambda-con 0.3 --out data/stages/05_model/v11_hybrid_0.3.pt

# A2 纯对比（消融；预期全局结构差 → 证明混合是必要的）
python3 code_for_model/train_contrastive_v11.py --pairs data/stages/02_msu/llm_pairs.json \
    --lambda-tsne 0 --lambda-con 1 --out data/stages/05_model/v11_con_only.pt
```
每次全批 1500 步，CPU 约几分钟。

## 步骤 2 — 同尺子评测

```bash
for m in bert2d_mapper_all_v10 v11_tsne_only v11_hybrid_0.3 v11_con_only; do
  python3 code_for_model/eval_contrastive.py \
    --ckpt data/stages/05_model/$m.pt --pairs data/stages/02_msu/llm_pairs.json
done
```

## 步骤 3 — 填表对比

| 模型 | knnOv@12 | trust | paperSil | **bz_recall@12 2D** | highD 参考 |
|---|---|---|---|---|---|
| v10（基线） | 0.303 | 0.930 | ~0 | ? | ? |
| 混合 λ=0.1 | ? | ? | ? | ? | ? |
| 混合 λ=0.3 | ? | ? | ? | ? | ? |
| 混合 λ=1.0 | ? | ? | ? | ? | ? |
| 纯对比 | ? | ? | ? | ? | ? |

---

## 判定规则（决定写不写进正文）

**✅ 成功（写进正文当贡献）**：某个 λ 的混合臂满足——
- `bz_recall@12 2D` 相对 v10 **明显提升**（建议 ≥ +0.05 绝对值），**且**
- `knnOv@12` 不明显下降（保持 ≥ ~0.28，即相对 v10 掉幅 <8%），**且**
- `paperSil` 仍低（< 0.1，没把论文推开）。
→ 结论："LLM 增强的跨论文语义监督 + 参数化 t-SNE，在保真不降的前提下显著改善跨论文语义对应的空间共址（BZ）"。这是可扩展(LLM 生成)、on-thesis(强化 BZ)的新贡献。

**❌ 不成功（当 ablation 放附录，正文保持 v10）**：
- 若 `bz_recall` 不升 → LLM 对没带来 bge 之外的信号；
- 或 `knnOv` 明显掉 / `paperSil` 明显升 → 对比在伤保真/在推分离 → 调小 λ 再试，仍不行就放弃。

**纯对比臂（A2）预期表现差**（全局结构差、knnOv 低）——这本身是有用的 ablation：证明"对比单独不够，必须和 t-SNE 保真目标混合"。

---

## 诚实前提（别自欺）
- bge 是"忠实还原 bge 邻域"的天花板，所以**不要期望 knnOv 超过 v10**；方案B 的价值只可能体现在 `bz_recall`（把 bge 也没放好的跨论文对应拉到一起）。
- 因此**真实 LLM 判定的对**很关键：要包含"语义对应但用词不同"的难例（highD 参考没那么高的那些），2D 提升才有意义。dry-run 的纯高余弦对会让 highD 参考虚高。
- 正文写什么 = 实际跑什么 = 报告哪套数字。赢了才写成贡献，输了就如实放附录。
