# Case 1 — Corrected LaTeX Text
**Status:** Factually verified at rhex=18  
**Key changes from original PDF draft:**
1. Para 2: "with no spatial overlap" → "with scattered boundary zones confined to their periphery" (Background has 18 BZs, Method has 38 BZs at rhex=18; dominant pattern IS separate clusters but "no overlap" is false)
2. Para 3: MSU-372 (NOT in BZ at rhex=18, single-paper hex) → MSU-368 (IS in BZ-R1(−9,4), verified P₂ LES); quote updated accordingly
3. Para 3: Added MSU-249 (P₂: OH in cavity shear-layer) as third co-located BZ unit — key evidence grounding the jet-wake label in P₁'s OH-based diagnostic
4. Para 4: Added MSU citation to P₂'s OH report; Background BZ narrative made more concrete (MSU-609 explicitly in Background BZ)

**Cite keys:** Replace `\cite{CITEKEY_P1}` → your actual key for ref [26] (Jia et al. 2026, TemporalFlowViz)
               Replace `\cite{CITEKEY_P2}` → your actual key for ref [51] (Tian et al. 2021, Acta Astronautica)
**Figure refs:** Replace `\ref{fig:case1}` and `\ref{fig:case1stepwise}` with your actual Overleaf label names

---

## LaTeX Source

```latex
\subsection{Case~1: Comparing Visual Analytics and Numerical
Simulation Papers on Scramjet Combustion}

The analyst uploaded $P_1$~\cite{CITEKEY_P1} (a visual analytics paper on scramjet
combustion) and $P_2$~\cite{CITEKEY_P2} (a large-eddy simulation study of dual-mode
scramjet flame dynamics) as PDF files. In the Chat with LLM View,
the analyst issued ``show jet combustion-related papers in gallery'' to
add both papers to the Semantic Source Gallery, then ``show all sub\-spaces in case1'' to create five semantic subspaces; dragging the HSU
aggregation radius slider in real time, the analyst settled on 18\,px. A
preliminary scan found the Experiment subspace dominated by $P_2$
HSUs and the Conclusion subspace sparse; the analyst closed both and
focused on Background, Method, and Result.

Across the three subspaces (Fig.~\ref{fig:case1}), $P_1$ and $P_2$ occupy
largely separate regions in the Background and Method subspaces---pink
and green HSUs form spatially distinct clusters, with scattered boundary
zones confined to their periphery. In the Result subspace, a prominent
boundary zone emerges between the two color regions, marking the first
dense site of cross-paper semantic convergence. This progression from
largely exclusive separation in Background and Method to a shared boundary
in Result is directly readable from the spatial layout without requiring
MSU-level inspection.

\textbf{A boundary zone surfaces cross-domain terminological convergence.}
The analyst expanded the Result boundary zone in the Stepwise
Analysis View (Fig.~\ref{fig:case1stepwise}), revealing MSUs from both papers
co-located in the same projection region. $P_2$ characterizes the
jet-wake combustion mode through physical fluid dynamics (MSU-368:
\textit{``The combustor is at the jet-wake mode in this stage''}),
while $P_1$ independently labels its primary discovered cluster using
the same term through data-driven clustering (MSU-961:
\textit{``the yellow cluster corresponded to the jet-wake combustion mode''}).
A third co-located unit, MSU-249 ($P_2$:
\textit{``Most of the OH radicals are localized in the cavity shear-layer''}),
directly grounds the jet-wake label in the physical OH
distribution that $P_1$ uses as its mode-detection diagnostic.
Published in a combustion engineering journal and a visualization
venue respectively, with no mutual citations, the two papers had
independently assigned identical terminology to the same phenomenon via
entirely different analytical paths. Boundary zone detection localized
this cross-domain correspondence at the MSU level, grounding it in
verifiable sentence-level evidence across two publication communities.

\textbf{A cross-subspace flight makes the latent analytical connection
traceable.} To determine whether the shared label reflected substantive
semantic correspondence beyond surface-level terminology, the analyst
constructed a flight spanning the Background and Result subspaces
(Fig.~\ref{fig:case1stepwise}). In the Background subspace, $P_1$ designates the
hydroxyl radical (OH) field as a primary indicator of combustion activity;
in the Result subspace, $P_2$ reports that OH radicals concentrate in
the cavity shear-layer under the jet-wake mode (MSU-249)---a unit that
co-projects in the same boundary zone as $P_1$'s jet-wake cluster
label (MSU-961). The LLM synthesis over the curated Stepwise Analysis
records confirmed this alignment: the measurement target $P_1$
designated as analytically central is the physical signature by which
$P_2$ characterizes the mode both papers independently name. The flight
rendered this correspondence spatially traceable and preserved it as a
structured, reviewable analytical record.
```

---

## Diff Summary (what changed line by line)

| Original | Corrected | Reason |
|----------|-----------|--------|
| "cluster independently with **no spatial overlap**" | "form spatially distinct clusters, with **scattered boundary zones confined to their periphery**" | 18 BZs exist in Background, 38 in Method at rhex=18 |
| "MSU-372: *'the jet-wake stabilized mode is established when the cavity recirculation zone completely disappears'*" | "MSU-368: *'The combustor is at the jet-wake mode in this stage'*" | MSU-372 is at hex (−10,3), single-paper, NOT in BZ; MSU-368 is at BZ-R1(−9,4) |
| "characterizes a 'jet-wake **stabilized** mode'" | "characterizes the **jet-wake combustion** mode" | Match MSU-961 terminology; drop "stabilized" which was MSU-372's term |
| *(nothing)* | Added "A third co-located unit, MSU-249 (P₂: *'Most of the OH radicals are localized in the cavity shear-layer'*)..." | Verified MSU-249 IS in BZ(−9,4); provides OH-based grounding for jet-wake convergence |
| "P₂ reports that OH radicals concentrate in the cavity shear-layer under the jet-wake mode." | "...(MSU-249)---a unit that co-projects in the same boundary zone as P₁'s jet-wake cluster label (MSU-961)." | Makes the BZ evidence specific and traceable; connects OH measurement to co-location |

---

## Verified Evidence (rhex=18)

| MSU | Paper | Hex | IS_BZ | Sentence |
|-----|-------|-----|-------|---------|
| 368 | P₂-LES (c1) | (−9,4) | ✅ | "The combustor is at the jet-wake mode in this stage." |
| 961 | P₁-viz (c2) | (−9,4) | ✅ | "the yellow cluster corresponded to the jet-wake combustion mode" |
| 249 | P₂-LES (c1) | (−9,4) | ✅ | "Most of the OH radicals are localized in the cavity shear-layer as opposed to the cavity." |
| 609 | P₁-viz (c2) | (−11,3) BG BZ | ✅ | "Traditional scalar metrics (e.g., thrust, equivalence ratio) offer limited insights into combustion mode behavior." |
| 322 | P₂-LES (c1) | (−11,3) BG BZ | ✅ | "Scrutinizing the local flow regimes is significant for comprehending the underlying mechanisms." |
| 372 | P₂-LES (c1) | (−10,3) | ❌ | single-paper hex, 2 hops from BZ — **not usable as BZ evidence** |
