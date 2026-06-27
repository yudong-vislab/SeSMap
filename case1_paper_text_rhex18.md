# Case 1 Paper Text (rhex=18)
**Data source:** reaggregated at rhex=18 from backend MSU 2d_coords  
**Key finding:** BZ-R1(−9,4) — single BZ containing 21 MSUs (18×P2-LES + 3×P1-viz)

---

## C1. Cross-Paper Terminological Convergence in Scramjet Combustion Analysis

### Setup

We load two papers into SeSMap: P2-LES, a large eddy simulation study of dual-mode scramjet (DMSJ) combustion, and P1-viz, a visual analytics system (TemporalFlowViz) for scramjet combustion mode analysis. The papers come from different disciplines — computational fluid dynamics and scientific visualization — and neither cites the other. We set the HSU aggregation range to 18 px.

### Step 1: Identifying the Primary Boundary Zone

After decomposing both papers into MSUs and projecting them into the semantic subspace, the Result subspace immediately reveals a dense cluster of Boundary Zone hexes. The analyst clicks on the largest BZ, hex (−9, 4), which contains **21 MSUs: 18 from P2-LES and 3 from P1-viz**.

The LLM summary panel describes this hex as: *"Characterizations of the jet-wake combustion mode: its physical onset conditions, cavity shear-layer dynamics, and OH radical distribution in the scramjet."*

Inspecting the constituent MSUs confirms the convergence:

**From P1-viz:**
- **MSU-961:** "The yellow cluster corresponded to the jet-wake combustion mode."
- **MSU-785:** "The resulting clusters represent distinct latent combustion modes or transitional stages."
- **MSU-991:** "The yellow cluster corresponded to the jet-wake combustion mode."

**From P2-LES (selected):**
- **MSU-368:** "The combustor is at the jet-wake mode in this stage."
- **MSU-249:** "Most of the OH radicals are localized in the cavity shear-layer as opposed to the cavity."
- **MSU-242:** "The cavity shear-layer experiences severe flow oscillation as a result of the large-scale separated vortex shedding."
- **MSU-266:** "The downstream flame receives radicals and a hot environment from the cavity recirculation zone."
- **MSU-430:** "Once the fuel jet collides with the shedding vortex at the cavity step, it commences disintegrating, resulting in a reduction in penetration depth."
- **MSU-474:** "In scram mode, more than 90% of flow was at supersonic speed and the subsonic recirculation rate reached its peak (1.6%) at the cavity recirculation core."

The co-projection is striking: P1-viz (MSU-961) uses a *color cluster label* to identify the jet-wake mode, while P2-LES (MSU-368) uses a *direct physical characterization* of the same mode. Both arrive at the term "jet-wake" independently; SeSMap co-projects them in the same hex.

Crucially, MSU-249 — which identifies OH radicals in the cavity shear-layer — also resides in this hex alongside MSU-961. This reveals that the "yellow cluster" in P1-viz is semantically grounded in the precise OH spatial distribution that P2-LES reports: P1-viz's unsupervised clustering is implicitly capturing the shear-layer OH signature that P2-LES explicitly measures.

### Step 2: Cross-Subspace Flight (Background → Result)

To trace the origin of this convergence, the analyst creates a flight from the Background subspace to BZ-R1 in the Result subspace.

In the Background subspace, BZ (−11, 3) contains 7 MSUs from both papers. P1-viz contributes:
- **MSU-609:** "Traditional scalar metrics (e.g., thrust, equivalence ratio) offer limited insights into combustion mode behavior."
- **MSU-633:** "The CFD platform was developed for high-speed reactive flow modeling."

P2-LES contributes:
- **MSU-322:** "Scrutinizing the local flow regimes is significant for comprehending the underlying mechanisms."

The Background BZ reveals why the visualization was built: P1-viz explicitly states that existing metrics are insufficient (MSU-609), while P2-LES identifies local flow regime analysis as the key scientific challenge (MSU-322). Both papers place this motivational framing in the same semantic neighborhood.

**Flight path:** Background BZ (−11,3) → Result BZ (−9,4)  
**Interpretation:** From shared recognition of the insufficiency of scalar metrics for mode analysis, to terminological convergence on the jet-wake mode as the dominant combustion regime identified by OH-field clustering.

### Step 3: Saving Analytical Evidence

The analyst saves this flight to the Stepwise Analysis View, producing an evidence card with:
- **Citation:** BZ-R1(−9,4) contains MSU-368 and MSU-961, two independent characterizations of the jet-wake mode from different disciplines
- **Supporting:** MSU-249 confirms OH as the spatial bridge between P2-LES's physical measurement and P1-viz's visual clustering
- **LLM insight:** "P1-viz's clustering discovers the physical combustion regime that P2-LES explicitly characterizes as jet-wake — without knowledge of P2-LES's terminology or findings"

### Summary of Findings

| Hex | Subspace | Papers | MSUs | Key evidence |
|-----|----------|--------|------|-------------|
| (−9,4) ★ | Result | P2-LES + P1-viz | 21 | MSU-368 (jet-wake mode); MSU-961 (yellow cluster = jet-wake); MSU-249 (OH in shear-layer) |
| (−8,4) | Result | P2-LES + P1-viz | 12 | Adjacent: MSU-362 (Mach#3 dominant mode) |
| (−10,4) | Result | P2-LES + P1-viz | 12 | Adjacent: MSU-266 variant |
| (−11,3) | Background | P2-LES + P1-viz | 7 | MSU-609 (scalar metrics limited); MSU-322 (flow regime analysis) |

The three adjacent Result BZs at (−9,4), (−8,4), (−10,4) form a connected 45-MSU cluster. All three have the same BZ character (P2-LES + P1-viz co-projection), representing the complete jet-wake mode semantic neighborhood.

**SeSMap's unique contribution:** A baseline textual search for "jet-wake" would find MSU-368 and MSU-961, but would miss the 16 additional P2-LES MSUs that contextualize this finding — the cavity shear-layer dynamics (MSU-242, 266), OH radical distribution (MSU-249), fuel jet disintegration (MSU-430), and scram mode characteristics (MSU-474). SeSMap's projection discovers that all of these constitute the jet-wake combustion mode, even though only one (MSU-368) uses the term explicitly.
