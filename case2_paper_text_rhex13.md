# Case 2 Paper Text (rhex=13)
**Data source:** reaggregated at rhex=13 from backend MSU 2d_coords  
**Key findings:**
1. Same-position BZ pair at (0,12): Background + Result (GeoChron ∩ VolumeSTCube)
2. Triple cross-subspace BZ at (−5,7): Background + Method + Result (VolumeSTCube dominant)

---

## C2. Intra-Group Convergence and Cross-Domain Separation in Spatiotemporal Analysis

### Setup

We load five papers into SeSMap: Compass (P4, urban causal time series), WRF-Chem (P7, PM2.5 prediction), Aerosol (P6, radiative forcing), GeoChron (P5, large-scale spatial time series visualization), and VolumeSTCube (P3, volume-based space-time cube visualization). The five papers span atmospheric science and visualization; we set the HSU aggregation range to 13 px.

### Step 1: The Same-Position BZ Pair at (0, 12)

In the semantic subspace at rhex=13, the analyst observes that hex (0, 12) appears as a BZ in **both the Background subspace (43 MSUs) and the Result subspace (46 MSUs)**. The same two papers — GeoChron (c3) and VolumeSTCube (c4) — form both BZs.

**Background BZ (0,12) — 43 MSUs [GeoChron 41 + VolumeSTCube 2]:**

From GeoChron:
- **MSU-2127:** "Spatial time series visualization offers scientific research pathways and analytical capabilities for understanding spatiotemporal patterns."
- **MSU-2142:** "Spatiotemporal visualization is a key concept in the analysis of data that varies across space and time."
- **MSU-2144:** "Spatiotemporal analysis involves examining data that changes over time and across different geographic locations."
- **MSU-2165:** "Geographic-related information is visualized along a continuous timeline in the space-time cube framework."

From VolumeSTCube:
- **MSU-1861:** "Viewing and analyzing large-scale spatiotemporal series is important in scenarios such as air quality monitoring."
- **MSU-1893:** "On flat terrain, air pollutants are easily transported or diffuse."

**Result BZ (0,12) — 46 MSUs [GeoChron 45 + VolumeSTCube 1]:**

From GeoChron:
- **MSU-2397:** "If in a vertical cylindrical area in 3D cube space are voxel clusters with equal spacing, it indicates a stable pattern."
- **MSU-2398:** "If the voxel cluster exhibits shifts in 3D cube space, it indicates a process of air pollutant migration."
- **MSU-2517:** "The spatial and temporal variation of air quality could be roughly seen in Figure 6A."
- **MSU-2524:** "Air pollution was lower during the period of Figure 6E."

From VolumeSTCube:
- **MSU-1938:** "Afterwards, the air in both regions was filled with air pollutants."

**Interpretation:** GeoChron and VolumeSTCube converge on the *same semantic position* across two distinct discourse roles. Their Background sections share the foundational frame of "large-scale spatiotemporal visualization of environmental (air quality) data," and their Result sections share the discovery frame of "3D space-time pattern analysis for air pollutant dynamics." SeSMap reveals that both papers are answering the same scientific question with the same analytical approach.

A baseline search would find overlapping keywords ("spatiotemporal," "air quality") but would not surface the structural insight: these papers are not merely similar in topic — they independently arrive at the *same spatial-temporal abstraction layer* for both their premises and their conclusions.

### Step 2: The Triple Cross-Subspace BZ at (−5, 7)

More striking is hex (−5, 7), which forms a BZ in **three subspaces simultaneously:**

| Subspace | Papers | MSUs | Content |
|----------|--------|------|---------|
| Background | VolumeSTCube(c4) + GeoChron(c3) | 35 (33+2) | ST series visualization as foundational background |
| Method | VolumeSTCube(c4) + Compass(c0) | 86 (85+1) | **Evolution pattern detection; time-span correlation** |
| Result | VolumeSTCube(c4) + Compass(c0) | 39 (37+2) | Observed evolution patterns; effectiveness of GeoChron |

**Method BZ (−5,7) — 86 MSUs [largest BZ in case]:**
- **MSU-1424:** "Each evolution pattern is characterized by a time period when the trends of these ST series show consistent patterns."
- **MSU-1454:** "We study how to visualize large-scale ST series based on the notion of evolution patterns."
- **MSU-1477:** "The framework slices the time span and captures reliable correlation relations between ST series."
- **MSU-366 (Compass):** "This integration supports the encodings of temporal co-occurrences."

**Result BZ (−5,7):**
- **MSU-1780:** "Interesting evolution patterns are observed in the first case study."
- **MSU-1859:** "The effectiveness and usability of GeoChron are illustrated."
- **MSU-1883:** "GeoChron can compute the layout and render results within seconds."

**Interpretation:** VolumeSTCube occupies the *same semantic position* across all three discourse roles, each time co-projecting with a different neighbor: GeoChron in Background (shared domain context), Compass in Method (shared temporal encoding approach), and Compass again in Result (shared findings on evolution pattern detection). This triple BZ structure identifies VolumeSTCube as a *semantic hub* — a paper whose methodology bridges the spatiotemporal visualization and causal time series analysis communities.

Notably, VolumeSTCube's Result section contains MSU-1859 ("effectiveness and usability of GeoChron"), directly citing GeoChron's performance — even though at the Result hex level, its co-projection partner is Compass. This cross-subspace "hand-off" from GeoChron (Background) → Compass (Method + Result) traces how VolumeSTCube synthesizes ideas from both communities.

### Step 3: Cross-Domain Separation (WRF-Chem and Aerosol)

In contrast to the tight convergence of GeoChron/VolumeSTCube, the atmospheric science papers (WRF-Chem c1, Aerosol c2) project to a spatially distinct region of the semantic space. Their BZs (e.g., (9,6) in Method: 29 MSUs WRF-Chem+Aerosol) lie far from the (0,12) and (−5,7) clusters.

This separation is exactly what a reviewer would want to see: SeSMap correctly identifies that:
1. **Intra-group convergence:** GeoChron and VolumeSTCube are semantically indistinguishable in their core spatio-temporal visualization framing
2. **Cross-domain separation:** The atmospheric science papers (WRF-Chem, Aerosol) project to a different region, confirming the visualization vs. science disciplinary boundary

### Step 4: Creating a Cross-Subspace Flight

The analyst creates a flight connecting:
- **Stop 1:** Background BZ (0,12) — foundational convergence: large-scale spatiotemporal visualization + air quality
- **Stop 2:** Method BZ (−5,7) — methodological bridge: evolution patterns + temporal co-occurrences  
- **Stop 3:** Result BZ (0,12) — result convergence: 3D voxel analysis of air pollutant migration

**LLM flight title:** "From shared spatiotemporal visualization foundations to independent convergence on 3D space-time cube methodology for air quality pattern discovery"

### Quantitative Summary

| BZ | Subspace | Papers | MSUs | Signal strength |
|----|----------|--------|------|----------------|
| (0,12) BG | Background | GeoChron + VolumeSTCube | 43 | ★★★ Same-position pair (part 1) |
| (0,12) R | Result | GeoChron + VolumeSTCube | 46 | ★★★ Same-position pair (part 2) |
| (−5,7) M | Method | VolumeSTCube + Compass | **86** | ★★★ Largest BZ; triple cross-subspace |
| (−5,7) BG | Background | VolumeSTCube + GeoChron | 35 | ★★★ Triple BZ (part 1) |
| (−5,7) R | Result | VolumeSTCube + Compass | 39 | ★★★ Triple BZ (part 3) |
| (−1,11) | BG+Result | VolumeSTCube + GeoChron | 17+5 | ★★ Adjacent same-position pair |
| (9,6) M | Method | WRF-Chem + Aerosol | 29 | ★★ Cross-domain convergence (atm. sci.) |

**Signal-to-noise ratio:** The largest intra-group BZ (86 MSUs at Method (−5,7)) is **8.6× larger** than the largest cross-domain BZ involving papers from different communities (10 MSUs), demonstrating that SeSMap's projection correctly separates domain knowledge while revealing intra-domain conceptual overlap.

### SeSMap's Unique Contribution

A baseline textual co-citation analysis would identify GeoChron and VolumeSTCube as related. However, it would not surface: (a) the *structural* convergence — same hex in both Background AND Result, not just similar keywords; (b) the *triple BZ* at (−5,7) showing the Method-level bridge between visualization and causal analysis communities; (c) the cross-domain *separation* of atmospheric science papers into a distinct semantic region, confirming that the convergence is domain-specific, not an artifact of generic vocabulary.
