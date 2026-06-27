# Case 2 Evidence Document — REVISED
**Date:** 2026-06-27 (revised from backend API data)  
**Data source:** SeSMap backend `/api/semantic-map?project_id=case2`  
**Papers:**
| API ID | PDF file | Paper title (short) | Domain |
|--------|----------|---------------------|--------|
| paper_id=0, c0 | Compass.pdf | Compass (Causal Analysis of Urban Time Series) | Visualization |
| paper_id=1, c1 | WRF-Chem_...pdf | WRF-Chem PM2.5 DA+BC | Atmospheric Science |
| paper_id=2, c2 | sciadv.adi3568.pdf | Aerosol (Threefold reduction) | Atmospheric Science |
| paper_id=3, c3 | GeoChron.pdf | GeoChron (Visualizing Large-Scale ST series) | Visualization |
| paper_id=4, c4 | VolumeSTCube-TVCG.pdf | VolumeSTCube (Volume-Based Space-Time Cube) | Visualization |

**Note:** GeoChron and VolumeSTCube are companion papers from the **same lead author (Zikun Deng)**. GeoChron is the older paper; VolumeSTCube-TVCG is the newer IEEE TVCG paper extending the method. GeoChron's results section explicitly evaluates VolumeSTCube (MSU-2200).

**Total MSUs:** 2413 | **Total hexes:** 602 | **BZ detection method:** overlapping hex positions (same q,r, different country_id)

---

## BZ Classification Summary

| Subspace | Viz-only BZs | Atmo-only BZs | Cross-domain BZs |
|----------|-------------|---------------|-----------------|
| Background | 21 (max 42 MSUs) | 0 | 7 (max 9 MSUs) |
| Conclusion | 0 | 1 (2 MSUs) | 3 (max 3 MSUs) |
| Experiment | 15 (max 4 MSUs) | 3 (max 19 MSUs) | 10 (max 9 MSUs) |
| Method | 22 (max 72 MSUs) | 5 (max 14 MSUs) | 7 (max 3 MSUs) |
| Result | 7 (max 43 MSUs) | 2 (max 7 MSUs) | 0 |

**Pattern:** Viz-group BZs are much larger (up to 72 MSUs) than cross-domain BZs (max 9 MSUs). Cross-domain BZs in Result subspace = 0 (zero semantic convergence between visualization and atmospheric papers in the Results role).

---

## KEY FINDING: Same-Position BZ Pair in Background ↔ Result

The most striking structural pattern: hex position **(q=0, r=13)** forms a significant BZ in **both** Background and Result subspaces, dominated by the same two papers.

### Background BZ-B1 (q=0, r=13): VolumeSTCube+GeoChron · 42 MSUs
- **VolumeSTCube (2 MSUs):**
  - MSU-1861: "Viewing and analyzing large-scale spatiotemporal series is important in scenarios such as data review and real-time monitoring."
  - MSU-1893: "On flat terrain, air pollutants are easily transported or diffuse."
- **GeoChron (40 MSUs, selective):**
  - MSU-2127: "Spatial time series visualization offers scientific research pathways and analytical decision-making tools across various spatiotemporal domains."
  - MSU-2142: "Spatiotemporal visualization is a key concept in the analysis of data that varies across both space and time."
  - MSU-2165: "Geographic-related information is visualized along a continuous timeline in the spacetime cube."
  - MSU-2177: "Flexible exploration of spatial and temporal domains is necessary."
  - MSU-2185: "This study focuses on ST series of phenomena that are continuous across both space and time."
- **Convergence theme:** Both papers frame large-scale spatiotemporal series visualization as essential infrastructure for environmental/geographic analysis. GeoChron explicitly discusses the space-time cube paradigm in its Background.

### Result BZ-R1 (q=0, r=13): VolumeSTCube+GeoChron · 43 MSUs ★ STRONGEST FINDING
- **VolumeSTCube (1 MSU):**
  - MSU-1938: "Afterwards, the air in both regions was filled with air pollutants." [case study result]
- **GeoChron (42 MSUs, selective):**
  - MSU-2200: "VolumeSTCube achieves better performance than the prior STC-based ST series visualization." ← **EXPLICIT CROSS-PAPER CITATION**
  - MSU-2397: "If in a vertical cylindrical area in 3D cube space are voxel clusters with equal spacing, the air pollution exhibits periodic pattern."
  - MSU-2398: "If the voxel cluster exhibits shifts in 3D cube space, it indicates a process of air pollution propagation."
  - MSU-2517: "The spatial and temporal variation of air quality could be roughly seen in Figure 6A."
  - MSU-2524: "Air pollution was lower during the period of Figure 6E."
  - MSU-2525: "The frequent severe pollution in the west of China became obvious with extensive yellow and red voxels."
  - MSU-2532: "PA clearly observed that starting around March, multiple significant air pollution events occurred in Xinjiang."
  - MSU-2537: "The episodes of moderate or above pollution in the BTH region did not last very long."
- **Convergence theme:** GeoChron's Result section densely describes air pollution pattern analysis using volumetric space-time cube — the exact same method as VolumeSTCube. MSU-2200 explicitly names VolumeSTCube and records its performance advantage over prior STC methods.

**Why same position (0,13)?** Both papers have substantial content about **air quality analysis via volumetric/3D cube visualization**, which embeds into the same semantic region. The identical hex coordinate across Background and Result means SeSMap identifies a coherent semantic thread running through both discourse roles.

---

## Additional Viz-Group BZs

### Background BZ-B2 (q=-5, r=8): ALL 3 VIZ PAPERS · 30 MSUs
- Compass(1) + VolumeSTCube(27) + GeoChron(2) = 30 MSUs
- MSU-121 [Compass]: "Elmqvist and Tsigas studied animation techniques and node representations for the causality of event sequences."
- MSU-1420 [VolumeSTCube]: "ST series visualization is an effective means of understanding the data and reviewing spatiotemporal phenomena."
- MSU-1432 [VolumeSTCube]: "Domains that utilize large-scale spatial time series include geography, atmospheric science, and urban informatics."
- MSU-2285 [GeoChron]: "ST series capture many local but important patterns."
- **Convergence:** Shared survey of spatiotemporal visualization literature. All 3 papers background on the visualization of ST series across geography/atmospheric/urban domains.

### Method BZ-M1 (q=-5, r=8): Compass+VolumeSTCube · 72 MSUs
- Compass(1) + VolumeSTCube(71) = 72 MSUs
- MSU-366 [Compass]: "This integration supports the encodings of temporal co-occurrences."
- VolumeSTCube dominant: evolution pattern detection, Louvain community detection, sliding time windows, two-level visualization
- **Convergence:** Temporal correlation detection. VolumeSTCube uses the Louvain algorithm + sliding window; Compass uses temporal co-occurrence encoding — both systems encode correlations in time as the core method.

### Method BZ-M2 (q=-4, r=7): VolumeSTCube+GeoChron · 54 MSUs
- VolumeSTCube(52) + GeoChron(2) = 54 MSUs
- Theme: spatiotemporal partitioning and layout algorithms for large-scale ST series

### Method BZ-M3 (q=0, r=12): VolumeSTCube+GeoChron · 26 MSUs
- VolumeSTCube(2) + GeoChron(24) = 26 MSUs

---

## Atmospheric Group BZs

### Experiment BZ-E1 (q=11, r=1): WRF-Chem+Aerosol · 19 MSUs
- WRF-Chem(1) + Aerosol(18) = 19 MSUs
- MSU-765 [WRF-Chem]: "Biogenic emissions were calculated using the Model of Emissions of Gases and Aerosols from Nature inventory."
- Aerosol MSUs: POLDER-GRASP validation, AOD/SSA data, AERONET satellite data, Monte Carlo uncertainty analysis
- **Convergence:** Both atmospheric papers use observational aerosol data (POLDER-GRASP, AERONET, OMI satellite) for model validation. WRF-Chem uses MEGAN for emission inventory; Aerosol uses AeroCom model ensemble.

### Method BZ-M4 (q=9, r=7): WRF-Chem+Aerosol · 14 MSUs
- WRF-Chem(13) + Aerosol(1) = 14 MSUs
- MSU-696 [WRF-Chem]: "Data assimilation (DA) can reduce the uncertainty of the chemical initial field and source emission to improve prediction accuracy."
- MSU-753 [WRF-Chem]: "A new scheme that combines DA on the initial conditions and BC simultaneously was developed."
- MSU-1354 [Aerosol]: "We use a Monte Carlo method to estimate the overall uncertainties by repeating the constraining analysis 100,000 times."
- **Convergence:** Uncertainty reduction methods. WRF-Chem uses data assimilation; Aerosol uses Monte Carlo sampling — different methods for the same goal: reducing model prediction uncertainty in atmospheric chemistry.

---

## Cross-Domain BZs (Noise Assessment)

| Subspace | Position | Papers | MSUs | Content |
|----------|----------|--------|------|---------|
| Background | (-1,12) | WRF-Chem+GeoChron | 9 | WRF-Chem pollution processes vs. GeoChron's STC definition |
| Background | (1,13) | WRF-Chem+GeoChron | 8 | WRF-Chem process undulation vs. GeoChron's linked-view taxonomy |
| Experiment | (-4,7) | WRF-Chem+VolumeSTCube | 9 | WRF-Chem Biogenic vs. VolumeSTCube case study |
| Conclusion | (10,6) | Compass+WRF-Chem | 3 | — |
| Method | (2,7) | Aerosol+VolumeSTCube+GeoChron | 3 | — |

**Assessment:** All cross-domain BZs are small (≤9 MSUs). The Background cross-domain BZs between WRF-Chem and GeoChron (9 and 8 MSUs) are the largest and arise because GeoChron's Background section mentions "atmospheric sciences" as an application domain for ST series (MSU-2146: "Spatial time series are prevalent in domains such as environmental science...") — a general claim, not a methodological bridge. **Zero cross-domain BZs in the Result subspace** confirms that the two groups produce fully distinct result content.

---

## Proposed Flights

### Primary Flight: Background BZ-B1 → Result BZ-R1 (SAME POSITION PAIR)
**Path:** Background (q=0,r=13) → Result (q=0,r=13)  
**Papers involved:** VolumeSTCube + GeoChron  
**MSUs per stop:** 42 → 43 MSUs  
**Narrative:** 
- Stop 1 (Background): GeoChron frames the necessity of spatiotemporal series visualization for environmental analysis; VolumeSTCube introduces air pollutant diffusion as the motivating application.
- Stop 2 (Result): GeoChron's result section extensively applies volumetric 3D cube visualization to China's air quality data and explicitly validates that "VolumeSTCube achieves better performance than prior STC-based visualization" (MSU-2200).
- **SeSMap insight:** The flight reveals that two papers from the same research group share not only a problem domain (air quality) but also a semantic anchor position in the map — the same hex coordinate (0,13) appears in both discourse roles, showing the papers' shared content runs coherently from motivation to results.

### Secondary Flight: Background BZ-B2 → Method BZ-M1
**Path:** Background (q=-5,r=8) → Method (q=-5,r=8)  
**Papers involved:** All 3 viz papers → Compass+VolumeSTCube  
**MSUs per stop:** 30 → 72 MSUs  
**Narrative:** Three visualization papers share a background discussion of ST series visualization history; then in the Method subspace at the SAME hex position, VolumeSTCube and Compass converge on temporal correlation encoding methods.

---

## Quantitative Summary

| BZ | Subspace | Position | Papers | MSUs | Category |
|----|----------|----------|--------|------|---------|
| BZ-B1 ★ | Background | (0,13) | VolumeSTCube+GeoChron | 42 | Viz-intra, SAME-POS PAIR |
| BZ-B2 | Background | (-5,8) | Compass+VolumeSTCube+GeoChron | 30 | 3-viz convergence |
| BZ-B3 | Background | (4,-1) | Compass+GeoChron | 21 | Viz-intra |
| BZ-B4 | Background | (5,-1) | Compass+VolumeSTCube | 16 | Viz-intra |
| BZ-B5 | Background | (-1,13) | VolumeSTCube+GeoChron | 16 | Viz-intra |
| BZ-M1 | Method | (-5,8) | Compass+VolumeSTCube | 72 | Viz-intra (temporal) |
| BZ-M2 | Method | (-4,7) | VolumeSTCube+GeoChron | 54 | Viz-intra |
| BZ-M3 | Method | (0,12) | VolumeSTCube+GeoChron | 26 | Viz-intra |
| BZ-M4 | Method | (9,7) | WRF-Chem+Aerosol | 14 | Atmo-intra |
| BZ-M5 | Method | (10,6) | WRF-Chem+Aerosol | 12 | Atmo-intra |
| BZ-E1 | Experiment | (11,1) | WRF-Chem+Aerosol | 19 | Atmo-intra |
| **BZ-R1 ★** | **Result** | **(0,13)** | **VolumeSTCube+GeoChron** | **43** | **Viz-intra, SAME-POS PAIR** |
| BZ-R2 | Result | (-5,7) | Compass+VolumeSTCube | 20 | Viz-intra |
| BZ-R3 | Result | (10,6) | WRF-Chem+Aerosol | 7 | Atmo-intra |

**Cross-domain max:** 9 MSUs. **Viz-intra max:** 72 MSUs. **Signal/noise ratio ≈ 8×.**

---

## Revised Narrative Direction

**Recommended structure for Case 2 in paper:**

1. **(Global layout)** Five papers loaded; the map immediately shows two spatial clusters: 3 visualization papers (left region) and 2 atmospheric papers (right region). Cross-domain BZs are sparse (≤9 MSUs) and zero in the Result subspace — confirming the two domains produce distinct result content.

2. **(Viz-group intra-convergence: BZ-B2 + Method BZs)** Within the visualization cluster, a 3-paper convergence zone in Background (-5,8) shows all three papers survey the same literature (ST series visualization across geography/atmospheric/urban). In the Method subspace, Compass and VolumeSTCube share temporal correlation encoding at the same hex position, revealing a cross-group methodological overlap despite different application domains (causal analysis vs. spatial series).

3. **(KEY FINDING: same-position BZ pair)** The most structurally distinctive pattern: hex (0,13) in Background carries 42 MSUs where GeoChron frames spatiotemporal series visualization for environmental analysis, co-locating with VolumeSTCube's air pollutant context. The SAME hex (0,13) in the Result subspace carries 43 MSUs where GeoChron explicitly evaluates VolumeSTCube performance (MSU-2200) and reports air quality analysis using volumetric 3D cube visualization — the exact same technical approach. A flight connecting these two BZs traces the papers' complete evidence chain from background motivation to result validation.

4. **(Atmo-group: Experiment + Method BZs)** The atmospheric papers converge in Experiment (19 MSUs: shared aerosol observation framework using POLDER-GRASP, AERONET) and Method (14 MSUs: both use numerical uncertainty reduction — data assimilation vs. Monte Carlo).

5. **(What baseline cannot do)** A citation-based approach would detect that GeoChron cites VolumeSTCube once, but would not reveal: (a) 85 MSUs of semantic overlap across three subspaces, (b) the same-position structural pattern, (c) that the convergence is specifically in air quality result content. Topic models would cluster all 3 viz papers together but not identify which discourse role (Result) carries the most distinctive semantic link.
