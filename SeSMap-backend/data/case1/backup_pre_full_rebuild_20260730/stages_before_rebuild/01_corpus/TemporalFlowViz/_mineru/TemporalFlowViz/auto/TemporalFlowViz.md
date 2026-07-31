# TemporalFlowViz: Parameter-Aware Visual Analytics for Interpreting Scramjet Combustion Evolution

Yifei Jia<sup>1,2†</sup>, Shiyu Cheng<sup>1†</sup>, Yu Dong<sup>1</sup>, Guan Li<sup>1,2</sup>, Dong Tian<sup>1,2</sup>, Ruixiao Peng<sup>1,2</sup>, Xuyi Lu<sup>1,2</sup>, Yu Wang<sup>2,3</sup>, Wei Yao<sup>3\*</sup>, Guihua Shan<sup>1,2,4\*</sup>

<sup>1</sup>Computer Network Information Center, Chinese Academy of Sciences, Beijing, China.

<sup>2</sup>University of Chinese Academy of Sciences (UCAS), Beijing, China. <sup>3</sup>Institute of Mechanics, Chinese Academy of Sciences, Beijing, China. <sup>4</sup>Hangzhou Institute for Advanced Study, UCAS, Hangzhou, 310024, China.

\*Corresponding author(s). E-mail(s): weiyao@imech.ac.cn; sgh@cnic.cn; Contributing authors: yfjia@cnic.cn; chengshiyu@cnic.cn; dongyu@cnic.cn; liguan@cnic.cn; tiandong@cnic.cn; rxpeng@cnic.cn; luxuyi@cnic.cn; yuwang@imech.ac.cn; <sup>†</sup>These authors contributed equally to this work.

## Abstract

Understanding the complex combustion dynamics within scramjet engines is critical for advancing high-speed propulsion technologies. However, the large scale and high dimensionality of simulation-generated temporal flow field data present significant challenges for visual interpretation, feature diferentiation, and crosscase comparison. In this paper, we present TemporalFlowViz, a parameter-aware visual analytics workflow and system designed to support expert-driven clustering, visualization, and interpretation of temporal flow fields from scramjet combustion simulations. Our approach leverages hundreds of simulated combustion cases with varying initial conditions, each producing time-sequenced flow field images. We use pretrained Vision Transformers to extract high-dimensional embeddings from these frames, apply dimensionality reduction and density-based clustering to uncover latent combustion modes, and construct temporal trajectories in the embedding space to track the evolution of each simulation over time.

To bridge the gap between latent representations and expert reasoning, domain specialists annotate representative cluster centroids with descriptive labels. These annotations are used as contextual prompts for a vision-language model, which generates natural-language summaries for individual frames and full simulation cases. The system also supports parameter-based filtering, similarity-based case retrieval, and coordinated multi-view exploration to facilitate in-depth analysis. We demonstrate the efectiveness of TemporalFlowViz through two expertinformed case studies and expert feedback, showing TemporalFlowViz enhances hypothesis generation, supports interpretable pattern discovery, and enhances knowledge discovery in large-scale scramjet combustion analysis.

Keywords: Visual Analytics, Temporal Flow Field, Scientific Data Simulation, Vision Transformer.

## 1 Introduction

Supersonic combustion in scramjet engines plays a pivotal role in enabling sustained hypersonic flight. Computational simulations have become indispensable for analyzing the complex interactions among turbulence, chemical kinetics, and thermal efects in these high-speed propulsion systems (Curran et al. 2003). However, the resulting data—typically high-resolution, time-sequenced flow field images across diverse initial conditions—are massive in scale and high in dimensionality, posing significant challenges for downstream analysis and expert interpretation (Ladeinde 2009).

Traditionally, combustion performance is assessed using scalar metrics such as ignition delay, pressure rise, or thrust eficiency. While useful for summarizing global trends, these metrics ofer limited insight into the spatial and temporal structures underlying reactive flows. As domain experts increasingly seek to understand how initial conditions afect combustion dynamics, there is a growing demand for tools that support exploratory, interpretable, and human-in-the-loop analysis of large-scale simulation data.

Recent advances in deep learning ofer promising solutions. Vision Transformers (ViTs), in particular, have demonstrated strong capabilities in extracting semantically rich features from images by modeling them as sequences of patches (Dosovitskiy et al. 2021). Their application to fluid dynamics—such as PDE operator learning (Wang et al. 2025) and particle image velocimetry (Reddy et al. 2025)—shows promise for capturing structural complexity in scientific visualizations.

To aid expert interpretation, dimensionality reduction techniques like t-SNE and UMAP (van der Maaten and Hinton 2008; McInnes et al. 2020), combined with density-based clustering methods such as DBSCAN (Ester et al. 1996), have been widely used to expose latent patterns, combustion modes, and transitional behaviors in temporal flow field data (Bae et al. 2020). Visual analytics systems further enhance interpretability by linking latent embeddings with raw simulation frames (Wu et al. 2023), and by enabling multi-view exploration through cluster-based interaction (Yu et al. 2023; Chen et al. 2024), reasoning (Peng et al. 2025), and trajectory-based interpretation (Bostock et al. 2011; Eirich et al. 2021; Chen et al. 2024).

Despite these advances, key challenges remain. Expert interpretation of flow behavior often relies on implicit knowledge, such as the position of flame fronts, shock structures, or pressure gradients—elements that are not automatically encoded or explained by conventional embedding pipelines. While recent developments in vision–language models (VLMs) ofer mechanisms for converting visual input into natural language, their application to scientific visualization is still limited, particularly in domains where domain-specific terminology and interpretive precision are essential.

In this paper, we introduce TemporalFlowViz, a parameter-aware visual analytics workflow and system designed to support expert-driven interpretation of scramjet combustion simulations. Our workflow extracts high-dimensional embeddings from flow field images using pretrained ViTs, applies dimensionality reduction and clustering to identify latent combustion modes, constructs temporal trajectories to trace flow evolution under varying initial conditions, and integrates expert annotations into a VLM-based summarization pipeline to generate descriptive, domain-aligned reports. An interactive multi-view interface connects these components through coordinated filtering, trajectory exploration, case retrieval, and frame-level semantic inspection.

We summarize our contributions as follows:

• We propose a unified visual analytics workflow for analyzing temporal flow field data from scramjet combustion simulations. This workflow integrates pretrained Vision Transformer–based embedding, dimensionality reduction and clustering to uncover combustion mode patterns, temporal trajectory modeling to capture flow evolution, and vision–language summarization grounded in expert annotation.

• We implement the workflow into an interactive visual analytics system that supports parameter filtering, embedding-space exploration, case similarity retrieval, and coordinated multi-view interpretation of combustion behavior. The system enables experts to compare combustion evolution across cases and interpret dynamic flow structures via both visual and semantic cues.

• We conduct two expert-informed case studies and synthesize qualitative feedback to demonstrate the system’s efectiveness. Results show that TemporalFlowViz facilitates hypothesis generation, reveals parameter-sensitive combustion behavior, and enhances the interpretability and communicability of large-scale temporal simulation data.

## 2 Related Work

## 2.1 Visualization of Combustion Flow Fields

Visualization techniques for flow fields have evolved to address the challenges of interpreting complex, high-dimensional fluid dynamics data. Domain-specific applications such as FlowVisual have tailored flow visualization for educational contexts, integrating 3D volume rendering and glyph techniques to teach fluid concepts (Wang et al. 2016). Znamenskaya’s panoramic visualization survey reviews methods for thermophysical flow fields, including refractive index methods and digital particle image velocimetry, emphasizing the need for scalable, quantitative analysis tools (Znamenskaya 2021). (Zhao et al. 2025) conduct an empirical experiment to evaluate the time costs and frame rates recorded of popular web-based libraries on visualizing hundreds of graph datasets. He et al. (2023) present a parallel visualization framework for large-scale flow-field data, combining grid-segmented data partitioning with optimized communication strategies to achieve real-time interactivity on millions of grid cells. (Chen et al. 2024) introduce an open dataset of DLGs (data lineage graph), including the DLG model, the dataset construction process, and applied areas. Complementary approaches, such as saliency-guided importance sampling (Bae et al. 2020), have been proposed to focus visualization efort on regions of greatest dynamical interest. Li et al. (2025) propose the ParamsDrag model to facilitate parameter space exploration through directly dragging structure-related features in the visualizations.

## 2.2 Image Embedding and Latent Pattern Clustering

Image Embeddings. Efective image representation is fundamental to visual analytics systems, enabling subsequent tasks such as clustering and semantic interpretation. Early handcrafted methods, such as SIFT, extract local keypoint descriptors to capture distinctive image patches invariant to scale and rotation (Lowe 2004). Histogram of Oriented Gradients similarly encodes gradient orientations within localized cells, providing robust structural features for object recognition (Dalal and Triggs 2005). The advent of deep convolutional neural networks marked a paradigm shift: AlexNet demonstrates that end-to-end learning of hierarchical features yields superior performance on large-scale image classification (Krizhevsky et al. 2012), while ResNet introduces residual connections to facilitate training of very deep architectures (He et al. 2016). Unsupervised methods, such as stacked denoising autoencoders, learn compact representations by reconstructing corrupted inputs and have been applied successfully to clustering and anomaly detection (Vincent et al. 2010). More recently, Vision Transformers model images as sequences of patch embeddings, leveraging selfattention to capture long-range dependencies and achieve state-of-the-art results on various benchmarks (Dosovitskiy et al. 2021). These advances form the foundation for representing complex flow-field imagery in scramjet combustion analysis.

Embedding Clustering Methods. Many studies have been devoted to data clustering methods, and existing methods can be roughly divided into three categories: distance-based methods, density-based methods, and connectivity-based methods. Distance-based methods, such as K-means (Wang et al. 2012, 2015) and agglomerative clustering (Chidananda Gowda and Krishna 1978), attempt to find relationships between data points based on various distance metrics. Density-based methods attempt to cluster data points by appropriate density functions, including densitybased spatial clustering with noise (Wang et al. 2015). Compared with the previous methods, connectivity-based methods cluster data points into a cluster if they are highly connected. A commonly used method is spectral clustering (Zelnik-Manor and Perona 2004). The above ideas form the basis of many methods, such as ensemble clustering (Huang et al. 2016), clustering based on non-negative matrix factorization (Cai et al. 2009), etc.

## 2.3 Report Generation on Scientific Data

Zhang et al. (2025) introduce an LLM-driven framework for circular genome visualization generation, harnessing LLM agents to automate not only code generation but also text summarization for subsequent analysis. Li et al. (2019) propose a novel approach for generating medical image reports by integrating knowledge-driven encoding, retrieval, and paraphrasing mechanisms. It decomposes medical report generation into three main steps: encoding visual features into a structured abnormality graph, retrieving relevant text templates based on detected abnormalities, and paraphrasing these templates to generate the final report. Wang et al. (2023) introduce the Chat-CAD to integrate LLMs with medical-image CAD networks. The proposed approach uses LLMs to enhance the outputs of multiple CAD networks—including diagnosis, lesion segmentation, and report generation networks—by summarizing and reorganizing information into natural language text. R2GenGPT leverages the pre-trained knowledge of LLMs while overcoming their inability to process visual data directly (Wang et al. 2023). By aligning image features with the LLM’s text embedding space, the model enables frozen LLMs to generate radiology reports that integrate both visual insights and clinical language, without fine-tuning the whole LLM.

## 3 Analysis Requirements from Domain Collaboration

Understanding the dynamic behavior of scramjet combustion processes involves interpreting complex spatial and temporal patterns across numerous simulation runs. To ensure our research addresses real-world analytical challenges, we conducted structured interviews and co-design sessions with two domain experts, E1 and E2. E1 is a researcher specializing in computational fluid dynamics and scramjet combustion modeling, with expertise in reactive flow structures and combustion mode transitions, while E2 is an aerospace engineer engaged in propulsion system development, focusing on simulation-based evaluation of engine configurations and their sensitivity to input parameters of initial conditions.

These two experts represent complementary perspectives from academic and engineering domains. We have collaborated with them over the past two years during a collaborative project involving scramjet combustion simulation analysis. Based on recurring discussions and workflow observations, we distilled a set of representative domain challenges that guided the design of our visual analytics workflow and system.

## 3.1 Domain Challenges

C1: High-dimensional temporal data overload. Each simulation of combustion produces dozens of time-sequenced flow field frames. Traditional scalar metrics (e.g., thrust, equivalence ratio) ofer limited insights into the spatial and temporal transitions during combustion. Experts expressed the necessity to visually cluster and summarize combustion modes across time.

C2: Dificulty in interpreting combustion evolution. Experts have to frequently assess whether combustion reaches steady state, exhibits oscillatory or transitional behaviors, or fails to ignite. However, manual inspection is labor-intensive and susceptible to subjectivity.

C3: Limited semantic support for knowledge communication. The interpretation of combustion modes often relies on tacit expert knowledge. There is a lack of structured mechanisms to capture and share this expertise across teams or with automated systems.

C4: Complexity in parameter-driven case comparison. Design exploration requires understanding how changes in initial condition parameters afect combustion outcomes. Experts need support for identifying and comparing cases with similar behaviors under diferent initial conditions.

## 3.2 Analytical Requirements

Based on the above challenges, we derived the following system-level analytical requirements that guided the design of TemporalFlowViz:

R1: Clustering and summarizing temporal flow field data to uncover latent combustion modes and reveal evolution patterns across time.

R2: Trajectory-based visualization and comparison to support stable combustion detection and interpretation of transitional combustion behaviors.

R3: Semantic annotation and report generation to bridge expert interpretations with natural language outputs for communication and automation.

R4: Parameter-aware case retrieval and analysis to enable comparative exploration across simulations with diferent initial conditions.

These domain-informed requirements serve as the foundation for the system pipeline and interaction design described in the following sections.

## 4 Data Background and Simulation Parameters

To support the parameter-aware analysis tasks described in the previous section, we constructed a dataset of scramjet combustion simulations in collaboration with domain experts. The dataset was generated using a specialized CFD (Computational Fluid Dynamics) platform developed for high-speed reactive flow modeling.

The simulation campaign covers over 200 individual scramjet combustion cases, each conducted under a distinct combination of initial conditions. Each simulation case outputs a time series of flow field data recorded at regular time intervals, capturing the evolution of combustion dynamics from ignition to either a stable combustion state or an instability-induced shutdown. The total dataset includes approximately 3 terabytes of flow field data. Each case is defined by a unique combination of the following three initial condition parameters:

Static pressure (P) — ranging from 0.8 MPa to 2.1 MPa, controlling upstream compression and shock strength;

Static temperature (T) — ranging from 565 K to 830 K, influencing chemical reactivity and ignition delay;

Water vapor content $\mathbf { ( H _ { 2 } O ) }$ — ranging from 7.8% to 14%, afecting the mixture equivalence ratio and combustion eficiency.

At each timestep, the CFD solver outputs multiple physical fields including static pressure, OH mass fraction, velocity, Mach number, and others. Among these, the pressure field and OH field were identified by domain experts as particularly important for combustion interpretation. The pressure field reflects global flow evolution and the onset of steady or oscillatory combustion states, while the OH field indicates local combustion activity and flame structure. Each physical field’s VTK data is then processed by a python program to map into an image under a uniform colormap and cropped out of excessive background regions to generate a clean visual representation of flow field inside engine.

This dataset serves as the empirical foundation for our visual analytics workflow and enables the exploration of parameter-dependent combustion phenomena across a wide range of initial conditions.

## 5 TemporalFlowViz Overview

To meet the analytical requirements identified in collaboration with domain experts, we present TemporalFlowViz, an end-to-end visual analytics workflow and system designed to support the parameter-aware interpretation of combustion evolution in scramjet simulations. TemporalFlowViz addresses the four core requirements—temporal summarization (R1), trajectory-based pattern recognition (R2), semantic communication (R3), and parameter-aware case comparison (R4)—via deep feature embedding, temporal trajectory modeling, and expert-guided semantic summarization, all coordinated through a unified visual analytics system.

The TemporalFlowViz consists of three main stages, each corresponding to one or more analytical requirements:

Flow Field Data Preprocessing and Embedding (supports R1, R4). Simulation outputs are organized into time-sequenced flow field frames. These frames are encoded into high-dimensional embeddings and filtered based on user-specified initial conditions (e.g., static pressure, temperature, humidity), enabling comparative analysis across selected cases.

Temporal Exploration (supports R1, R2). Frame-level embeddings are reduced and clustered to reveal latent combustion states. Temporal trajectories are constructed for each simulation by connecting the embeddings of frames chronologically in Temporal Trajectory View, allowing experts to assess mode transitions, steady-state convergence, or anomalies.

Semantic Summarization and Report Generation (supports R3). Expertprovided annotations on centroids of representative clusters are used to condition a vision-language model, which generates descriptive summaries of individual frames and entire cases. These outputs help contextualize visual observations and facilitate communication.

![](images/91bfd420c7f54805120ba9f648305a6b60818b51277892a0e216d5faeee167ab.jpg)  
Fig. 1 Overview of TemporalFlowViz, including preprocessing and embedding of flow fields, temporal trajectory-based exploration, and semantic summarization and report generation via expertinformed vision-language modeling.

## 6 Workflow Design and Experiment

We propose a five-stage backend workflow to support the expert-driven visual analytics goals of TemporalFlowViz. The pipeline includes spatially focused preprocessing, feature embedding, dimensionality reduction, temporal trajectory construction, and cluster-based semantic summarization. These components are tightly integrated to translate complex simulation outputs into interpretable latent structures and textual reports.

## 6.1 Input Preprocessing and Region Cropping

TemporalFlowViz begins with preprocessing simulation outputs into multichannel flow field frames, represented such as $X _ { i } ( t ) = [ P _ { i } ( t ) , O H _ { i } ( t ) , . . . ]$ , where $P _ { i } ( t )$ and $O H _ { i } ( t )$ denote the pressure and hydroxyl fields at time step t for simulation case $i .$

In iterative collaboration, both E1 and E2 emphasized that key combustion phenomena—such as ignition onset, shock–flame interaction, and stabilization—primarily manifest near the isolator and cavity zones of the scramjet geometry. Based on this insight, we explored whether focusing on these regions could improve embedding coherence and downstream interpretability.

To test the hypothesis, we compared two preprocessing strategies: using the whole flow field frames versus applying a spatial crop around the isolator and cavity region. We kept the size and location of crop box fixed across all frames of a flow field channel (e.g. pressure field) to ensure comparability. Both inputs were passed through the same embedding extraction pipeline and then projected into the UMAP-reduced latent space.

As shown in Figure 2, cropped inputs led to tighter clustering and more convergent temporal trajectories, whereas whole-frame inputs produced more scattered and noisy distributions. In our experiment, the average convergence radius r (the average radius of the last K vertices of all the case trajectories, $K = 5$ in our experiment) for PCA reduced embeddings of cropped flow fields $( p _ { i } ( t )  P C A  q _ { i } ( t ) )$ , was reduced by 35.24 ± 0.59%, compared to those of un-cropped flow fields. The convergence radius $r _ { i }$ of a trajectory of case i is defined as: $\begin{array} { r } { r _ { i } = \frac { 1 } { K } \sum _ { t = t _ { i } - K + 1 } ^ { t _ { i } } | | q _ { i } ( t ) - \overline { { q _ { i } } } | | } \end{array}$ where $\begin{array} { r } { \overline { { q _ { i } } } = \frac { 1 } { K } \sum _ { t = t _ { i } - K + 1 } ^ { t _ { i } } q _ { i } ( t ) } \end{array}$ . These results confirm that spatially focusing on expertidentified regions enhances latent structure clarity and combustion mode separability. Consequently, we adopt cropped inputs centered on the isolator–cavity region as the default configuration throughout our pipeline.

## 6.2 Frame Embedding with Vision Transformers

After preprocessing, each cropped flow field frame $P _ { i } ^ { \prime } ( t )$ is converted into a highdimensional embedding vector $p _ { i } ( t )$ using a pretrained Vision Transformer (ViT). We define the embedding process as:

$$
p _ { i } ( t ) = \mathrm { V i T } ( P _ { i } ^ { \prime } ( t ) , W _ { P } )\tag{1}
$$

where $W _ { P }$ is an optional spatial weighting matrix and ViT refers to the selected encoder backbone.

To evaluate how diferent ViT architectures capture combustion-related spatial features, we tested three pretrained models: BLIP2-opt-2.7b (Li et al. 2023), InternViT-6B-v2.5 (Chen et al. 2024), and AIMv2-1B (Fini et al. 2024). For each model, we extracted frame-level embeddings from the same cropped input images and visualized their latent distributions using dimensionality reduction method.

These comparisons revealed that although all models reflect temporal evolution to varying degrees, the latent structures they produce difer significantly in terms of trajectory continuity, cluster separability, and interpretability: BLIP-2 tended to generate smooth, temporally ordered embeddings, with many combustion sequences forming U-shaped manifolds indicative of progressive evolution; InternViT produced multiple fragmented clusters with less clear chronological transitions; AIMv2 resulted in noisier patterns with reduced cluster coherence.

![](images/8fd4442e886fef94ad5a173dcaa50bcbaf63354d7e03e2ce0da28d05d577b75a.jpg)  
Fig. 2 Comparison of UMAP-projected trajectories using pressure field embeddings from whole frames (top 2 graphs) versus cropped regions (bottom 2 graphs). Cropped inputs yield more compact clusters and clearer trajectory convergence.

Rather than selecting a single default model, we retain all three ViT backbones as viable options within our system. This allows researchers to experiment with diferent embedding strategies depending on the analysis task. For example, using BLIP-2 for trajectory-based reasoning, or InternViT to highlight mode transitions via clustering boundaries.

By decoupling embedding from downstream components, TemporalFlowViz supports flexible plug-and-play use of visual encoders while maintaining a consistent analysis pipeline.

## 6.3 Dimensionality Reduction for Reasoning

To interpret and compare the high-dimensional embeddings produced by diferent ViT models, we applied two commonly used nonlinear dimensionality reduction techniques—t-SNE and UMAP—to project frame-level embeddings $p _ { i } ( t ) \in \mathbb { R } ^ { n }$ into a shared two-dimensional latent space $p _ { i } ^ { \prime } ( t ) \in \mathbb { R } ^ { 2 }$ for visualization and clustering.

This projection allows experts to visually assess latent structure, combustion mode separation, and temporal trajectory continuity. As shown in Figure 3, the top two plots illustrate the projection results using t-SNE for both pressure and OH field embeddings, while the bottom two plots show the corresponding UMAP projections.

While t-SNE tends to preserve local neighborhood relationships, it often distorts global layout and trajectory geometry, making it less efective for temporal reasoning. In contrast, UMAP maintains both local and global structural continuity, which better preserves combustion trajectory shapes, cluster boundaries, and transition trends.

Based on visual evaluation and expert feedback, we selected UMAP as the default projection method for OH fields. UMAP not only enables more faithful temporal trajectory construction but also produces cleaner latent clusters that are easier to annotate and interpret.

To ensure comparability across experiments, we fixed the UMAP parameters to n neighbors = 15 and min dist = 0.1, following recommended values from prior work and empirical tuning. This consistent configuration ensures that embeddings from diferent backbone models can be analyzed under the same projection basis.

## 6.4 Trajectory Modeling and Similarity Matching

To capture the dynamic evolution of combustion behavior over time, we construct latent-space trajectories for each simulation case. Given a case i with $t _ { i }$ time steps, we denote the sequence of UMAP-projected frame embeddings as: $p _ { i } ^ { \prime } ( 1 ) , p _ { i } ^ { \prime } ( 2 ) , . . . , p _ { i } ^ { \prime } ( t _ { i } )$ These points are chronologically connected to form a trajectory in the 2D latent space, visually representing the temporal evolution of flow characteristics. Trajectories that gradually converge or oscillate confinedly often indicate stable combustion modes, while diverging ones may suggest unstabilized combustion modes.

To enable comparative reasoning and pattern reuse, we identify the top-k most similar trajectories to any selected case i. Let the vertices of trajectory of case i be $p _ { i } ^ { \prime } ( 1 ) , \ldots , p _ { i } ^ { \prime } ( t _ { i } )$ , and let candidate cases be indexed by $j _ { 1 } , j _ { 2 } , \ldots , j _ { k }$ . The most similar k trajectories retrieved are denoted as $p _ { j _ { 1 } } ^ { \prime } , p _ { j _ { 2 } } ^ { \prime } , \ldots , p _ { j _ { k } } ^ { \prime }$

To quantify trajectory similarity, we have designed and iteratively refined a temporal dissimilarity metric in collaboration with domain experts. Experts evaluated various formulations based on their ability to capture perceptual and physically meaningful similarities—particularly in late-stage behavior related to convergence and stabilization.

![](images/5b66560c7c97ff1f4e70bafb6ce12d6daa78ed5ea7bf0d547518fc3f536c895d.jpg)  
Fig. 3 UMAP versus t-SNE: The top two scatter plots display the dimensionality reduction results of the pressure and OH fields’ embeddings obtained through t-SNE, while the bottom ones present the corresponding results using UMAP. UMAP better preserves trajectory shapes and cluster separations at OH fields compared to t-SNE.

The diference between trajectories $p _ { i } ( \cdot )$ and $p _ { j } ( \cdot )$ , revised through multiple rounds of expert review, is defined using a time-warped alignment:

$$
\Delta ( i , j ) = \sum _ { ( a , b ) \in \mathrm { D T W } ( t _ { i } , t _ { j } ) } \frac { \Vert p _ { i } ( a ) - p _ { j } ( b ) \Vert } { \Vert p _ { i } ( a ) - p _ { i } ( a - 1 ) \Vert + \Vert p _ { j } ( b ) - p _ { j } ( b - 1 ) \Vert }\tag{2}
$$

Here, $\mathrm { D T W } \left( t _ { i } , t _ { j } \right)$ denotes the optimal alignment path between the two trajectories using Dynamic Time Warping, allowing for flexible temporal matching. $t _ { i }$ and $t _ { j }$ are the number of time steps in cases i and $j ,$ respectively. This metric maintains normalization by local motion magnitudes while allowing for elastic alignment, thereby addressing minor temporal shifts that do not afect the overall behavior of temporal evolution.

The computed similarity scores serve two purposes. First, they are used to retrieve the top-k most similar cases, which are visualized in the Similar Trajectories View for comparative inspection (see Figure $4 ( \mathrm { c } ) )$ . Second, they help build a retrieval basis for downstream interpretation, allowing experts to generalize observations from one case to others exhibiting similar dynamics.

## 6.5 Semantic Summarization and Report Generation

To support interpretable summarization of latent flow field structures, we integrate clustering, expert annotation, and vision–language modeling into a unified semantic pipeline.

We begin by clustering all UMAP-projected frame-level embeddings $p _ { i } ( t )$ using DBSCAN. The resulting clusters represent distinct latent combustion modes or transitional stages. For each cluster $C _ { k } .$ , we identify a representative centroid by selecting the embedding closest to the mean of cluster:

$$
\hat { p } _ { k } = \arg \operatorname* { m i n } _ { p _ { i } ( t ) \in C _ { k } } \left\| p _ { i } ( t ) - \frac { 1 } { | C _ { k } | } \sum _ { p _ { j } ( t ^ { \prime } ) \in C _ { k } } p _ { j } ( t ^ { \prime } ) \right\|\tag{3}
$$

These centroids form the basis for expert annotation. During workflow execution, domain experts interactively assign combustion-specific terminology to selected centroids through an annotation interface. The resulting annotated exemplars serve as semantic anchors for guiding the generation process.

We compared two open-source VLMs for this task: InternVL2.5-38B and Gemma-3-27B (Team et al. 2025). While InternVL2.5 achieves strong performance on visionlanguage benchmarks, its deployment presented practical limitations, including high VRAM usage, limited quantization support, and a lack of pre-optimized deployment formats. In contrast, Gemma-3 ofers lightweight 4-bit and 8-bit quantizations and eficient image-to-text generation speed. It successfully processes multiple images in a single pass without out-of-memory failures, making it more suitable for our batchoriented visual analytics workflow. As a result, Gemma-3 is adopted in our workflow deployment.

When selecting a flow field frame for interpretation, the workflow retrieves the top-k nearest cluster centroids in latent space and gathers their associated images and expert-provided annotations. These are used as context inputs to Gemma-3 which generates descriptive summaries of the selected frame. This mechanism supports multiple levels of semantic abstraction. At the frame level, the workflow produces direct semantic interpretations based on nearby cluster exemplars. For entire simulation cases, Gemma-3 aggregates information along latent trajectories to generate case-level overviews. Additionally, when cluster transitions occur—such as during ignition or mode shifts—focused summaries are produced to explain key transitional events like flame stabilization or failure recovery.

## 7 Visual Analytics System Design

The visual analytics system interface, as illustrated in Figure 4, is composed of five tightly coordinated views that correspond to the main stages of the TemporalFlowViz workflow: data filtering, embedding projection, trajectory construction, similarity retrieval, and semantic interpretation. These views support multilevel visual reasoning, enabling experts to move fluidly from parameter-level filtering to latent structure inspection, raw field validation, and case-level reporting.

![](images/0127f3de12a9b4d1266859d8548f89703fdd1f00ab6f5326a31002421ea33a67.jpg)  
Fig. 4 The visual analytics system interface of TemporalFlowViz: (a) the Filtering Panel presents initial parameters filters, case selection table and clustering parameters. (b) the Temporal Trajectory View visualizes the clustering result and temporal trajectory of the selected case in latent space. (c) the Similar Trajectories View draws the top three most similar temporal trajectories of three simulation cases. (d) the Report View shows the AI generated or expert annotated description of selected frame in (b). (e) the Details View displays all the flow field frames of the same case of selected point in (b).

## 7.1 Filtering Panel

Filtering Panel (Figure 4(a1)) enables users to subset and organize the simulation data based on initial conditions and clustering parameters. The Initial Parameter Filters section provides interactive sliders and text boxes to constrain the simulation dataset by static pressure (P, in MPa), static temperature (T, in Kelvin), and water vapor concentration $( H _ { 2 } O ,$ , in %). These filters allow users to isolate specific combustion modes of interest, such as high-pressure lean combustion, and reduce visual clutter in downstream views.

Below the filters, the Selected Cases Table (Figure 4(a2)) presents a tabular overview of the filtered simulation cases, showing their initial conditions (parameters) for quick comparison. Users can select multiple cases using checkboxes and initiate visualization by clicking the “Draw Scatter Plot” button. Sorting by column headers allows experts to prioritize cases for analysis.

The Clustering Parameters subsection (Figure 4(a3)) allows customization of the DBSCAN clustering algorithm, including the minSamples and eps parameters. Users can also select the flow field component (e.g., pressure, OH concentration, Mach) used to generate feature embeddings. This enables exploration of how diferent physical variables influence the latent structure and cluster formation.

## 7.2 Temporal Trajectory View

Temporal Trajectory View (Figure 4(b)) visualizes high-dimensional frame embeddings projected into a two-dimensional space via dimensionality reduction (e.g., UMAP). Each point represents a frame from a combustion simulation, colored by its cluster assignment. When a case is selected, its frames are chronologically connected to form a temporal trajectory, revealing the evolution of combustion dynamics in latent space.

This view supports analysis at multiple levels. Globally, experts can identify clusters corresponding to diferent combustion modes. Locally, the trajectory of a single case reveals whether the process remains within a stable mode, transitions between modes, or shows oscillatory behavior. Users can zoom, pan, and hover for tooltips. Clicking a point highlights the corresponding frame in the Details View and triggers semantic generation in the Report View. Cluster centroids can be toggled using the “Hide Centroids / Show Centroids” button.

## 7.3 Similar Trajectories View

Similar Trajectories View (Figure 4(c)) supports comparative reasoning by displaying the top six similar temporal trajectories to the currently selected case. Similarity is computed using a domain-informed diference metric that emphasizes relative deviation near the final time steps of each trajectory, as defined in previous section. This design prioritizes the convergence and stability characteristics that are critical for interpreting combustion behavior.

Each retrieved trajectory is visualized as a mini scatter plot, where the target trajectory is highlighted against a dimmed background of all other points. This layout enables quick visual comparison of trajectory shapes, cluster transitions, and stabilization patterns across cases. Selecting a trajectory in this view updates the main Temporal Trajectory View and allows direct navigation to the corresponding frames in the Details View and Report View, supporting seamless multi-view inspection and cross-case pattern recognition.

## 7.4 Details View

Details View (Figure 4(e)) provides a scrollable view showing the full temporal sequence of flow field frames of scientific visualizations for the selected case. Each row corresponds to a simulation timestep, presenting the raw field output (e.g., pressure, OH). Users can click a frame to highlight its embedding in the scatter plot and view its description in the Report View.

Details view allows experts to validate embedding-based patterns against flow fields, identify key events such as ignition or blowout, and visually confirm the structure of latent trajectories. Hover previews and direct selection support eficient inspection of important frames.

## 7.5 Report View

Report View (Figure 4(d)) displays the description of the currently selected frame in the scatter plot. Descriptions may be VLM-generated or manually annotated.

When generating new frame descriptions, the system retrieves the top-k nearest cluster centroids in latent space and uses their annotated descriptions as context for a vision-language model, which produces a frame-specific description. The images and descriptions of frames from the selected case are used by VLM to generate case summary.

This mechanism links low-level visual data to expert interpretation. By grounding the model in expert-labeled centroids, the generated summaries align with combustionspecific terminology and provide interpretable insights into dynamic phenomena. The Report View supports quick review, edit, and export for documentation or cross-expert discussion.

Together in this interface, these five views form an integrated environment that operationalizes the entire workflow. Through coordinated filtering, embedding visualization, trajectory comparison, raw flow inspection, and semantic interpretation, the system enables experts to explore, interpret, and report on complex combustion simulations with high precision and interactivity.

## 8 Case Studies

To validate the efectiveness of TemporalFlowViz in supporting real-world analytical workflows, we conducted two case studies involving independent exploration. These cases were designed to reflect practical use scenarios and to evaluate how well the system supports expert reasoning, pattern recognition, and semantic interpretation in combustion simulation analysis.

To ensure objective evaluation of the final implementation, two additional experts, E3 and E4, were invited to conduct independent analyses using the deployed system.

Case Study 1 focused on pressure field evolution and combustion mode interpretation and was performed by E3, a postdoctoral researcher specializing in reactive flow diagnostics. Case Study 2 explored the classification of combustion modes based on OH field trajectories and was conducted by E4, a senior simulation engineer engaged in engine configuration evaluation.

## 8.1 Case Study 1: Pressure Field Evolution and Combustion Mode Interpretation

In this case study, we analyzed how pressure field evolution and latent-space trajectories reflect combustion mode transitions, and how the system facilitated interpretation through trajectory-based comparison and automatic report generation.

We first cropped the pressure field images to isolate the core region of the scramjet engine—focusing on the isolator and cavity—and assigned larger weights to subregions with more pronounced combustion activity. The resulting weighted images were encoded into high-dimensional embeddings using Blip2-opt-2.7b, InternViT-6B-v2.5, and AIMv2-1B, and processed through TemporalFlowViz. In all three kinds of ViTbased embeddings tested, the Temporal Trajectory View revealed a common structure: the temporal trajectories began in a dispersed region and gradually converged into a compact cluster. To better display the clustering results, we used InternViT to extract embeddings from frames.

We filtered simulation cases using the Filtering Panel to constrain initial conditions such as pressure, temperature, and humidity. For three selected cases (a1–c1), we observed the spatiotemporal pressure evolution in Figure 5(a2–c2). The blue-red interface indicated a sharp pressure surge. In these cases, the surges appeared well before the cavity—indicative of ramjet-like behavior. As the simulation progressed, the surges either stabilized or oscillated within a narrow region. After 6 ms, the pressure fields remained largely stable in a2, b2, and c2, consistent with steady-state combustion. The first two frames in each series (a2–c2) showed surge positions between the isolator and cavity, a signature of early-stage scramjet ignition.

By comparing the Temporal Trajectory View and the Similar Trajectories View, we confirmed that cases with similar trajectory shapes exhibited consistent pressure field evolution. In particular, cases (a1)–(c1) in Figure 5 followed highly similar combustion pathways: each began with unstable scramjet ignition and transitioned to stable ramjet combustion by approximately 6 ms. This demonstrated the efectiveness of latent trajectory analysis for understanding dynamic combustion behavior.

In a separate analysis, we focused on case (a) to study stable scramjet combustion. As shown in Figure 6(a), the trajectory remained within a tight cluster throughout the simulation. A zoomed-in view was shown in Figure 6(a1), while frame snapshots were shown in (af). In this case, the pressure surges consistently occured between the isolator and cavity, and remained stationary—indicative of stable scramjet operation. According to InternViT-based clustering results, trajectories corresponding to stable scramjet combustion always remained within small round clusters, such as circles (a)–(d) in Figure 6. In contrast, ramjet cases typically originated from such clusters and terminated in larger clusters such as (e), which aggregated stable ramjet frames.

To evaluate the semantic reporting capability, we collaborated with expert E3 to manually annotate representative cluster centroids. As shown in Figure 7, pressing the save description button stored the expert-written description for the centroid of cluster 0 (top-left). The bottom-left panel showed the auto-generated description for the 12th frame of case (a), produced by the Gemma-3 VLM using nearest centroid descriptions and images as context. The right panel showed the case-level summary generated by aggregating frame-level outputs.

Users could access these semantic summaries by clicking on points in the Temporal Trajectory View or selecting frames in the Report View. E3 positively evaluated this feature, citing its ability to accelerate the correction of inappropriate combustion-mode descriptions. The ability to toggle between frame- and trajectory-level summaries allowed for eficient browsing and interpretation of diferent simulation outcomes.

## 8.2 Case Study 2: OH Field Exploration and Combustion Mode Diferentiation

In this case, we focused on the hydroxyl radical (OH) field, which served as an important indicator of combustion activity. To reduce noise and enhance spatial interpretability, the OH field images were preprocessed by cropping them to retain only the core scramjet engine region, removing irrelevant background areas. Based on domain knowledge, we assigned higher weights to the upper region of the engine, where the OH field exhibited stronger variation and contains more informative flame structures. Less active regions, such as the left portion of the domain, received smaller weights. These weighted OH field images were then encoded into high-dimensional embeddings using a pretrained BLIP-2 Vision Transformer (ViT) model, capturing localized combustion characteristics across time.

![](images/4da1af406442eaa69a0a7f104351f973c8f8ad48c80c69d505a7c8aced32b645.jpg)

![](images/7e5b9d77b456e977cfe94f98192732ab1745fce3d38bb9885b820316e776736e.jpg)

c2f1.3h20.10 (Diff 174.08) (b1)  
![](images/d8a7b12db68b9f39178a971ff2b29cea95b1bd3ced61a8a9034d7e9246e7b888.jpg)

c2f1.3h2o.11 (Diff: 203.78) (c1)  
![](images/75652daf70603956e9e3e68924fc5df9d8ffa20fa296c3cab5c5378b5867f0e2.jpg)

![](images/6a14f6c83bcd9f972bf13b4746b19b2731e235d418c8332134f64bbe48fbc9ac.jpg)  
Fig. 5 The case study 1: Ramjet Pressure field exploration. Cropped pressure field frames were processed by InternViT and clustered using DBSCAN. Scatter plot (a1) exhibited the clustering result in the Temporal Trajectory View. (b1) and (c1) were the 2 most similar trajectories calculated in Similar Trajectory View. (a2), (b2) and (c2) were the flow field frames of the cases in (a1), (b1) and (c1), respectively. (d) was the geometric configurations (Shi et al. 2021) of the scramjet engine in our simulations.

![](images/9560ecb5b5a59abf5ab1b1d8d187883b2a6ef451a05f84171123ca470e63fcbb.jpg)  
Fig. 6 The case study 1: Scramjet Pressure fields Exploration. The temporal trajectory of selected case resided in a small cluster (a), zoomed in to (a1), its flow field frames were in figure (af), and similar trajectories at (a2) and (a3). Clusters (a) to (d) had scramjet combustion mode, while the points in the larger cluster surrounding (e) had ramjet combustion mode.

Using the Filtering Panel in visual analytics system, E4 interactively adjusted the ranges of initial conditions, along with DBSCAN clustering parameters including ε and MinPts, to refine the separation of trajectory groups. As illustrated in Figure $^ { 8 , }$ the temporal trajectory of the selected case was shown in region (s1), with its two most similar trajectories located at (s2) and (s3). The corresponding OH field snapshots were presented in regions (f1)–(f3).

The clustering results revealed two distinct combustion modes. The red cluster in the scatter plot corresponded to the shear layer combustion mode, characterized by narrow, stratified flame structures forming along the fuel–air interface. This could be observed in the circled regions of Figure 8(d)(e), where the OH field intensity was concentrated along a thin horizontal band. In contrast, the yellow cluster corresponded to the jet-wake combustion mode, where the flame emerged directly from the fuel inlet and formed a broad, curved tail. This was evident in the circled regions of Figure 8(a)(b), where a large flame wake propagated along the upper part of the field.

![](images/bbb0b85a3033a080bcdf12114957127bffe5c9e97ba4a4a756ccacb81e31b82d.jpg)  
Fig. 7 The case study 1: Report generation on frames and cases. Centroids of the clusters were highlighted in diamonds. We drawed dashed circles around them to highlight. User could inspect or generate frame description on any selected embedding. The description of frames of the selected case were generated from the frame and description of the closest centroids. The case summary was generated from the frames within the case and their descriptions.

Interestingly, E4 also observed that even after the pressure field of a simulation reaches a steady state, the OH field trajectory may continue to fluctuate within a relatively wide region of the embedding space. However, these fluctuations consistently remain within the same cluster. This suggested that while the flame shape exhibited spatial variability, the underlying combustion mode remained stable. This behavior reflected the system’s ability to distinguish transient structural changes from actual combustion mode transitions—enabling robust combustion classification despite internal oscillations.

![](images/e30248128d52999d6d9c881da173c6bf7e2ef817e0de54766bf9abb8bdad061b.jpg)

![](images/62c65dae651819a20dc67fb6ac99c67bc2b695a53f6217428fe8eee7182b7307.jpg)  
Fig. 8 Case Study 2: Clustering of OH Field Trajectories to Identify Combustion Modes. The temporal trajectory of the selected case was shown in region (s1), with its two most similar trajectories located at (s2) and (s3). The corresponding OH field images for these cases were displayed in (f1)–(f3). Two distinct combustion modes were identified: the red cluster corresponds to the shear layer combustion mode, characterized by narrow, stratified flame structures along the fuel–air interface (see circled regions in (d)–(e)); the yellow cluster corresponded to the jet-wake combustion mode, where the flame formed a wide, curved tail extending from the fuel inlet (see circled regions in (a)–(b)). The clustering results demonstrated that the system could efectively separate and characterize combustion behaviors across diferent cases, even when OH field trajectories fluctuated at steady state.

Overall, expert E4 highlighted that the combined use of trajectory views, spatial clustering, and multiframe comparison enabled rapid identification of representative combustion modes and supported diagnostic insight across multiple simulation cases.

## 9 Discussion

## 9.1 Expert Feedback and Insights

In addition to the structured analyses presented in our case studies, we also conducted post-session interviews with the domain experts to gather broader impressions of system usability, interpretability, and analytical value. These sessions aimed to evaluate not only how well the system supported specific tasks, but also how experts perceived its potential for generalization to larger workflows and collaboration scenarios. The following summarizes key insights drawn from these expert reflections.

First, experts consistently emphasized the benefit of visual-semantic alignment enabled by our workflow. By allowing cluster centroids to be annotated and reused across simulations, the system supports knowledge transfer and interpretive consistency across teams. One expert highlighted that “annotating some frames once and inspecting them echoed across dozens of similar cases” allowed them to validate combustion behavior at scale without redoing frame-by-frame inspections.

Second, the semantic report generation was found to be valuable not only as an interpretive tool, but also as a mechanism for cross-expert communication. The VLM-generated summaries, grounded in human-provided insights, served as a shared semantic layer that all experts could use to explain, compare, or question simulation behavior—particularly in unfamiliar parameter regimes. This capacity to produce interpretable narratives from latent structures significantly lowers the barrier to entry for less experienced analysts or interdisciplinary collaborators.

Third, the interactive views—particularly the combination of the Temporal Trajectory View and the Similar Trajectories View—were praised for facilitating rapid understanding and intuitive reasoning about the simulation data. Experts could trace how ignition and stabilization evolved in the embedding space, and then explore comparable cases with a single click. Several experts noted that this not only helped them validate existing hypotheses, but also led to the discovery of unexpected cases exhibiting hybrid combustion characteristics.

Finally, experts viewed the system as scalable to high-volume simulation settings, where hundreds of cases must be triaged or grouped. By turning latent trajectory structure into interpretable visual clusters and linking them to semantic summaries, the system allows domain specialists to move from raw images to hypothesis-driven analysis at a significantly accelerated pace.

## 9.2 Limitations and Opportunities

While TemporalFlowViz demonstrates strong performance and expert usability across combustion interpretation tasks, several design choices present opportunities for future enhancement. Currently, clustering relies on manually selected parameters (e.g., ε, minPts), which allow fine control but may benefit from adaptive tuning strategies for improved consistency across variable datasets.

The pretrained Vision Transformers used in our pipeline are sourced from generalpurpose corpora. Despite their success in capturing combustion-relevant features in pressure and OH fields, task-specific fine-tuning could further enhance sensitivity to subtle flow structures and transitional events.

Expert annotation of cluster centroids has proven efective in anchoring semantic generation, but it still introduces subjectivity. To scale this process, future versions may incorporate structured annotation templates, provenance tracking, or collaborative validation mechanisms to streamline consensus building.

Although our case studies focused on steady and quasi-steady combustion, the system is designed to generalize across unsteady or transient events. Preliminary tests indicate that the current trajectory modeling can serve as a foundation for more expressive representations—such as hierarchical time-scale embedding or event-aware trajectory segmentation—making TemporalFlowViz adaptable to a wider range of phenomena including blowout or re-ignition.

These limitations suggest directions for refinement rather than structural shortcomings, and do not detract from the system’s core contributions in enabling interpretable, expert-guided visual analysis of combustion simulation data.

## 10 Conclusion and Future Work

In this paper, we present TemporalFlowViz, a visual analytics workflow and system for expert-guided exploration, clustering, and interpretation of temporal flow-field data from scramjet combustion simulations. TemporalFlowViz combines ViT embeddings, interactive trajectory construction, density-based clustering, and expert-augmented semantic summarization to support interpretable analysis of high-dimensional, timesequenced combustion data. By allowing experts to annotate representative cluster centroids, which then condition a vision–language model for generating frame- and case-level descriptions, we establish a reusable semantic layer that transforms complex flow field patterns into interpretable textual narratives. This multimodal annotation pipeline enables automated reporting, similarity-driven retrieval, and collaborative analysis with minimal manual overhead. We demonstrate the utility of TemporalFlowViz through two expert-driven case studies focusing on pressure and OH field dynamics. These studies revealed the system’s ability to distinguish combustion modes, uncover evolution patterns, and support scalable interpretation across parameterized simulations. Expert feedback confirmed that the system accelerates hypothesis validation, reduces cognitive load, and facilitates knowledge sharing across teams. In summary, TemporalFlowViz represents a step toward deeply integrated, expert-informed visual analytics for scientific simulations—fusing deep learning, trajectory-based reasoning, and semantic interpretability into a unified framework for understanding combustion dynamics at scale.

We identify two main directions for future research. First, we plan to construct a domain-adaptive Vision Transformer through fine-tuning on large-scale combustion simulation datasets. This will enhance the sensitivity of embeddings to subtle physical structures such as ignition fronts and weak flame zones. Second, we aim to develop a multimodal annotation workflow powered by large vision–language models to support scalable, automated labeling of scientific phenomena. This will reduce expert workload and help surface recurring knowledge patterns across large simulation corpora.

## Declarations

## Open Source Availability

The code and demonstration video are available at: https://github.com/JYF98/ temporalflowviz

## Acknowledgements

We thank all reviewers for their constructive comments. This work was supported by the Strategic Priority Research Program of Chinese Academy of Sciences (Grant No. XDB0500103), the National Natural Science Foundation of China (No. 62202446), and the Beijing Natural Science Foundation (No.4254090).

## References

Bae, J., Helldin, T., Riveiro, M., Nowaczyk, S., Bouguelia, M.-R., Falkman, G.: Interactive clustering: A comprehensive review. ACM Comput. Surv. 53(1) (2020) https://doi.org/10.1145/3340960

Bostock, M., Ogievetsky, V., Heer, J.: D3 data-driven documents. IEEE Transactions on Visualization and Computer Graphics 17(12), 2301–2309 (2011) https://doi.org/ 10.1109/TVCG.2011.185

Curran, E., Heiser, W., Pratt, D.: Fluid phenomena in scramjet combustion systems. Annual Review of Fluid Mechanics 28, 323–360 (2003) https://doi.org/10.1146/ annurev.fl.28.010196.001543

Cai, D., He, X., Wang, X., Bao, H., Han, J.: Locality preserving nonnegative matrix factorization. In: Proceedings of the 21st International Joint Conference on Artificial Intelligence. IJCAI’09, pp. 1010–1015. Morgan Kaufmann Publishers Inc., San Francisco, CA, USA (2009)

Chen, J., Huang, H., Ye, H., Peng, Z., Li, C., Wang, C.: Salientime: user-driven selection of salient time steps for large-scale geospatial data visualization. In: Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, pp. 1–19 (2024)

Chen, H., Jiang, S., Yu, X., Yin, H., Wang, X., Hu, Y., Wang, C., Li, C.: Geovis: a datadriven geographic visualization recommendation system via latent space encoding. Journal of Visualization 27(4), 603–622 (2024)

Chidananda Gowda, K., Krishna, G.: Agglomerative clustering using the concept of mutual nearest neighbourhood. Pattern Recognition 10(2), 105–112 (1978) https: //doi.org/10.1016/0031-3203(78)90018-3

Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al.: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24185–24198 (2024)

Chen, Y., Zhao, Y., Li, X., Zhang, J., Long, J., Zhou, F.: An open dataset of data lineage graphs for data governance research. Visual Informatics 8(1), 1–5 (2024) https://doi.org/10.1016/j.visinf.2024.01.001

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Others: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2021)

Dalal, N., Triggs, B.: Histograms of oriented gradients for human detection. In: 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), vol. 1, pp. 886–893 (2005). IEEE

Eirich, J., Bonart, J., J¨ackle, D., Sedlmair, M., Schmid, U., Fischbach, K., Schreck, T., Bernard, J.: Irvine: A design study on analyzing correlation patterns of electrical engines. IEEE Transactions on Visualization and Computer Graphics 28(1), 11–21 (2021)

Ester, M., Kriegel, H.-P., Sander, J., Xu, X.: A density-based algorithm for discovering clusters in large spatial databases with noise. In: Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD), pp. 226–231 (1996)

Fini, E., Shukor, M., Li, X., Dufter, P., Klein, M., Haldimann, D., Aitharaju, S., Costa, V.G.T., B´ethune, L., Gan, Z., Toshev, A.T., Eichner, M., Nabi, M., Yang, Y., Susskind, J.M., El-Nouby, A.: Multimodal Autoregressive Pre-training of Large Vision Encoders (2024). https://arxiv.org/abs/2411.14402

He, Z., Chen, C., Wu, Y., Tian, X., Chu, Q., Huang, Z., Zhang, W.: Real-time interactive parallel visualization of large-scale flow-field data. Applied Sciences 13(16), 9092 (2023) https://doi.org/10.3390/app13169092

Huang, D., Lai, J., Wang, C.-D.: Ensemble clustering using factor graph. Pattern Recognition 50, 131–142 (2016) https://doi.org/10.1016/j.patcog.2015.08.015

He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. Proceedings of the IEEE conference on computer vision and pattern recognition, 770–778 (2016)

Krizhevsky, A., Sutskever, I., Hinton, G.E.: Imagenet classification with deep convolutional neural networks. In: Advances in Neural Information Processing Systems, pp. 1097–1105 (2012)

Ladeinde, F.: A Critical Review of Scramjet Combustion Simulation (Invited), (2009). https://doi.org/10.2514/6.2009-127 . https://arc.aiaa.org/doi/abs/10.2514/6.2009- 127

Li, C.Y., Liang, X., Hu, Z., Xing, E.P.: Knowledge-driven encode, retrieve, paraphrase for medical image report generation. Proceedings of the AAAI Conference on Artificial Intelligence 33(01), 6666–6673 (2019) https://doi.org/10.1609/aaai. v33i01.33016666

Li, G., Liu, Y., Shan, G., Cheng, S., Cao, W., Wang, J., Wang, K.-C.: Paramsdrag: Interactive parameter space exploration via image-space dragging. IEEE Transactions on Visualization and Computer Graphics 31(1), 624–634 (2025) https: //doi.org/10.1109/TVCG.2024.3456338

Li, J., Li, D., Savarese, S., Hoi, S.: BLIP-2: Bootstrapping Language-Image Pretraining with Frozen Image Encoders and Large Language Models (2023). https: //arxiv.org/abs/2301.12597

Lowe, D.G.: Distinctive image features from scale-invariant keypoints. International journal of computer vision 60(2), 91–110 (2004)

McInnes, L., Healy, J., Melville, J.: UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction (2020). https://arxiv.org/abs/1802.03426

Peng, R., Dong, Y., Li, G., Tian, D., Shan, G.: Textlens: large language modelspowered visual analytics enhancing text clustering. Journal of Visualization, 1–19 (2025)

Reddy, Y.A., Wahl, J., Sj¨odahl, M.: Twins-pivnet: Spatial attention-based deep learn ing framework for particle image velocimetry using vision transformer. Ocean Engineering 318, 120205 (2025) https://doi.org/10.1016/j.oceaneng.2024.120205

Shi, W., Tian, Y., Le, J., Zhong, F.: Efect of pilot hydrogen on the formation of dynamic flame in an ethylene-fueled scramjet with a cavity. Physics of Fluids 33(5) (2021)

Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ram´e, A., Rivi\`ere, M., et al.: Gemma 3 technical report. arXiv preprint arXiv:2503.19786 (2025)

Maaten, L., Hinton, G.: Visualizing data using t-sne. Journal of Machine Learning Research 9(86), 2579–2605 (2008)

Vincent, P., Larochelle, H., Bengio, Y., Manzagol, P.-A.: Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. Journal of machine learning research 11, 3371–3408 (2010)

Wu, C., Chen, Y., Dong, Y., Zhou, F., Zhao, Y., Liang, C.J.: Vizoptics: Getting insights into optics via interactive visual analysis. Computers and Electrical Engineering 107, 108624 (2023)

Wang, Z., Liu, L., Wang, L., Zhou, L.: R2GenGPT: Radiology Report Generation with Frozen LLMs (2023). https://arxiv.org/abs/2309.09812

Wang, S., Seidman, J.H., Sankaran, S., Wang, H., Pappas, G.J., Perdikaris, P.: CViT: Continuous Vision Transformer for Operator Learning (2025). https://arxiv.org/ abs/2405.13998

Wang, M., Tao, J., Ma, J., Shen, Y., Wang, C.: Flowvisual: A visualization app for teaching and understanding 3d flow field concepts. In: Proceedings of the IEEE Pacific Visualization Symposium, pp. 133–140 (2016). IEEE

Wang, J., Wang, J., Ke, Q., Zeng, G., Li, S.: Fast approximate k-means via cluster closures. In: 2012 IEEE Conference on Computer Vision and Pattern Recognition, pp. 3037–3044 (2012). https://doi.org/10.1109/CVPR.2012.6248034

Wang, J., Wang, J., Song, J., Xu, X.-S., Shen, H.T., Li, S.: Optimized cartesian kmeans. IEEE Transactions on Knowledge and Data Engineering 27(1), 180–192 (2015) https://doi.org/10.1109/TKDE.2014.2324592

Wang, W.-T., Wu, Y.-L., Tang, C.-Y., Hor, M.-K.: Adaptive density-based spatial clustering of applications with noise (dbscan) according to data. In: 2015 International Conference on Machine Learning and Cybernetics (ICMLC), vol. 1, pp. 445–451 (2015). https://doi.org/10.1109/ICMLC.2015.7340962

Wang, S., Zhao, Z., Ouyang, X., Wang, Q., Shen, D.: ChatCAD: Interactive Computer-Aided Diagnosis on Medical Image using Large Language Models (2023). https: //arxiv.org/abs/2302.07257

Yu, D., Ian, O., Jie, L., Xiaoru, Y., Vinh, N.Q.: User-centered visual explorer of inprocess comparison in spatiotemporal space. Journal of Visualization 26(2), 403–421 (2023)

Zhang, C., Dong, Y., Wang, Y., Han, Y., Shan, G., Tang, B.: Auragenome: An llm-powered framework for on-the-fly reusable and scalable circular genome visualizations. IEEE Computer Graphics and Applications (01), 1–14 (2025)

Zelnik-Manor, L., Perona, P.: Self-tuning spectral clustering. In: Neural Information Processing Systems (2004). https://api.semanticscholar.org/CorpusID:17066951

Znamenskaya, I.: Methods for panoramic visualization and digital analysis of thermophysical flow fields: A review. arXiv preprint arXiv:2108.01464 (2021)

Zhao, X., Wang, X., Zou, X., Liang, H., Bai, G., Zhang, N., Huang, X., Zhou, F., Zhao, Y.: Graph visualization eficiency of popular web-based libraries. Visual Computing for Industry, Biomedicine, and Art 8(1), 12 (2025) https://doi.org/10.1186/ s42492-025-00193-y