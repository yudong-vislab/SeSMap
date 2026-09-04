# Comparative analysis of eccDNA and circRNA tools shows increased accuracy of tool combination

## Introduction

Circular nucleic acids such as extrachromosomal circular DNA (eccDNA) and circular RNA (circRNA) are increasingly recognized for their biological relevance and potential as biomarkers in disease contexts. Despite their growing importance, their detection remains challenging due to tool-specific biases, limited validation frameworks, and high variability in performance across datasets.

## Methods

We benchmarked 10 circle detection tools across diverse conditions using both simulated and biological datasets. Our evaluation included classical performance metrics and a novel internal measure of read distribution symmetry ($\Delta$CJ) to assess circle prediction confidence. We explored the impact of sequencing protocols, filtering strategies, and combined tool consensus.

## Results

We found that detection accuracy was highly influenced by sequencing depth, alignment algorithm, and experimental enrichment protocols. $\Delta$CJ proved effective in flagging potential false positive circles, showing improved accuracy of Intersect (circles detected by all tools) and Rosette (circles detected by $\ge$2 tools) combinations.

## Discussion

This study offers a broad evaluation of circular detection tools, suggesting that the combination of $\ge$3 tools is necessary for a correct prediction. These insights will inform future experimental design and data analysis pipelines in both experimental and clinical settings.
