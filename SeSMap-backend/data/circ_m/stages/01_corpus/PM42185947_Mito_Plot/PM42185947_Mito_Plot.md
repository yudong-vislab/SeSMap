# Mito_Plot: open-source pipeline for quantification and visualization of mitochondrial DNA heteroplasmy

## Background

Mitochondrial DNA heteroplasmy plays a crucial role in mitochondrial function, aging, and a wide range of human diseases. Recent advances in high-throughput sequencing have enabled large-scale detection of heteroplasmic variants; however, effective cohort-level integration, comparison, and visualization of Mutant Allele Frequency (MAF) values remain challenging. Existing tools often focus on single-sample visualization or require substantial manual preprocessing, limiting their scalability and usability for large cohorts. To address these challenges, we developed Mito_Plot, an open-source computational pipeline designed for standardized quantification and intuitive visualization of Mitochondrial DNA (mtDNA) heteroplasmy across multiple samples.

## Results

Mito_Plot accepts standard mitochondrial VCF files and automatically calculates MAF based on allelic depth information. MAF data from multiple samples are aggregated into a unified matrix aligned by genomic position, enabling direct cross-sample comparison. The pipeline provides interactive two-dimensional circular plots that map MAF onto the mitochondrial genome with gene-level annotations, facilitating rapid identification of mutation hotspots and sample-specific patterns. In addition, Mito_Plot offers optional three-dimensional visualizations that enhance exploration of large cohorts by separating variant distributions across samples and genomic regions. Application of Mito_Plot to multi-sample mitochondrial sequencing datasets demonstrated robust handling of both variants with low and high MAF values, efficient processing of large cohorts, and improved interpretability compared with static or single-sample visualizations.

## Conclusions

Mito_Plot is a scalable, user-friendly software pipeline for cohort-scale quantification and visualization of mtDNA MAF. By integrating standardized MAF calculation with interactive 2D and 3D visualizations, Mito_Plot facilitates comprehensive exploration of mitochondrial variant landscapes across large datasets. The open-source and modular design of the software supports reproducible research and flexible integration into existing analysis workflows, making Mito_Plot a practical resource for mitochondrial genomics research and clinical investigations.
