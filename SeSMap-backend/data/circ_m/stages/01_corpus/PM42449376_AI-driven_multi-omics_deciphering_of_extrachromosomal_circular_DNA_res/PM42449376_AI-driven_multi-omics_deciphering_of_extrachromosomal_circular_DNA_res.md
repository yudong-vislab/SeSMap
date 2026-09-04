# AI-driven multi-omics deciphering of extrachromosomal circular DNA rescues hidden targets in colorectal cancer

## Background

Extrachromosomal circular DNA (eccDNA) is an emerging tumor biomarker notable for its tumor-specific amplification and contribution to genomic heterogeneity. However, the substantial heterogeneity of eccDNAs poses a significant challenge for characterization. Current identification tools rely predominantly on single data types, overlooking the integrative potential of multi-omics layers and often discarding sparse but biologically meaningful signals.

## Methods

To address this, we developed eccDNAOmix, an AI-empowered framework that integrates a specialized eccDNA sequencing pipeline with multi-omics data to evaluate the relative contributions of different biological modalities. Basing matched tumor and adjacent normal tissues from colorectal cancer patients, eccDNAOmix employs a masking strategy to eliminate non-biological noise and features a multimodal deep learning model with an adaptive gated fusion mechanism.

## Results

Our model achieved robust identification performance (AUC = 0.844), with the DNA sequence modality (AUC = 0.822) being the most identifiable feature for eccDNA identification. Leveraging this finding, we established a web server to facilitate practical eccDNA identification. Biological characterization showed that eccDNAs are preferentially derived from promoter regions and exhibit a significant enrichment of RNA modifications within 100 bp downstream of junction sites. Notably, eccDNAOmix rescued hidden eccDNA fragments from conventionally discarded unmapped sequencing reads and this was confirmed via experimental validation.

## Conclusions

By integrating multi-omics sequencing and eccDNA sequencing, this study provides the first comprehensive characterization of eccDNAs. Our multi-omics analysis demonstrates that the DNA sequence contributes the highest proportion to eccDNA identification. By leveraging this dominant identifiable role and providing an accessible single-modality identification tool, our work proposes a practical computational framework for advancing discovery in colorectal cancer.
