# eccDNA2Ca: an ensemble deep learning framework for interpretable prediction of cancer-associated extrachromosomal circular DNA

## Background

Extrachromosomal circular DNA (eccDNA) is increasingly recognized as a critical driver of oncogene amplification, therapeutic resistance, and intratumoral heterogeneity in cancer. However, existing computational approaches predominantly focus on eccDNA detection and structural identification, without addressing their functional or clinical significance in oncogenesis.

## Methods

We developed eccDNA2Ca, an interpretable ensemble learning framework that integrates extreme gradient boosting (XGBoost) with deep neural networks (CNN and LSTM) to predict cancer-associated eccDNAs directly from raw sequences. A manually curated dataset comprising 465 experimentally validated human eccDNAs across 16 cancer types was compiled from 21 studies. Models were trained using both engineered genomic features and deep sequence encodings. Interpretability was achieved through SHAP analysis and motif discovery. The final ensemble model was validated on external datasets and implemented as an open-source command-line tool.

## Results

eccDNA2Ca demonstrated superior performance (AUC > 0.96, AUPR > 0.95) across cross-validation and independent test sets, outperforming conventional classifiers. Feature interpretation revealed the dominant contribution of repeat content and specific k-mer frequencies, while motif analysis identified enrichment of zinc finger transcription factor binding sites among cancer-specific eccDNAs. Pan-cancer validation using TCGA data showed strong associations between predicted eccDNAs and tumor mutational burden, immune infiltration, microsatellite instability, and patient survival. Importantly, the model demonstrated robust generalizability across tumor types and eccDNA size categories.

## Conclusions

eccDNA2Ca represents the first interpretable and publicly available computational framework specifically designed to prioritize cancer-relevant eccDNAs after experimental detection based on sequence features. By enabling scalable and biologically informed prioritization of eccDNAs, eccDNA2Ca provides a valuable resource for eccDNA functional studies, cancer biomarker discovery, and translational cancer genomics. Source code and datasets are freely available at: https://github.com/bread1006/eccDNA2Ca. An interactive web server implementing eccDNA2Ca is accessible at: http://43.138.143.50:5000/en.
