# TaxaScope: a container-native, visualization-centric workstation for genome-based bacterial taxonomy

## Introduction

Genome-based bacterial taxonomy requires standardized and reproducible analytical workflows for species delineation and phylogenomic placement; however, the practical deployment of these workflows remains a significant barrier for experimental biologists and clinical scientists. Widely adopted tools such as Prokka, antiSMASH, and PhyloPhlAn underpin key steps in genome annotation, functional characterization, and phylogenomic reconstruction, but their practical deployment in routine laboratory settings, especially on Windows based systems, remains non trivial due to complex software dependencies and command line centric workflows. Existing solutions, including cloud-based platforms (e.g., Galaxy and KBase) and commercial software suites (e.g., CLC Genomics Workbench), partially alleviate these challenges but may also involve considerations related to data-privacy concerns, upload latency, storage quotas, shared computing resources, and recurring licensing costs.

## Methods

To address these limitations, we introduce TaxaScope, a graphical-interface-driven desktop workstation designed to support reproducible, genome-based bacterial taxonomy by integrating a curated set of community-validated tools for genome quality assessment, annotation, phylogenomic inference, genome relatedness estimation, and functional profiling within a unified local graphical user interface (GUI). By leveraging Docker- and Podman-based containerization behind a user-friendly frontend, TaxaScope provides version-locked, standardized execution environments across computing platforms without requiring manual dependency management or prior Linux expertise.

## Result And Discussion

We demonstrate the utility of TaxaScope through a comprehensive re-analysis of
