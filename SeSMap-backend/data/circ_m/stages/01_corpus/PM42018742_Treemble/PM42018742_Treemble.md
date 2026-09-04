# Treemble: a graphical tool to generate Newick strings from phylogenetic tree images

## Summary

Phylogenetic trees are ubiquitous and central to biology, but most published trees are available only as visual diagrams and not in the machine-readable Newick format. There are, thus, thousands of published trees in the scientific literature that are unavailable for follow-up analyses, comparisons, and supertree construction. Experts can easily read such diagrams, but the manual construction of a Newick string from a diagram is laborious, error-prone, and time-consuming. Previous attempts to semi-automate the reading of tree images relied on image processing techniques. These often encounter difficulties as typical published tree diagrams contain various graphical elements and annotations that overlap the branches, such as error bars on internal nodes. Here we introduce Treemble, a user-friendly desktop application for generating Newick strings from tree images. The user simply clicks to mark node locations, assisted by a deep learning-based node detection tool, and Treemble algorithmically assembles the tree from the node coordinates alone. Treemble also facilitates the automatic reading of tip name labels and can be used for both rectangular and circular trees.

## Availability And Implementation

Treemble is a native desktop application for macOS and Windows and is freely available, with documentation, at treemble.org. Source code is available at github.com/John-Allard/Treemble. The trained node detection model is available at huggingface.co/John-Allard/treemble-1.
