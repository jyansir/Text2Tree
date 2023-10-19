# Text2Tree

This repository will include the original implementation and experiment codes of [Text2Tree](). Text2Tree is a framework-agnostic text representation learning algorithm for imbalanced text classification, it uses label tree hierarchy to supplement analysis of rare samples (samples in rare labels) with ones from other labels. Text2Tree is adaptable to any DNN-based learning process (like [DivideMix](https://arxiv.org/abs/2002.07394)), and our work demonstrates its effectiveness in ICD coding tasks (by aligning text representation to ICD code tree).

By adapting Text2Tree to ordinary finetune paradigm, it outperforms common framework-agnostic imbalanced classification methods, and is comparable to advanced hierarchical text classification (HTC) methods.