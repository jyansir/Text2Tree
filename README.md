# Text2Tree

This repository include the original implementation and experiment codes of Text2Tree([Arxiv](https://arxiv.org/abs/2311.16650)). Text2Tree is a framework-agnostic text representation learning algorithm for imbalanced text classification, it uses label tree hierarchy to supplement analysis of rare samples (samples in rare labels) with ones from other labels. Text2Tree is adaptable to any DNN-based learning process (like [DivideMix](https://arxiv.org/abs/2002.07394)), and our work demonstrates its effectiveness in ICD coding tasks (by aligning text representation to ICD code tree).

By adapting Text2Tree to ordinary finetune paradigm, it outperforms common framework-agnostic imbalanced classification methods, and is comparable to advanced hierarchical text classification (HTC) methods. Moreover, Text2Tree is an orthogonal technical contribution and can be adopted jointly with other methods for imbalanced text classification when label hierarchy relation exists.

## 1. Runtime Environment

All necessary packages are listed in `requirement.txt`, if you do not to perform GNN-based label embedding methods (e.g., HTC baselines, or Text2Tree variants with GNN label encoders), you can omit the `torch_geometric` package. All experiments are conducted on one or more NVIDIA GeForce RTX 3090 GPUs.

## 2. Datasets

All data preprocessing pipelines are given in the `init_data_handler` function in `data_utils.py`. 2 public multi-label datasets and 3 in-house multi-classification datasets are used.

`PubMed(multilabel)`: we use a multi-label PubMed version from this [kaggle link](https://www.kaggle.com/datasets/owaiskhan9654/pubmed-multilabel-text-classification) with fine-grained multi-level MeSH code annotations, you can also simply fetch it through [Huggingface Datasets APIs](https://huggingface.co/datasets/owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH). You can also directly download our preprocessed splits in this [link (update soon)]().

`MIMIC3-top50(multilabel)`: we use the textual parts from the MIMIC-III dataset, you should obtain access authorization from [PhysioNet](https://physionet.org/content/mimiciii/1.4/) and extract textual data with preprocessing pipeline in [CAML-MIMIC](https://github.com/jamesmullenbach/caml-mimic/blob/master/notebooks/dataproc_mimic_III.ipynb) before executing our preprocessing codes. We only retain the disease codes in the 50 most frequent labels, and convert the ICD-9 codes into ICD-10 codes for the similar label coding structure as MeSH codes in PubMed.

`other multilabel datasets`: for other 3 in-house real-world medical record datasets, we are trying to contact the relevant healthcare institutions to grant limited accessibility. If you want to apply them please contact us (email: jyansir@zju.edu.cn)

## 3. Experiments

`run_experiment/` directory contains all main experiment scripts of each baseline with grid search hyperparameter tuning.

For example, you can run experiment of Text2Tree on `PubMed(multilabel)` datasets as follows:

```
CUDA_VISIBLE_DEVICES=X bash run_experiment/pubmed_multilabel/text2tree/grid_search.sh
```

## 4. Applications

Apart from medical text classificaiton, Text2Tree can be further applied in broad imbalanced classification scenarios once label hierarchy is explicitly or implicitly accessible.

## 5. How to cite Text2Tree

```
@inproceedings{yan2023text2tree,
  title={Text2Tree: Aligning Text Representation to the Label Tree Hierarchy for Imbalanced Medical Classification},
  author={Yan, Jiahuan and Gao, Haojun and Kai, Zhang and Liu, Weize and Chen, Danny and Wu, Jian and Chen, Jintai},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  year={2023}
}
```