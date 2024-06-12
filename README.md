# Machine Translation Quality Evaluation using Textual Similarity

This repository focuses on using textual similarity as a key metric to evaluate machine translation quality. It builds upon the foundation of the Complex Question Answering Evaluation of GPT-family repository.

## Introduction
Evaluating the quality of machine translations can be challenging. Traditional metrics like BLEU score are widely used, but textual similarity metrics such as cosine similarity, Jaccard index, and others can provide deeper insights into the quality and accuracy of translations. This repository provides tools and scripts to evaluate machine translations using various textual similarity metrics.

## Installation
1. Clone the repository:
    ```
    git clone https://github.com/fivehills/TextSim_MTQE
    cd TextSim_MTQE
    ```
2. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Usage
To evaluate machine translations, you can use the provided evaluation script. Here is an example of how to use it:

1. Prepare your reference and translation texts in separate files.
2. Run the evaluation script:
    ```
    python comput_sim.py --reference path/to/reference.txt --translation path/to/translation.txt
    ```
3. The script will output similarity scores using the implemented metrics.

## Evaluation Metrics
This repository includes the following textual similarity metrics:

- **Cosine Similarity:** Measures the cosine of the angle between two vectors. It ranges from -1 to 1, where 1 indicates identical texts.
- **Jaccard Index:** Measures the similarity between two sets by dividing the size of the intersection by the size of the union. It ranges from 0 to 1, where 1 indicates identical sets.

## Example
Here is a basic example of how to compute textual similarity metrics using the provided functions:

```
from textual_similarity import compute_cosine_similarity, compute_jaccard_index

source = "这是一份参考范文。"
translation = "This is a sample translation text."

text_sim = compute_similarity(source, translation)
jaccard_idx = compute_jaccard(source, translation)

print(f"Cosine Similarity: {text_sim}")
print(f"Jaccard Index: {jaccard_idx}")
```

# Citation:
pleas cite this paper if you use the code and sources in this repo:
```
@article{sun2024textsim,
  title={Textual similarity as a key metric in machine translation quality estimation},
  author={Kun Sun and Rong Wang},
  year={2024},
  journal={ArXiv},
  url={http://arxiv.org/abs/2406.07440}
}
```

The following provides important sources in MT assessments:


# Metrics for MT with reference translations

- [BLEU] (https://github.com/mjpost/sacrebleu)

- [NIST] (https://www.nltk.org/api/nltk.translate.nist_score.html)

- [METEOR](https://github.com/nltk/nltk)

- [CHR F] (https://github.com/mjpost/sacrebleu)

- [BERT SCORE] (https://github.com/Tiiiger/bert_score)

- [BEER] (https://github.com/stanojevic/beer)

- [BLEURT] (https://github.com/google-research/bleurt)

- [BART SCORE] (https://github.com/neulab/BARTScore)




# Datasets and tools on TM evaluations

- [HADQAET](https://github.com/surrey-nlp/HADQAET)

- [QUAK: A Synthetic Quality Estimation Dataset for Korean-English Neural Machine Translation (COLING22)](https://arxiv.org/pdf/2209.15285)

- [MLQE-PE: A Multilingual Quality Estimation and Post-Editing Dataset LREC22](https://aclanthology.org/2022.lrec-1.530/)


- [OpenKiwi](https://github.com/Unbabel/OpenKiwi)

- [MosQEto](https://github.com/zouharvi/MosQEto)

- [DeepQuest](https://github.com/sheffieldnlp/deepQuest)

- [DeepQuest-py](https://github.com/sheffieldnlp/deepQuest-py)

- [TransQuest](https://github.com/mfomicheva/TransQuest)
 

# Evaluation tasks

- [WMT](https://www2.statmt.org/)

- [CCMT](http://mteval.cipsc.org.cn:81/CCMT2022/index.html#2)


# Evaluation overview

- [Kreutzer et al. (2015): QUality Estimation from ScraTCH (QUETCH): Deep Learning for Word-level Translation Quality Estimation]

- [Martins et al. (2016): Unbabel's Participation in the WMT16 Word-Level Translation Quality Estimation Shared Task]

- [Martins et al. (2017): Pushing the Limits of Translation Quality Estimation]

- [Kim et al. (2017): Predictor-Estimator using Multilevel Task Learning with Stack Propagation for Neural Quality Estimation]

- [Wang et al. (2018): Alibaba Submission for WMT18 Quality Estimation Task]

- [Kepler et al. (2019): Unbabel’s Participation in the WMT19 Translation Quality Estimation Shared Task]

- [Specia et al. (2020): Findings of the WMT 2020 Shared Task on Quality Estimation]

- [Specia et al. (2021): Findings of the WMT 2021 Shared Task on Quality Estimation]

- [Zerva et al. (2022): Findings of the WMT 2020 Shared Task on Quality Estimation]

- [[10]Freitag et al,(2023): Results of WMT23 Metrics Shared Task]


# Important Papers on TM assessments

- [Improving Translation Quality Estimation with Bias Mitigation (ACL23)](https://anthology.org/2023.acl-long.121.pdf)

- [Bias Mitigation in Machine Translation Quality Estimation (ACL22)](https://aclanthology.org/2022.acl-long.104.pdf)

- [Unsupervised Quality Estimation for Neural Machine Translation (TACL20)](https://aclanthology.org/2020.tacl-1.35.pdf)

- [Self-Supervised Quality Estimation for Machine Translation (EMNLP21)](https://aclanthology.org/2021.emnlp-main.267.pdf)

- [DirectQE: Direct Pretraining for Machine Translation Quality Estimation (AAAI21)](https://ojs.aaai.org/index.php/AAAI/article/view/17506/17313)

- [Quality Estimation without Human-labeled Data (EACL21)](https://aclanthology.org/2021.eacl-main.50.pdf)

- [Beyond Glass-Box Features: Uncertainty Quantification Enhanced Quality Estimation for Neural Machine Translation (EMNLP21)](https://arxiv.org/pdf/2109.07141)

- [Continual Quality Estimation with Online Bayesian Meta-Learning (ACL21)](https://aclanthology.org/2021.acl-short.25.pdf)

- [Knowledge Distillation for Quality Estimation (ACL21)](https://aclanthology.org/2021.findings-acl.452.pdf)

- [PreQuEL: Quality Estimation of Machine Translation Outputs in Advance (EMNLP22)](https://arxiv.org/pdf/2205.09178)

- [Competency-Aware Neural Machine Translation: Can Machine Translation Know its Own Translation Quality? (EMNLP22)](https://arxiv.org/pdf/2211.13865.pdf)


