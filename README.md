# NLPromptEval - Official Code from "What Makes a Good Natural Language Prompt?"

This repository provides code and instructions for reproducing the figures and tables presented in the paper:

**Title:** [What Makes a Good Natural Language Prompt?](https://arxiv.org/pdf/2506.06950)

### Paper Link:
[https://arxiv.org/pdf/2506.06950](https://arxiv.org/pdf/2506.06950)

The purpose of this repository is to facilitate the reproduction of results presented in the paper, specifically the figures and tables demonstrating the performance of various large language models on instruction-following tasks.


**Setup Instructions**
1. Clone the repository
To clone the repository, run the following command:

```sh
git clone https://github.com/dxlong2000/NLPromptEval.git
```

2. Install dependencies

3. Reproducing the figures and tables

To reproduce Figure 1: Correlations of properties evaluated by GPT-4o, run 
```sh
python / 
```

To reproduce Table 2: Performance of models (%) on various tasks under different configurations
```sh
cd inference-codes
```
and then run the script for the corresponding models and datasets.

To reproduce Table 3: Performance of two fine-tuned Qwen-2.5-7B-it models, run 
```sh
python /finetuning-codes/finetuning_qwen.py
```


**Citations:**

If you use this repository in your work, please cite the original paper:

```sh
@article{long2025makes,
  title={What Makes a Good Natural Language Prompt?},
  author={Long, Do Xuan and Dinh, Duy and Nguyen, Ngoc-Hai and Kawaguchi, Kenji and Chen, Nancy F and Joty, Shafiq and Kan, Min-Yen},
  journal={arXiv preprint arXiv:2506.06950},
  year={2025}
}
```

**License**

This repository is licensed under the MIT License - see the LICENSE file for details.