# NLPromptEval - Reproduction of Figures and Tables from "NLPromptEval"

This repository provides code and instructions for reproducing the figures and tables presented in the paper:

**Title:** [NLPromptEval: Evaluating Large Language Models for Instruction Following](https://arxiv.org/pdf/2506.06950)

### Paper Link:
[https://arxiv.org/pdf/2506.06950](https://arxiv.org/pdf/2506.06950)

The purpose of this repository is to facilitate the reproduction of results presented in the paper, specifically the figures and tables demonstrating the performance of various large language models on instruction-following tasks.


Setup Instructions
1. Clone the repository
To clone the repository, run the following command:

```sh
git clone https://github.com/dxlong2000/NLPromptEval.git
cd NLPromptEval
```

2. Install dependencies
First, make sure you have python3 and pip installed. Then, create a virtual environment (optional but recommended) and install the dependencies:


3. Download the datasets
This repository uses specific datasets for evaluating large language models. You can download the necessary datasets using the provided datasets.py script or by following the instructions in the script.

```sh
python scripts/download_datasets.py
```

4. Reproducing the figures and tables
To reproduce the figures and tables from the paper:

Model Evaluation:
Run the evaluation script to generate results for different models on instruction-following tasks:

python scripts/eval.py --model <model_name> --dataset <dataset_name> --output_dir results/
Replace <model_name> with the model you wish to evaluate (e.g., gpt-3.5, t5-large) and <dataset_name> with the dataset you want to evaluate on (e.g., super_glue).

Reproducing Figures:
After running the evaluation, the results (tables and metrics) will be stored in the results/ directory. The figures (such as bar charts, accuracy plots) will automatically be generated in the results/ folder.

You can generate plots manually using:

python scripts/plot_results.py --results_dir results/
Reproducing Tables:
Tables summarizing the evaluation metrics for each model will be saved in CSV format inside the results/ folder.

You can inspect and modify the scripts/eval.py file for additional customization on the output format.

5. Visualizing results
You can also visualize the results interactively with Jupyter notebooks located in the notebooks/ directory. Open the notebook in Jupyter to view detailed plots and analysis of the results:


**Citations**
If you use this repository in your work, please cite the original paper:

@article{long2025nlprompteval,
  title={NLPromptEval: Evaluating Large Language Models for Instruction Following},
  author={Long, D. X. et al.},
  journal={arXiv preprint arXiv:2506.06950},
  year={2025},
}

**License**
This repository is licensed under the MIT License - see the LICENSE file for details.