# NLPromptEval - Reproduction of Figures and Tables from "NLPromptEval"

This repository provides code and instructions for reproducing the figures and tables presented in the paper:

**Title:** [NLPromptEval: Evaluating Large Language Models for Instruction Following](https://arxiv.org/pdf/2506.06950)

### Paper Link:
[https://arxiv.org/pdf/2506.06950](https://arxiv.org/pdf/2506.06950)

The purpose of this repository is to facilitate the reproduction of results presented in the paper, specifically the figures and tables demonstrating the performance of various large language models on instruction-following tasks.

## Requirements

To run the code and reproduce the results from the paper, you will need to set up your environment with the following dependencies:

- Python 3.x (recommend using `python 3.7+`)
- Required Python libraries:
  ```bash
  pip install -r requirements.txt
Main Dependencies:
transformers (Hugging Face library for pre-trained models)

datasets (Hugging Face dataset library for dataset loading and processing)

matplotlib (for plotting figures)

pandas (for data manipulation)

scikit-learn (for performance evaluation metrics)

Repository Structure
The directory structure of this repository is as follows:

php
Sao chép
NLPromptEval/
├── README.md               # This file
├── requirements.txt         # List of required Python packages
├── finetuning-codes/        # Fine-tuning scripts for instruction-following tasks
│   ├── ...
├── results/                 # Folder containing generated figures and tables
│   ├── figure_1.png         # Example output from figure 1 of the paper
│   ├── table_1.csv          # Example table 1 output
├── data/                    # Datasets used for fine-tuning and evaluation
├── scripts/                 # Scripts for model evaluation and result generation
│   ├── eval.py              # Main evaluation script
├── notebooks/               # Jupyter notebooks to visualize results
└── ...
Setup Instructions
1. Clone the repository
To clone the repository, run the following command:

bash
Sao chép
git clone https://github.com/dxlong2000/NLPromptEval.git
cd NLPromptEval
2. Install dependencies
First, make sure you have python3 and pip installed. Then, create a virtual environment (optional but recommended) and install the dependencies:

bash
Sao chép
python -m venv venv
source venv/bin/activate    # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
3. Download the datasets
This repository uses specific datasets for evaluating large language models. You can download the necessary datasets using the provided datasets.py script or by following the instructions in the script.

```sh
python scripts/download_datasets.py
```

4. Reproducing the figures and tables
To reproduce the figures and tables from the paper:

Model Evaluation:
Run the evaluation script to generate results for different models on instruction-following tasks:

bash
Sao chép
python scripts/eval.py --model <model_name> --dataset <dataset_name> --output_dir results/
Replace <model_name> with the model you wish to evaluate (e.g., gpt-3.5, t5-large) and <dataset_name> with the dataset you want to evaluate on (e.g., super_glue).

Reproducing Figures:
After running the evaluation, the results (tables and metrics) will be stored in the results/ directory. The figures (such as bar charts, accuracy plots) will automatically be generated in the results/ folder.

You can generate plots manually using:

bash
Sao chép
python scripts/plot_results.py --results_dir results/
Reproducing Tables:
Tables summarizing the evaluation metrics for each model will be saved in CSV format inside the results/ folder.

You can inspect and modify the scripts/eval.py file for additional customization on the output format.

5. Visualizing results
You can also visualize the results interactively with Jupyter notebooks located in the notebooks/ directory. Open the notebook in Jupyter to view detailed plots and analysis of the results:

bash
Sao chép
jupyter notebook notebooks/figure_visualization.ipynb
Citations
If you use this repository in your work, please cite the original paper:

mathematica
Sao chép
@article{long2025nlprompteval,
  title={NLPromptEval: Evaluating Large Language Models for Instruction Following},
  author={Long, D. X. et al.},
  journal={arXiv preprint arXiv:2506.06950},
  year={2025},
}
License
This repository is licensed under the MIT License - see the LICENSE file for details.