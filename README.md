# HW3: Sentiment Analysis with Deep Learning

This project implements a Deep Learning-based Sentiment Analysis model to classify social media posts into three categories: Negative, Neutral, and Positive. The solution leverages pre-trained Transformer models (e.g., DistilBERT, BERT) combined with customized classification heads (MLP) and advanced training strategies.

## Environment Setup

It is recommended to use a virtual environment (e.g., conda or venv) to manage dependencies.

1. Create and activate a virtual environment:
   # Example using conda
   conda create -n sentiment python=3.13.7
   conda activate sentiment

2. Install requirements:
   pip install -r requirements.txt

   Dependencies include: torch, transformers, pandas, numpy, scikit-learn, tqdm, matplotlib, seaborn.

## Usage

### 1. Training (Standard Split)
To train the model using the standard train/validation/test split (80/10/10), run:

python main.py \
  --modelName distilbert-base-uncased \
  --head mlp1 \
  --epochs 5 \
  --batchSize 32 \
  --lrEncoder 2e-5 \
  --lrHead 2e-5 \
  --outDir ./saved_models_mlp1

* --modelName: Name of the pre-trained backbone (e.g., distilbert-base-uncased, google-bert/bert-large-uncased).
* --head: Classification head type (mlp1 for Linear, mlp2 for Deep).
* --outDir: Directory to save the best checkpoint and history.

### 2. Reproducing the Best Result (Submission)
To maximize performance for the final submission, use the --allin flag to train on the full dataset:

python main.py \
  --allin \
  --modelName distilbert-base-uncased \
  --head mlp1 \
  --epochs 5 \
  --batchSize 32 \
  --lrEncoder 2e-5 \
  --lrHead 2e-5 \
  --outDir ./saved_models/

### 3. Evaluation
To evaluate a trained checkpoint on a specific CSV file:

python main.py --eval --ckpt ./saved_models/checkpoint --csv ./dataset/test.csv

## Hyperparameters

The default hyperparameters used for the best-performing model (DistilBERT + MLP1) are listed below:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| Backbone | distilbert-base-uncased | Pre-trained Transformer encoder |
| Head Architecture | mlp1 | Linear Probing (768 -> 3) |
| Batch Size | 32 | Balanced for convergence and memory |
| Epochs | 5 | Sufficient for Cosine Annealing convergence |
| Optimizer | AdamW | With Gradient Clipping (norm=1.0) |
| Scheduler | Cosine Annealing | Warmup Ratio = 0.1 |
| Learning Rate | 2e-5 | Applied to both Encoder and Head |
| Dropout | 0.1 | Standard regularization |
| Max Length | 128 | Token sequence length |

## Implementation Details

* Data Processing: Utilized Stratified Split to maintain class distribution across training and validation sets.
* Optimization: Implemented Gradient Clipping to prevent exploding gradients and a Cosine Annealing Scheduler with warmup to ensure smooth convergence.
* Model Checkpointing: The training loop saves the model state with the highest Validation Accuracy, rather than the final epoch's state, to mitigate overfitting.
