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
    --modelName "google-bert/bert-large-uncased" \
    --head mlp2 \
    --lrHead 5e-5 \
    --batchSize 8 \
    --dropout 0.2 \
    --epochs 5 \
    --outDir ./saved_models \
    --maxLength 128 \
    --warmupRatio 0.1 \
    --lrEncoder 1e-5
  
or run default setting

python main.py

* --modelName: Name of the pre-trained backbone (e.g., distilbert-base-uncased, google-bert/bert-large-uncased).
* --head: Classification head type (mlp1 for Linear, mlp2 for Deep).
* --outDir: Directory to save the best checkpoint and history.

### 2. Reproducing the Best Result (Submission)
To maximize performance for the final submission, use the --allin flag to train on the full dataset:

python main.py \
    --modelName "google-bert/bert-large-uncased" \
    --head mlp2 \
    --lrHead 5e-5 \
    --batchSize 8 \
    --dropout 0.2 \
    --epochs 5 \
    --outDir ./saved_models \
    --maxLength 128 \
    --warmupRatio 0.1 \
    --lrEncoder 1e-5 \
    --seed 929 \
    --allin

* --allin: Using 99.9% of dataset for training and 0.1% for validation/testing.