import os
import re
import gc
import json
import random
import argparse
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    PretrainedConfig,
    PreTrainedModel
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setSeed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

class sentimentDataset(Dataset):
    def __init__(self, dataPath: str, tokenizer: AutoTokenizer, maxLength: int):
        self.data = pd.read_csv(dataPath)
        self.text = self.data["text"].tolist()
        self.label = self.data["label"].tolist()
        self.tokenizer = tokenizer
        self.maxLength = maxLength
    def __len__(self):
        return len(self.text)
    def __getitem__(self, index):
        encoding = self.tokenizer(self.data[index], truncation=True, padding="max_length", max_length=self.maxLength, return_tensors="pt") 
        label = self.label[index]
        token = encoding["input_ids"].squeeze(0)
        mask = encoding["attention_mask"].squeeze(0)
        return {"input_ids": token, "attention_mask": mask, "labels": torch.tensor(label)} 

def train(
    modelName: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    outDir: str,
    epochs: int,
    batchSize: int,
    maxLength: int,
    seed: int = 42,
    lrEncoder: float,
    lrHead: float,
    dropout: float,
    warmupRatio: float
):   
    setSeed(seed)
    os.makedirs(outDir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

    

def main():
    parser = argparse.ArgumentParser()
    # file paths
    parser.add_argument("--train_csv", type=str, default="./dataset/train.csv")
    parser.add_argument("--test_csv", type=str, default="./dataset/test.csv")
    parser.add_argument("--outDir", type=str, default="./saved_models/") # DO NOT change the file name [cite: 1019]
    
    # model / data
    parser.add_argument("--modelName", type=str, default="...")
    parser.add_argument("--maxLength", type=int, default=int)
    parser.add_argument("--batchSize", type=int, default=int)
    parser.add_argument("--epochs", type=int, default=int)
    
    # architecture
    parser.add_argument("--head", type=str, choices=["mlp"], default="mlp")
    parser.add_argument("--dropout", type=float, default=float)
    
    # optimization
    parser.add_argument("--lrEncoder", type=float, default=float)
    parser.add_argument("--lrHead", type=float, default=float)
    parser.add_argument("--warmupRatio", type=float, default=float)
    
    # Setup
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    setSeed(args.seed)
    fullData = pd.read_csv(args.train_csv)
    trainData, validData = train_test_split(fullData, test_size=0.1, random_state=args.seed, stratify=fullData["label"])
    os.makedirs(args.out_dir, exist_ok=True)
    trainSplitPath = os.path.join(args.out_dir, "train_split.csv")
    validSplitPath = os.path.join(args.out_dir, "val_split.csv")
    trainData.to_csv(trainSplitPath, index=False)
    validData.to_csv(validSplitPath, index=False)

if __name__ == "__main__":
    main()