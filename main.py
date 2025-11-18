'''
HW3: Sentiment Analysis with Deep Learning

In this homework, we will explore the fascinating field of sentiment analysis using deep learning techniques. 
Specifically, we will focus on multi-class classification, where the goal is to predict each sentence from social media 
as belonging to the label.

Label definition:
0 -> Negative
1 -> Neutral
2 -> Positive
'''

import os
import re
import gc
import json
import random
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    PretrainedConfig,
    PreTrainedModel
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


def setSeed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# FLOPs estimation
def estimate_flops(hidden_size: int, num_layers: int, seq_len: int, batch_size: int) -> float:
    '''
    Roughly estimate the number of floating point operations (FLOPs) 
    per training step for models

    Args:
        hidden_size: model embedding dimension
        num_layers: number of encoder layers
        seq_len: number of tokens per input
        batch_size: number of samples processed per step

    Returns:
        Estimated FLOPs per training step (in GFLOPs)
    '''
    pass


# Dataset
class SentimentDataset(Dataset):
    def __init__(self, dataPath: str, tokenizer: AutoTokenizer, maxLength: int):
        """
        Step1. Load the CSV file using pandas -> HINT: use pd.read_csv(csv_path)
        Step2. Extract text and label columns -> HINT: df["text"].tolist(), df["label"].tolist()
        Step3. Store tokenizer and max_length for later use

        Args:
            csv_path: Path to the CSV file (with columns 'text' and 'label')
            tokenizer: Pre-trained tokenizer from Hugging Face
            max_length: Maximum token length for padding/truncation
        """
        self.data = pd.read_csv(dataPath)
        self.text = self.data["text"].tolist()
        self.label = self.data["label"].tolist()
        self.tokenizer = tokenizer
        self.maxLength = maxLength

    def __len__(self):
        """
        Returns:
            Total number of samples in the dataset -> HINT: len(self.texts)
        """
        return len(self.text)

    def __getitem__(self, index):
        """
        Step1. Select text and label by index -> HINT: text = self.texts[idx]; label = self.labels[idx]
        Step2. Tokenize the text -> HINT: use self.tokenizer with truncation, padding, max_length, return_tensors="pt"
        Step3. Convert results to proper tensor format -> HINT: enc["input_ids"].squeeze(0)
        Step4. Return a dictionary

        Returns:
            One sample (tokenized text and label)
        """
        encoding = self.tokenizer(self.text[index], truncation=True, padding="max_length", max_length=self.maxLength, return_tensors="pt") 
        label = self.label[index]
        token = encoding["input_ids"].squeeze(0)
        mask = encoding["attention_mask"].squeeze(0)
        return {"input_ids": token, "attention_mask": mask, "labels": torch.tensor(label)}


# Model Architecture Components
class CustomBlock(nn.Module): 
    def __init__(self): 
        """
        Initialize the layers and parameters of this block.

        HINTS:
        - Always call super().__init__() first to inherit from nn.Module.
        - Define any sub-layers you need (e.g., Linear, Conv1d, Dropout).
        - Store any configuration parameters (e.g., hidden size, kernel size).
        """
        pass

    def forward(self): 
        """
        Define how data moves through the block.

        Args:
            ... : input tensor(s)

        Returns:
            The transformed output tensor.

        HINTS:
        - The forward pass describes the actual computation.
        - Use the layers defined in __init__ to process the input.
        - Make sure to return the final output tensor.
        """
        pass


# Example of Custom Block
class CustomMLP(nn.Module):
    def __init__(self, inputDim, hiddenDim):
        super().__init__()
        self.f1 = nn.Linear(inputDim, hiddenDim)
        self.f2 = nn.Linear(hiddenDim, inputDim)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.f1(x)
        out = self.relu(out)
        out = self.f2(out)
        return out


# Model Config
class SentimentConfig(PretrainedConfig):
    model_type = "..."  # describe this model type

    def __init__(
        self,
        model: str, # name of pre-trained model backbone
        labelNum=3,     # number of output classes (Negative, Neutral, Positive)
        head="mlp",       # classifier head
                          # other hyperparameters
        **kwargs,
    ):
        # Always call the parent class initializer first
        super().__init__(**kwargs)
        '''
        Save all hyperparameters to self
        
        Example:
        self.model_name = model_name
        self.num_labels = num_labels
        self.head = head
        ...
        self.other_hyperparam = other_hyperparam
        
        These attributes will be automatically saved in config.json
        when you call `config.save_pretrained("./path")`.
        '''
        self.model = model
        self.labelNum = labelNum
        self.head = head


# Model (DO NOT change the name "SentimentClassifier")
class SentimentClassifier(PreTrainedModel):
    config_class = SentimentConfig # Which config class to use

    def __init__(self, config: Optional[SentimentConfig] = None):
        super().__init__(config)
        """
        Initialize the model components using the configuration.

        HINTS: 
        - Use AutoModel.from_pretrained(config.model_name)
        - Get hidden size from encoder config
        - Conduct layer normalization
        - Define classifier head
        - Use dropout layer for regularization
        - Apply loss function
        - Store any other hyperparameters from config

        Example:
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.norm = nn.LayerNorm(self.hidden_size)
        self.head_type = config.head
        self.head = CustomBlock(self.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.dropout)
        self.loss_fn = nn.CrossEntropyLoss()
        ...
        """
        self.encoder = AutoModel.from_pretrained(config.model)
        self.hiddenSize = self.encoder.config.hidden_size
        self.norm = nn.LayerNorm(self.hiddenSize)
        self.labelNum = config.labelNum
        self.headType = config.head
        self.head = CustomMLP(self.hiddenSize, self.labelNum)
        self.loss = nn.CrossEntropyLoss()
        #self.dropout =

    def forward(self, inputIds, attentionMask=None, tokenTypeIds=None, labels=None):
        """
        Defines how the input data flows through the model.

        Args:
            input_ids: tokenized input sequences
            attention_mask: masks for padding tokens
            labels: ground-truth labels (optional, for training)

        Returns:
            Dictionary with "logits" (and optionally loss)
        
        HINTS:
        - Pass inputs through the encoder
        - Apply dropout and classifier head
        - Compute loss if labels are provided
        - Return logits (and loss if computed)

        Example:
        outputs = self.encoder(...)
        feat = outputs.last_hidden_state
        feat = self.dropout(self.norm(feat))
        logits = self.head(feat)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels)
        return result
        """
        output = self.encoder(input_ids=inputIds, attention_mask=attentionMask)
        feature = output.last_hidden_state 
        feature = feature[:, 0, :]
        #feature = self.dropout(self.norm(feature))
        logits = self.head(feature)
        results = {"logits": logits}
        if labels is not None:
            results["loss"] = self.loss(logits, labels)
            return results


# Evaluation
@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model accuracy on a given dataset.

    Args:
        model: the trained PyTorch model
        dataloader: DataLoader for validation or test set

    Returns:
        acc: overall accuracy
        all_y: true labels
        all_pred: predicted labels
    """
    model.eval()  
    allY, allPred = [], []
    with torch.inference_mode():  
        for batch in dataloader:
            '''
            HINTS:
            - Move the batch to the correct device (GPU/CPU)
            - Run a forward pass through the model
            - Get predicted class from logits
            - Save ground-truth and predicted labels
            '''
            pass
    acc = accuracy_score(allY, allPred)
    return acc, np.array(allY), np.array(allPred)


# Training Loop
def train(
    modelName: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    outDir: str,
    epochs: int,
    batchSize: int,
    maxLength: int,
    lrEncoder: float,
    lrHead: float,
    dropout: float,
    warmupRatio: float,
    seed: int = 42
):
    '''
    HINTS:
    - Setup & Reproductibility
    - Prepare datasets and dataloaders
    - Initialize the model
    - Set up optimizer and learning rate scheduler
    - Run the training loop (and save the best checkpoint)
    - Evaluation and save results and metrics
    '''

    # 1. Setup & Reproducibility
    setSeed(seed)
    os.makedirs(outDir, exist_ok=True)

    # 2. Prepare datasets and dataloaders (train, val, test)
    '''
    Example:
    tokenizer = AutoTokenizer.from_pretrained(...)
    ds = SentimentDataset(...)
    dl = DataLoader(...)
    '''
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    trainDs = SentimentDataset(train_csv, tokenizer, maxLength)
    testDs = SentimentDataset(test_csv, tokenizer, maxLength)
    valDs = SentimentDataset(val_csv, tokenizer, maxLength)
    trainDl = DataLoader(trainDs, batchSize, shuffle=True)
    testDl = DataLoader(testDs, batchSize, shuffle=True)
    vlaDl = DataLoader(valDs, batchSize, shuffle=True)
    
    # 3. Initialize the model
    '''
    Example:
    config = SentimentConfig(...)
    model = SentimentClassifier(...).to(DEVICE)
    '''
    config = SentimentConfig(modelName)
    classifier = SentimentClassifier(config).to(DEVICE)

    # 4. Set up optimizer and learning rate scheduler
    '''
    Example:
    optimizer = optim.AdamW(...)
    scheduler = get_linear_schedule_with_warmup(...)
    '''

    # 5. Run the training loop
    bestVal = -1.0
    ckptDir = os.path.join(outDir, "checkpoint") # DO NOT change the file name
    os.makedirs(ckptDir, exist_ok=True)
    tokenizer.save_pretrained(ckptDir)

    for epoch in range(1, epochs + 1):
        classifier.train()  
        runningLoss = 0.0
        pbar = tqdm(trainDl, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            '''
            HINTS:
            - Move data to GPU/CPU (the same when doing evaluation)
              -> batch = ...

            - Reset gradients 
              -> optimizer.zero_grad(...)

            - Forward pass
              -> outputs = model(...)
              -> loss = outputs[...]
            
            - Backpropagation
              -> loss.backward()
              -> torch.nn.utils.clip_grad_norm_(...)
            
            - Optimizer step and scheduler update
              -> optimizer.step()
              -> scheduler.step()

            - Update running loss
              -> running_loss += loss.item()
            '''
            inputIds = batch["input_ids"].to(DEVICE)
            attentionMask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            output = classifier(inputIds=inputIds, attentionMask=attentionMask, labels=labels)
            loss = output["loss"]
            loss.backward()
            runningLoss +=  loss.item()

            # Display
            pbar.set_postfix(loss=f"{runningLoss / (pbar.n or 1):.4f}")

        # Validation Phase
        valAcc, _, _ = evaluate(model, dlVal)
        print(f"Epoch {epoch}: Val Acc = {valAcc:.4f}")

        # Save best model checkpoint
        if valAcc > bestVal:
            bestVal = valAcc
            model.save_pretrained(ckptDir)

    # 6. Evaluation and save results and metrics
    best = SentimentClassifier.from_pretrained(ckptDir).to(DEVICE)

    def eval(split, dl):
        acc, y, yhat = evaluate(best, dl)
        '''
        Save confusion matrix and classification report (you should plot the result prettier)

        Example:
        cm = confusion_matrix(y, yhat, labels=[0,1,2])
        pd.DataFrame(cm).to_csv(os.path.join(ckpt_dir, f"{split}_cm.csv"))
        rpt = classification_report(y, yhat, digits=4, labels=[0,1,2])
        with open(os.path.join(ckpt_dir, f"{split}_report.txt"), "w") as f:
            f.write(rpt)
        '''
        return float(acc)

    train_acc = eval("train", dl_train)
    val_acc   = eval("val", dl_val)
    test_acc  = eval("test", dl_test)

    # Save Summary in json format
    summary = {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "params_trainable": int(sum(p.numel() for p in best.parameters() if p.requires_grad))
    }
    with open(os.path.join(outDir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Cleanup
    try:
        best.to("cpu"); model.to("cpu")
    except Exception:
        pass
    del best, model, tokenizer, optimizer, scheduler, dl_train, dl_val, dl_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Main

def main():
    parser = argparse.ArgumentParser()
    # file paths
    parser.add_argument("--train_csv", type=str, default="./dataset/train.csv")
    parser.add_argument("--test_csv", type=str, default="./dataset/test.csv")
    parser.add_argument("--outDir", type=str, default="./saved_models/") # DO NOT change the file name [cite: 1019]
    
    # model / data
    parser.add_argument("--modelName", type=str, default="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    parser.add_argument("--maxLength", type=int, default=256)
    parser.add_argument("--batchSize", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    
    # architecture
    parser.add_argument("--head", type=str, choices=["mlp"], default="mlp")
    parser.add_argument("--dropout", type=float, default=0.5)
    
    # optimization
    parser.add_argument("--lrEncoder", type=float, default=1e-5)
    parser.add_argument("--lrHead", type=float, default=1e-5)
    parser.add_argument("--warmupRatio", type=float, default=0)
    
    # Setup
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    '''
    HINTS:
    - Load the dataset and split into training / validation 
    - Save split data
    - Start Training

    Example:
    # train/val split
    full = pd.read_csv(...)

    train_df, val_df = train_test_split(...)
    os.makedirs(arg.out_dir, exist_ok=True)
    train_split = os.path.join(...)
    val_split = os.path.join(...)
    train_df.to_csv(...)
    val_df.to_csv(...)

    # Start training
    train(
        model_name=args.model_name,
        train_csv=train_split,
        val_csv=val_split,
        test_csv=args.test_csv,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
                                     # any other hyperparameters you want to add (e.g., learning rate, dropout, etc.)
        seed=args.seed,
    )
    '''
    setSeed(args.seed)
    fullData = pd.read_csv(args.train_csv)
    trainData, validData = train_test_split(fullData, test_size=0.1, random_state=args.seed, stratify=fullData["label"])
    os.makedirs(args.outDir, exist_ok=True)
    trainSplitPath = os.path.join(args.outDir, "train_split.csv")
    validSplitPath = os.path.join(args.outDir, "val_split.csv")
    trainData.to_csv(trainSplitPath, index=False)
    validData.to_csv(validSplitPath, index=False)

    train(
        modelName=args.modelName,
        train_csv=trainSplitPath,
        val_csv=validSplitPath,
        test_csv=args.test_csv,
        outDir=args.outDir,
        epochs=args.epochs,
        batchSize=args.batchSize,
        maxLength=args.maxLength,
        seed=args.seed,
        lrEncoder=args.lrEncoder,
        lrHead=args.lrHead,
        dropout=args.dropout,
        warmupRatio=args.warmupRatio
    )

if __name__ == "__main__":
    main()

