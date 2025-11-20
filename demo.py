import torch
from transformers import AutoTokenizer
import os
import sys
from model import SentimentClassifier

CHECKPOINT_DIR = "./saved_models/checkpoint"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

def main():
    
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
        
    device = "cpu"
    model = SentimentClassifier.from_pretrained(CHECKPOINT_DIR).to(device)
    model.eval()
    
    os.system("cls")
    print(f"Using device: {device}...")
    print("-" * 30)

    # 4. 定義標籤
    id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

    # 5. 開始互動
    print("Type 'q' to leave")
    while True:
        text = input("\nPlease enter text: ")
        
        if text.lower() in ['q', 'quit', 'exit']:
            break
        if not text.strip():
            continue

        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs['logits']
            
            pred_id = torch.argmax(logits, dim=-1).item()
            probs = torch.softmax(logits, dim=-1)[0]
            confidence = probs[pred_id].item() * 100

        print(f"This text is: {id2label[pred_id]}  (confidence: {confidence:.1f}%)")

if __name__ == "__main__":
    main()