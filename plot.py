import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6)) 
        plt.imshow(cm, interpolation='nearest', cmap='Blues')

        # 2. 加入顏色條 (Seaborn 預設有，Matplotlib 要手動加)
        plt.colorbar()

        # 3. 設定座標軸標籤
        classes = ["Negative", "Neutral", "Positive"]
        plt.xticks(np.arange(len(classes)), classes)
        plt.yticks(np.arange(len(classes)), classes)

        # 4. 關鍵步驟：手動在每個格子填入數字 (取代 annot=True)
        thresh = cm.max() / 2. # 設定顏色閾值，超過一半變深色，文字改白色
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    verticalalignment="center",
                    # 如果背景顏色很深，字就用白色，否則用黑色
                    color="white" if cm[i, j] > thresh else "black")
        
        # 5. 設定標題與軸名稱 (維持原樣)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix - {split}')
        
        save_path = os.path.join(ckptDir, f"{split}_cm.png")
        plt.savefig(save_path)
        plt.close() 