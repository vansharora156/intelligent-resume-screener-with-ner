import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils import resample, compute_class_weight
import pandas as pd
import numpy as np
import json, joblib, re, os
import matplotlib.pyplot as plt
from data_preparation import load_data_raw, extract_manual_features, clean_text_simple

# 1. MLP Architecture (Requirement: input -> 256 -> 128 -> output)
class ResumeMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ResumeMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.network(x)

def train_model(optimizer_name, X_train, y_train, X_val, y_val, num_classes, weights):
    model = ResumeMLP(X_train.shape[1], num_classes)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # 2. Adam vs RMSProp Comparison (Requirement: Adam vs RMSProp)
    # 3. L2 Regularization (Requirement: weight_decay=1e-4)
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-4)
        
    train_l = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=64, shuffle=True)
    val_X, val_y = torch.FloatTensor(X_val), torch.LongTensor(y_val)
    
    best_acc = 0
    patience_counter = 0
    # 4. Early Stopping (Requirement: patience=5)
    patience = 5
    history = []

    print(f"\nTraining with {optimizer_name}...")
    for epoch in range(50):
        model.train()
        for X_b, y_b in train_l:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(val_X), dim=1)
            acc = accuracy_score(val_y, preds)
            history.append(acc)
            
            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
                if optimizer_name == "Adam": # Save the best one
                    torch.save(model.state_dict(), 'resume_mlp_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"  [{optimizer_name}] Early stopping at epoch {epoch+1}")
                break
    return best_acc, history

def main():
    df = load_data_raw('archive (1)/Resume/Resume.csv')
    categories = sorted(df['Category'].unique())
    
    df_train, df_test = train_test_split(df, test_size=0.15, random_state=42, stratify=df['Category'])
    
    # Balancing
    train_balanced = []
    for cat in categories:
        df_cat = df_train[df_train['Category'] == cat]
        df_upsampled = resample(df_cat, replace=True, n_samples=max(120, len(df_cat)), random_state=42)
        train_balanced.append(df_upsampled)
    df_train = pd.concat(train_balanced)
    
    # Feature Extraction
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english', binary=True, ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(df_train['Resume_str'].fillna('').apply(clean_text_simple))
    X_test_tfidf = tfidf.transform(df_test['Resume_str'].fillna('').apply(clean_text_simple))
    
    selector = SelectKBest(chi2, k=3000)
    X_train_sel = selector.fit_transform(X_train_tfidf, df_train['Category'])
    X_test_sel = selector.transform(X_test_tfidf)
    
    X_train_man = np.array([extract_manual_features(t, categories) for t in df_train['Resume_str']])
    X_test_man = np.array([extract_manual_features(t, categories) for t in df_test['Resume_str']])
    
    scaler = StandardScaler()
    X_train_man = scaler.fit_transform(X_train_man)
    X_test_man = scaler.transform(X_test_man)
    
    X_train = np.hstack((X_train_man, X_train_sel.toarray()))
    X_test = np.hstack((X_test_man, X_test_sel.toarray()))
    
    le = LabelEncoder()
    y_train = le.fit_transform(df_train['Category'])
    y_test = le.transform(df_test['Category'])
    
    weights = torch.FloatTensor(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train))
    
    # Run Comparison
    adam_acc, adam_hist = train_model("Adam", X_train, y_train, X_test, y_test, len(le.classes_), weights)
    rms_acc, rms_hist = train_model("RMSprop", X_train, y_train, X_test, y_test, len(le.classes_), weights)
    
    # Plot Comparison (Requirement 5)
    plt.figure(figsize=(10,6))
    plt.plot(adam_hist, label=f'Adam (Best: {adam_acc:.4f})')
    plt.plot(rms_hist, label=f'RMSProp (Best: {rms_acc:.4f})')
    plt.title('Adam vs RMSProp Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('optimizer_comparison.png')
    print("\n[OK] Saved optimizer_comparison.png")

    # Final Classification Report (Requirement 6)
    best_model = ResumeMLP(X_train.shape[1], len(le.classes_))
    best_model.load_state_dict(torch.load('resume_mlp_model.pth'))
    best_model.eval()
    with torch.no_grad():
        preds = torch.argmax(best_model(torch.FloatTensor(X_test)), dim=1)
        print("\nFinal Classification Report (Adam):")
        print(classification_report(y_test, preds, target_names=le.classes_, zero_division=0))

    # Save artifacts
    joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(selector, 'selector.joblib')
    with open('model_config.json', 'w') as f:
        json.dump({'input_size': X_train.shape[1], 'num_classes': len(le.classes_)}, f)
    
    print("\n[OK] Model & artifacts saved!")

if __name__ == '__main__':
    main()
