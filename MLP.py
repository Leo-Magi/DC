import time
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import random
import csv
import seaborn as sns
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_p=0.5):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout_p)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.output_layer(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

selection = 'Fusion'
data = pd.read_csv(f'{selection}_data.csv')

X_rgb = data[['Red', 'Green', 'Blue']].values
X_rgb = X_rgb / 255.0

scaler = StandardScaler()
X = scaler.fit_transform(X_rgb)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Location'].values)

test_size = 0.35
val_size = 0.3
condition_accuracy = 1
condition_precision = 1

log_file = f'training_log_{selection}.csv'

with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Seed', 'Batch Size', 'Learning Rate', 'Hidden Size', 'Val Accuracy', 'Val Precision', 'Val Recall', 'Val F1', 'Loss', 'Epoch'])

def train_and_evaluate(seed, hidden_size, batch_size, learning_rate):
    set_seed(seed)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=seed, stratify=y_temp)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = MLP(input_size=3, hidden_size=hidden_size, num_classes=7, dropout_p=0.01).to(device)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train.cpu().numpy()), y=y_train.cpu().numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.95)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    for epoch in range(350):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        model.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        loss = running_loss / len(train_loader.dataset)

        print(f'Epoch [{epoch + 1}/350], Loss: {loss:.4f}, Val Accuracy: {accuracy:.4f}, Val Precision: {precision:.4f}, Val Recall: {recall:.4f}, Val F1: {f1:.4f}')

        if epoch == 349:
            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([seed, batch_size, f"{learning_rate:.6f}", hidden_size, accuracy, precision, recall, f1, loss, epoch + 1])

        scheduler.step(loss)

        if accuracy >= condition_accuracy and precision >= condition_precision:
            torch.save(model.state_dict(), f'best_model_{selection}.pth')
            print(f'Model saved! number: Seed={seed}, Batch Size={batch_size}, Learning Rate={learning_rate:.6f}, Hidden Size={hidden_size}')
            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([seed, batch_size, f"{learning_rate:.6f}", hidden_size, accuracy, precision, recall, f1, loss, epoch + 1])
            break

    model_save_path = f'best_model_{selection}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved: {model_save_path}')

    plt.rc('font', family='Arial', size=16)
    class_names = ['AH', 'GN', 'GZ', 'MG', 'MH', 'RL', 'ZJ']
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 7))

    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                     xticklabels=class_names, yticklabels=class_names,
                     cbar=False,
                     square=True,
                     annot_kws={"size": 16})

    ax.tick_params(axis='both', which='major', pad=10)

    title_font = {'family': 'Arial', 'size': 22, 'weight': 'bold'}
    plt.xticks(fontsize=18, fontfamily='Arial', weight='normal')
    plt.yticks(fontsize=18, fontfamily='Arial', weight='normal')
    plt.tick_params(axis='both', which='both', length=0)

    plt.xlabel('True Origin', fontdict=title_font, labelpad=20)
    plt.ylabel('Predicted Origin', fontdict=title_font, labelpad=20)

    plt.tight_layout()

    plt.savefig(f'{selection}_confusion_matrix.tif',
                format='tif', dpi=300, bbox_inches='tight')
    plt.savefig(f'{selection}_confusion_matrix.eps',
                format='eps', dpi=300, bbox_inches='tight')
    plt.show()

    return accuracy >= condition_accuracy and precision >= condition_precision

def main():
    seed = 22604
    hidden_size = 1024
    batch_size = 111
    learning_rate = 0.0005
    print(f"Number: Seed={seed}, Hidden Size={hidden_size}, Batch Size={batch_size}, Learning Rate={learning_rate:.6f}")

    train_and_evaluate(seed, hidden_size, batch_size, learning_rate)

if __name__ == "__main__":
    main()