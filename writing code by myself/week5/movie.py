import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import BertTokenizer
import urllib.request

# Function to download the dataset
def download_nsmc_data():
    url_train = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
    url_test = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"
    os.makedirs('./data', exist_ok=True)
    urllib.request.urlretrieve(url_train, './data/ratings_train.txt')
    urllib.request.urlretrieve(url_test, './data/ratings_test.txt')

# Download the dataset
download_nsmc_data()

# Check for device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("PyTorch version:[%s]."%(torch.__version__))
print("device:[%s]."%(device))

# Load NSMC dataset
class NSMCDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        self.data = pd.read_csv(file_path, sep='\t').dropna()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['document']
        label = self.data.iloc[idx]['label']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Create datasets
train_dataset = NSMCDataset('./data/ratings_train.txt', tokenizer)
test_dataset = NSMCDataset('./data/ratings_test.txt', tokenizer)

# Create DataLoader
BATCH_SIZE = 256
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

# Define the RNN model with embedding
class RecurrentNeuralNetworkClass(nn.Module):
    def __init__(self, name='rnn', vocab_size=30522, emb_dim=128, hdim=256, ydim=2, n_layer=3):
        super(RecurrentNeuralNetworkClass, self).__init__()
        self.name = name
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hdim = hdim
        self.ydim = ydim
        self.n_layer = n_layer
        
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.rnn = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hdim, num_layers=self.n_layer, batch_first=True)
        self.lin = nn.Linear(self.hdim, self.ydim)
    
    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.n_layer, x.size(0), self.hdim).to(device)
        c0 = torch.zeros(self.n_layer, x.size(0), self.hdim).to(device)
        rnn_out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.lin(rnn_out[:, -1, :])
        return out

R = RecurrentNeuralNetworkClass(name='rnn', vocab_size=tokenizer.vocab_size, emb_dim=128, hdim=256, ydim=2, n_layer=2).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(R.parameters(), lr=1e-3)

# Evaluation function
def func_eval(model, data_iter, device):
    with torch.no_grad():
        n_total, n_correct = 0, 0
        model.eval()
        for batch in data_iter:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            _, preds = torch.max(outputs, 1)
            n_correct += (preds == labels).sum().item()
            n_total += input_ids.size(0)
        val_accr = (n_correct / n_total)
        model.train()
    return val_accr

# Training the model
EPOCHS = 5
log_file = open("training_log.txt", "w")
for epoch in range(EPOCHS):
    R.train()
    loss_val_sum = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = R(input_ids)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        loss_val_sum += loss.item()
    
    loss_val_avg = loss_val_sum / len(train_loader)
    train_accr = func_eval(R, train_loader, device)
    test_accr = func_eval(R, test_loader, device)
    log_file.write(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss_val_avg:.4f}, Train Acc: {train_accr:.4f}, Test Acc: {test_accr:.4f}\n")
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss_val_avg:.4f}, Train Acc: {train_accr:.4f}, Test Acc: {test_accr:.4f}")

log_file.close()

# Log test results with random samples
n_sample = 25
sample_indices = np.random.choice(len(test_dataset), n_sample, replace=False)
test_samples = [test_dataset[i] for i in sample_indices]
test_x = torch.stack([sample['input_ids'] for sample in test_samples]).to(device)
test_y = torch.tensor([sample['labels'] for sample in test_samples]).to(device)

with torch.no_grad():
    R.eval()
    y_pred = R(test_x)
    y_pred = y_pred.argmax(axis=1)

with open("test_results.txt", "w") as result_file:
    for idx in range(n_sample):
        result_file.write(f"Pred: {y_pred[idx].item()}, Label: {test_y[idx].item()}, Text: {tokenizer.decode(test_x[idx])}\n")
