import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchtext import data, datasets
import spacy
import numpy as np
import time
import random
import csv
from torchtext.datasets import UDPOS
import spacy_udpipe
from torch.utils.data import IterableDataset
import torchdata
from torchdata.datapipes.iter import IterableWrapper
from torchtext.vocab import GloVe
from nltk.tokenize import word_tokenize
from nltk.corpus import treebank
from nltk.tag.util import untag
from conllu import parse
import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English
from functools import partial
from torch.nn import init
import re

#loading data
train_data = open('en_ewt-ud-train.conllu', 'r', encoding='utf-8').read().strip().split('\n\n')
test_data = open('en_ewt-ud-test.conllu', 'r', encoding='utf-8').read().strip().split('\n\n')
dev_data = open('en_ewt-ud-dev.conllu', 'r', encoding='utf-8').read().strip().split('\n\n')

en_core_web_sm = spacy.load('en_core_web_sm')

TEXT = torchtext.data.Field(tokenize=partial(en_core_web_sm.tokenizer, keep_spacy_tokens=True))
UD_TAGS = torchtext.data.Field(unk_token=None)

# Load the CoNLL-U data
train_data, valid_data, test_data = torchtext.datasets.UDPOS.splits(fields=(('text', TEXT), ('udtags', UD_TAGS)))


# building vocabulary
# using GloVe prebuilt embeddings to initialize (they perform better than ones we initialize)
freq = 5
TEXT.build_vocab(train_data,
                 min_freq=freq,
                 vectors=GloVe(name='6B', dim=100),
                 unk_init=torch.Tensor.normal_)


UD_TAGS.build_vocab(train_data)

BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size = BATCH_SIZE, device = device)

class POS_TAGGER(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = 2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # if bidirectional:
        #     self.fc = nn.Linear(hidden_dim * 2, output_dim)
        # else:
        #     self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, text):        
        embedded = self.dropout(self.embedding(text))
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(outputs)
        return self.fc(hidden)

    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'embedding' not in name:
                nn.init.normal_(param.data, mean=0, std=0.1)

model = POS_TAGGER(len(TEXT.vocab), 100, 128, len(UD_TAGS.vocab), TEXT.vocab.stoi[TEXT.pad_token])
model.init_weights()
model.embedding.weight.data.copy_(TEXT.vocab.vectors)
model.embedding.weight.data[TEXT.vocab.stoi[TEXT.pad_token]] = torch.zeros(100)
optimizer = optim.Adam(model.parameters())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

padding_idx = UD_TAGS.vocab['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index = padding_idx)
criterion = criterion.to(device)

def accuracy_(pred, targ, tag_pad_idx):
    # Step 1: Flatten the inputs
    pred = pred.view(-1, pred.shape[-1])
    targ = targ.view(-1)  
    
    # Step 2: Ignore the pad elements
    non_pad_elements = targ != tag_pad_idx
    
    # Step 3: Get the index of the max probability
    max_preds = pred.argmax(dim=1) 
    
    # Step 4: Count the number of correct predictions
    correct = max_preds[non_pad_elements].eq(targ[non_pad_elements])
    num_correct = correct.sum().item()
    
    # Step 5: Calculate the accuracy
    num_total = non_pad_elements.sum().item()
    acc = num_correct / num_total
    
    return torch.tensor(acc)

def train(model, iterator, optimizer, criterion, padding_idx):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in iterator: 
        text = batch.text
        tags = batch.udtags
        
        # Step 6.1: zero the gradients
        optimizer.zero_grad()
        # Step 6.2: insert the batch of text into the model to get predictions
        predictions = model(text)
        # Step 6.3: reshape the predictions
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        # Step 6.4: calculate loss and accuracy between the predicted tags and actual tags
        loss = criterion(predictions, tags)
        acc = accuracy_(predictions, tags, padding_idx)
        # Step 6.5: call backward to calculate the gradients of the parameters w.r.t. the loss        
        loss.backward()
        # Step 6.6: optimizer step to update the parameters
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def eval_(model, iterator, criterion, padding_idx):
    # model (nn.Module): The model to evaluate.
    # iterator (torchtext.data.Iterator): The data iterator to evaluate on.
    # criterion (nn.Module): The loss function to use for evaluation.
    # padding_idx(int): The padding index for the tags.
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in iterator:
            text, tags = batch.text, batch.udtags
            predictions = model(text)
            batch_size, seq_len, n_tags = predictions.shape
            predictions = predictions.view(-1, n_tags)
            tags = tags.view(-1)
            loss = criterion(predictions, tags)
            acc = accuracy_(predictions, tags, padding_idx)
            epoch_loss += loss.item() * batch.batch_size
            epoch_acc += acc.item() * batch.batch_size
            predicted_labels.extend(torch.argmax(predictions, dim=-1).tolist())
            true_labels.extend(tags.tolist())

        target_names = [UD_TAGS.vocab.itos[i] for i in range(len(UD_TAGS.vocab))]
        report = classification_report(true_labels, predicted_labels, target_names=target_names, output_dict=True)



    return epoch_loss / len(iterator), epoch_acc / len(iterator), report

# N_EPOCHS = 10

# best_valid_loss = float('inf')

# for epoch in range(N_EPOCHS):
#     train_loss, train_acc = train(model, train_iterator, optimizer, criterion, padding_idx)
#     valid_loss, valid_acc, REPORT_ = eval_(model, valid_iterator, criterion, padding_idx)
    
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'trained_model.pt')

    
#     print(f'Epoch: {epoch+1:02}')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
#     print(f'\tVal. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')

# print(REPORT_)

def tag_sentence(model, device, sentence, text_field, tag_field):
    model.eval()
    tokens = sentence.split()
    token_emb = [text_field.vocab.stoi[t] for t in tokens]
    token_tensor = (torch.LongTensor(token_emb)).unsqueeze(-1).to(device)         
    predictions = (model(token_tensor)).argmax(-1)    
    predicted_tags = [tag_field.vocab.itos[t.item()] for t in predictions]
    
    return tokens, predicted_tags

input_sent = input("input sentence: \n")
input_sent = input_sent.lower()
text = input_sent
# Replace multiple spaces with a single space
text = re.sub(r'\s+', ' ', text)

# replace numbers with <NUM>
text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " <NUM> ", text)
text = re.sub(r'\b\d+\w*\b', '<NUM>', text)
text = re.sub(r'\w*\d\w*', '<NUM>', text)

# contractions
text = re.sub(r"can't", "can not", text)
text = re.sub(r"won't", "will not", text)

# hypens and underscore characters at beginning and end of words
text = re.sub(r'(\b|\-|_)(\w+)\-?(\b|\-|_)', r'\2 ', text)
text = re.sub(r'(\b|\-|_)(\w+)\-?(\b|\-|_)', r'\2 ', text)

# Ensure that there's a space between punctuation and words
text = re.sub(r'(\w)([.,!?])', r'\1 \2', text)
text = re.sub(r'([.,!?])(\w)', r'\1 \2', text)

# Replace URLs with <URL>
text = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '<URL>', text)

# Replace hashtags with <HASHTAG>
text = re.sub(r'#\w+', '<HASHTAG>', text)

# Replace mentions with <MENTION>
text = re.sub(r'@\w+', '<MENTION>', text)

# Replace with <PERCENT>
text = re.sub(r'(\d+(\.\d+)?%)', "<PERCENT>", text)


print(input_sent)

tokens, pred_tags = tag_sentence(model, device, input_sent, TEXT, UD_TAGS)
print("Pred. Tag\tToken\n")
for token, pred_tag in zip(tokens, pred_tags):
    print(f"{pred_tag}\t\t{token}")
