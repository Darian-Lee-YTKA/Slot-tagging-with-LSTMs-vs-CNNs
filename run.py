import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

train = pd.read_csv('/Users/darianlee/PycharmProjects/run.py/hw2_train.csv')
test = pd.read_csv("/Users/darianlee/PycharmProjects/run.py/hw2_test.csv")

print("starting")
import spacy





import re
import unicodedata


def remove_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

train["utterances"] = train["utterances"].apply(remove_accents)
test["utterances"] = test["utterances"].apply(remove_accents)

train = train.sample(frac=1, random_state=69).reset_index(drop=True)
import numpy as np

train, val = train_test_split(train, test_size=0.1, random_state=12)
print("print(train.columns)", train.columns)
print("print(val.columns)", val.columns)

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
# define dataset

print("defining the dataset")
class IOBDataset(Dataset):

    def __init__(self, data, token_vocab=None, tag_vocab=None, training=True, test_set = False):
        self.test_set = test_set


        if training:
            self.token_vocab = {'<UNK>': 1, '<PAD>': 0}
            self.tag_vocab = {'<PAD>': 0, '<Test>': 1}
            for utterance in data["utterances"]:
                tokens = utterance.split(" ")
                for token in tokens:
                    if token in self.token_vocab:
                        continue
                    else:
                        self.token_vocab[token] = len(self.token_vocab) # this will just be the index inwhich it was added
            for tag_row in data["IOB Slot tags"]:
                tags = tag_row.split(" ")
                for tag in tags:
                    if tag in self.tag_vocab:
                        continue
                    else:
                        self.tag_vocab[tag] = len(self.tag_vocab)  # this will just be the index inwhich it was added
                        print(self.tag_vocab)

        else:
            assert token_vocab is not None and tag_vocab is not None
            self.token_vocab = token_vocab
            self.tag_vocab = tag_vocab

        self.corpus_token_ids = []
        self.corpus_tag_ids = []

        # Convert sentences and tags to integer IDs during initialization
        new_utterances = []
        for utterance in data["utterances"]:
            token_list = utterance.split(" ")
            token_ids = [self.token_vocab.get(token, self.token_vocab['<UNK>']) for token in token_list]
            self.corpus_token_ids.append(torch.tensor(token_ids))

        if not test_set:
            for tag in data["IOB Slot tags"]:
                tag_list = tag.split(" ")
                tag_ids = [self.tag_vocab[tag] for tag in tag_list]
                self.corpus_tag_ids.append(torch.tensor(tag_ids))
        else:
            # for the test set
            for utterance in data["utterances"]:
                token_list = utterance.split(" ")
                tag_ids = [self.tag_vocab["<Test>"] for tag in token_list] # this is so that we can keep track of the utterance length and padding
                self.corpus_tag_ids.append(torch.tensor(tag_ids))
    def __len__(self):
        return len(self.corpus_token_ids)

    def __getitem__(self, idx):

        return self.corpus_token_ids[idx], self.corpus_tag_ids[idx]

train_dataset = IOBDataset(train, training=True)
val_dataset = IOBDataset(val, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab,
                             training=False)
test_dataset = IOBDataset(test, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab,
                             training=False, test_set = True)

    # collate token_ids and tag_ids to make mini-batches
print("collate_fn")
def collate_fn(batch):

        # batch: [(token_ids, tag_ids), (token_ids, tag_ids), ...]

                # Separate sentences and tags
                token_ids = [item[0] for item in batch]
                tag_ids = [item[1] for item in batch]

                # Pad sequences
                sentences_padded = pad_sequence(token_ids, batch_first=True, padding_value=train_dataset.token_vocab['<PAD>'])
                # sentences_pad.size()  (batch_size, seq_len)
                tags_padded = pad_sequence(tag_ids, batch_first=True, padding_value=train_dataset.tag_vocab['<PAD>'])
                # tags_pad.size()  (batch_size, seq_len)


                if sentences_padded.size() != tags_padded.size():
                    print("it happened in colate: ", sentences_padded, tags_padded)
                return sentences_padded, tags_padded


import gensim.downloader as api
from nltk import word_tokenize
from gensim.models import KeyedVectors

wv1 = api.load('glove-twitter-200')
def create_embedding_matrix(wv1, vocab, embedding_dim):
    embedding_matrix = np.zeros((len(vocab), embedding_dim))

    for word, idx in vocab.items():
        if word in wv1:
            embedding_matrix[idx] = wv1[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    return torch.FloatTensor(embedding_matrix)


embedding_matrix = create_embedding_matrix(wv1, train_dataset.token_vocab, 200)
print("definig the model")


class SeqTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, pretrained_embeddings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings)

        # Convolutional layers with multiple filters
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        # Fully connected layer for tag prediction
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids).transpose(1, 2)  # Switch to (batch_size, embedding_dim, seq_len)

        conv1_out = self.dropout(self.relu(self.conv1(embeddings)))
        conv2_out = self.dropout(self.relu(self.conv2(conv1_out)))  # (batch_size, hidden_dim*2, seq_len)

        # Transpose back to (batch_size, seq_len, hidden_dim*2)
        conv_out = conv2_out.transpose(1, 2)
        outputs = self.fc(conv_out)  # (batch_size, seq_len, tagset_size)
        return outputs


EMBEDDING_DIM = 200
HIDDEN_DIM = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 45
tag_vocab_inv = {id_: tag for tag, id_ in train_dataset.tag_vocab.items()} # for converting back

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
model = SeqTagger(
    vocab_size=len(train_dataset.token_vocab),
    tagset_size=len(train_dataset.tag_vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
pretrained_embeddings = embedding_matrix

)
loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataset.tag_vocab['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = model.to(device)

# Training Loop
best_model_state_F1 = None
best_val_F1 = 0
early_stop = 0
for epoch in range(NUM_EPOCHS):
    if early_stop == 6:
        print("early stopping")
        break
    # Training
    model.train()
    total_train_loss = 0
    for token_ids, tag_ids in train_loader:
        token_ids = token_ids.to(device)
        tag_ids = tag_ids.to(device)

        optimizer.zero_grad()

        outputs = model(token_ids)  # (batch_size, seq_len, tagset_size)



        if outputs.shape[0] > tag_ids.shape[0]:
            print("the batch size was somehow different... ?")
            outputs = outputs[:tag_ids.shape[0]]

        if outputs.shape[1] != tag_ids.shape[1]:
            print("they weren't equal")
            if outputs.shape[1] > tag_ids.shape[1]:
                outputs = outputs[:, :len(tag_ids), :]
            else:
                print("👩🏻‍🦰👩🏻‍🦰👩🏻‍🦰 this shouldnt happen very often 👩🏻‍🦰👩🏻‍🦰👩🏻‍🦰")
                pad_length = tag_ids.shape[1] - outputs.shape[1]
                outputs = torch.nn.functional.pad(outputs, (0, 0, 0, pad_length))


        loss = loss_fn(outputs.view(-1, outputs.shape[-1]), tag_ids.view(-1))
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    # Validation
    model.eval()
    total_val_loss = 0
    all_predictions = []  # This will store all predictions across the batch
    preds_per_utterance = []  # New list to store predictions per utterance
    all_tags = []

    with torch.no_grad():
        for token_ids, tag_ids in val_loader:
            token_ids = token_ids.to(device)
            tag_ids = tag_ids.to(device)

            outputs = model(token_ids)  # (batch_size, seq_len, tagset_size)

            if outputs.shape[0] > tag_ids.shape[0]:
                print("the batch size was somehow different... ?")
                outputs = outputs[:tag_ids.shape[0]]

            if outputs.shape[1] != tag_ids.shape[1]:
                print("they weren't equal")

                if outputs.shape[1] > tag_ids.shape[1]:
                    outputs = outputs[:, :len(tag_ids), :]
                else:
                    print("👩🏻‍🦰👩🏻‍🦰👩🏻‍🦰 this shouldnt happen very often 👩🏻‍🦰👩🏻‍🦰👩🏻‍🦰")
                    pad_length = tag_ids.shape[1] - outputs.shape[1]
                    outputs = torch.nn.functional.pad(outputs, (0, 0, 0, pad_length))

            # batch size and sequence length will be together concat
            outputs = outputs.view(-1, outputs.shape[-1])
            tag_ids = tag_ids.view(-1)

            loss = loss_fn(outputs, tag_ids)
            total_val_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            mask = tag_ids != train_dataset.tag_vocab['<PAD>']

            all_predictions.extend(predictions[mask].tolist())

            batch_preds_per_utterance = []
            for i in range(token_ids.shape[0]):

                utterance_pred = predictions[i * token_ids.shape[1]:(i + 1) * token_ids.shape[1]]
                utterance_pred = utterance_pred[mask[i * token_ids.shape[1]:(i + 1) * token_ids.shape[1]]]
                batch_preds_per_utterance.append(utterance_pred.tolist())

            preds_per_utterance.extend(batch_preds_per_utterance)

            all_tags.extend(tag_ids[mask].tolist())

    train_loss = total_train_loss / len(train_loader)
    val_loss = total_val_loss / len(val_loader)



    f1 = f1_score(all_tags, all_predictions, average='weighted')
    if f1 > best_val_F1:
        early_stop = 0
        print("\n ================================== WE HAVE A WINNER ================================")
        print("\n New best validation F1:", f1)
        best_val_F1 = f1
        best_model_state_F1 = model.state_dict()
    else:
        early_stop += 1
    # Print some sample predictions
    print(preds_per_utterance[:20])  # Show first 10 predictions per utterance
    for pred in preds_per_utterance[:20]:
        tag_names = [tag_vocab_inv[tag_id] for tag_id in pred]
        print(len(tag_names))
        print(tag_names)

    print(f'{epoch = } | train_loss = {train_loss:.3f} | val_loss = {val_loss:.3f} | f1 = {f1:.3f}')

""" # Validation
    model.eval()
    total_val_loss = 0
    all_predictions = []
    predictions_list = []
    all_tags = []

    with torch.no_grad():
        for token_ids, tag_ids in val_loader:
            token_ids = token_ids.to(device)
            tag_ids = tag_ids.to(device)

            outputs = model(token_ids)  # (batch_size, seq_len, tagset_size)

            if outputs.shape[0] > tag_ids.shape[0]:
                print("the batch size was somehow different... ?")
                outputs = outputs[:tag_ids.shape[0]]

            if outputs.shape[1] != tag_ids.shape[1]:
                print("they weren't equal")

                if outputs.shape[1] > tag_ids.shape[1]:
                    outputs = outputs[:, :len(tag_ids), :]
                else:
                    print("👩🏻‍🦰👩🏻‍🦰👩🏻‍🦰 this shouldnt happen very often 👩🏻‍🦰👩🏻‍🦰👩🏻‍🦰")
                    pad_length = tag_ids.shape[1] - outputs.shape[1]
                    outputs = torch.nn.functional.pad(outputs, (0, 0, 0, pad_length))

            outputs = outputs.view(-1, outputs.shape[-1])
            tag_ids = tag_ids.view(-1)

            loss = loss_fn(outputs, tag_ids)
            total_val_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            mask = tag_ids != train_dataset.tag_vocab['<PAD>']

            all_predictions.extend(predictions[mask].tolist())
            predictions_list.append(predictions[mask].tolist())
            all_tags.extend(tag_ids[mask].tolist())



        # compute train and val loss
        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(val_loader)

        # Calculate F1 score
        f1 = f1_score(all_tags, all_predictions, average='weighted')
        print(predictions_list[:10])
        for pred in predictions_list[:10]:
            tag_names = [tag_vocab_inv[tag_id] for tag_id in pred]
            print(len(tag_names))
            print(tag_names)

        print(f'{epoch = } | train_loss = {train_loss:.3f} | val_loss = {val_loss:.3f} | f1 = {f1:.3f}')
"""


#  testing
model.load_state_dict(best_model_state_F1)

print(tag_vocab_inv)
print(train_dataset.tag_vocab)
model.eval()
all_test_tag_strings = []
print("starting test")
with torch.no_grad():
    for token_ids, tag_ids in test_loader:
        token_ids = token_ids.to(device)
        tag_ids = tag_ids.to(device)
        outputs = model(token_ids)
        if outputs.shape[0] > tag_ids.shape[0]:
            print("the batch size was somehow different... ?")
            outputs = outputs[:tag_ids.shape[0]]

        if outputs.shape[1] != tag_ids.shape[1]:
            print("they weren't equal")

            if outputs.shape[1] > tag_ids.shape[1]:
                outputs = outputs[:, :len(tag_ids), :]
            else:
                print("👩🏻‍🦰👩🏻‍🦰👩🏻‍🦰 this shouldnt happen very often 👩🏻‍🦰👩🏻‍🦰👩🏻‍🦰")
                pad_length = tag_ids.shape[1] - outputs.shape[1]
                outputs = torch.nn.functional.pad(outputs, (0, 0, 0, pad_length))


        outputs = outputs.view(-1, outputs.shape[-1])
        tag_ids = tag_ids.view(-1)
        predictions = outputs.argmax(dim=1)


        mask = tag_ids != train_dataset.tag_vocab['<PAD>']

        batch_preds_per_utterance = []
        for i in range(token_ids.shape[0]):

            utterance_pred = predictions[i * token_ids.shape[1]:(i + 1) * token_ids.shape[1]]

            utterance_pred = utterance_pred[mask[i * token_ids.shape[1]:(i + 1) * token_ids.shape[1]]]
            batch_preds_per_utterance.append(utterance_pred.tolist())


        all_test_tag_strings.extend(batch_preds_per_utterance)
final_tag_names = []
for pred in all_test_tag_strings:
        tag_names = [tag_vocab_inv[tag_id] for tag_id in pred]
        print(len(tag_names))
        print(tag_names)
        final_tag_names.append(" ".join(tag_names))

import sys
output_file = sys.argv[3]

test_df = pd.DataFrame({
    "ID": range(1, len(final_tag_names) + 1),
    "IOB Slot tags": final_tag_names
})

test_df.to_csv(output_file, index=False)
