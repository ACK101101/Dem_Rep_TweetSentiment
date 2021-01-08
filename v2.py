import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import TweetTokenizer
from sklearn import preprocessing

# take away nrows later
text_data = pd.read_csv(r'../proj template/data/representatives-1.csv')
text_df = pd.DataFrame(text_data, columns=['user_name', 'user_screen_name', 'text'])
#print(text_df)

label_table = pd.read_csv(r'../proj template/data/rep_accounts.csv')
label_df = pd.DataFrame(label_table, columns=['Handle', 'Party'])
#print(label_df)

# x values, which are the tweets
x_raw = text_df['text'].values
#print(x_raw)
#print(x_raw.shape)

## MAIN DATA
# Make new df with everything needed
labeled_tweet_cols = ['Handle', 'Tweet', 'Party']
rows = []
for tweet_idx in range(len(text_df['text'].values)):
    rowData = text_df.loc[tweet_idx, :]
    handle = rowData.loc['user_screen_name']
    tweet = rowData.loc['text']
    partyRow = label_df.loc[label_df['Handle'] == handle, :]
    party = partyRow.loc[:, 'Party'].values
    rows.append({"Handle": handle, "Tweet": tweet, "Party": party[0]})

out_df = pd.DataFrame(rows, columns=labeled_tweet_cols)
#print(out_df)
labeled_tweet_data = out_df.to_csv('C:\\Users\\kalex\\PycharmProjects\\190\\proj v1\\data\\labeled_tweet_data.csv', index=True)

## X AND Y
# split into x and y
x_raw = out_df['Tweet'].values
y_raw = out_df['Party'].values
#print(x_raw.shape)
#print(y_raw.shape)

# clean by lowering everything and converting party to numbers
x_raw = [x.lower() for x in x_raw]
labels = ['D', 'R']
le = preprocessing.LabelEncoder()
targets = le.fit_transform(labels)
y = le.transform(y_raw)
#print(targets)
#print(y)

## TOKENIZE
# tokenize with nltk
tweet_tokenizer = TweetTokenizer()
x_raw = [tweet_tokenizer.tokenize(x) for x in x_raw]
#print(x_raw)


# Preprocessing parameters
seq_len = 240
num_words = 20000
## DICTIONARY
# Builds the vocabulary and keeps the "x" most frequent words
vocabulary = dict()
fdist = nltk.FreqDist()

for sentence in x_raw:
    for word in sentence:
        fdist[word] += 1

common_words = fdist.most_common(num_words)

for idx, word in enumerate(common_words):
    vocabulary[word[0]] = (idx + 1)
#print(vocabulary)

# Transform token based on dictionary index based position
x_tokenized = list()

for sentence in x_raw:
    temp_sentence = list()
    for word in sentence:
        if word in vocabulary.keys():
            temp_sentence.append(vocabulary[word])
    x_tokenized.append(temp_sentence)
#print(x_tokenized)

# Pad to length 240 with index 0
pad_idx = 0
x_padded = list()

for sentence in x_tokenized:
    while len(sentence) < seq_len:
        sentence.insert(len(sentence), pad_idx)
    x_padded.append(sentence)

x_padded = np.array(x_padded)
#print(x_padded)

## TEST TRAIN SPLIT
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_padded, y, test_size=0.25, random_state=42)
#print(x_train)
#print(x_train.shape)
#print(y_train)
#print(y_train.shape)

# Model parameters
embedding_size = 64
out_size = 32
stride = 2
## MODEL
class TextClassifier(nn.ModuleList):

    def __init__(self, seq_len, num_words, embedding_size, out_size, stride):
        super(TextClassifier, self).__init__()

        # Parameters regarding text preprocessing
        self.seq_len = seq_len
        self.num_words = num_words
        self.embedding_size = embedding_size

        # Dropout definition
        self.dropout = nn.Dropout(0.25)

        # CNN parameters definition
        # Kernel sizes
        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        self.kernel_4 = 5

        # Output size for each convolution
        self.out_size = out_size
        # Number of strides for each convolution
        self.stride = stride

        # Embedding layer definition
        self.embedding = nn.Embedding(self.num_words + 1, self.embedding_size, padding_idx=0)

        # Convolution layers definition
        self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
        self.conv_2 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)
        self.conv_3 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_3, self.stride)
        self.conv_4 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_4, self.stride)

        # Max pooling layers definition
        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)

        # Fully connected layer definition
        self.fc = nn.Linear(self.in_features_fc(), 1)

    def in_features_fc(self):
        '''Calculates the number of output features after Convolution + Max pooling

        Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
        Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1

        source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        '''
        # Calculate size of convolved/pooled features for convolution_1/max_pooling_1 features
        out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)

        # Calculate size of convolved/pooled features for convolution_2/max_pooling_2 features
        out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_conv_2 = math.floor(out_conv_2)
        out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_pool_2 = math.floor(out_pool_2)

        # Calculate size of convolved/pooled features for convolution_3/max_pooling_3 features
        out_conv_3 = ((self.embedding_size - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_conv_3 = math.floor(out_conv_3)
        out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_pool_3 = math.floor(out_pool_3)

        # Calculate size of convolved/pooled features for convolution_4/max_pooling_4 features
        out_conv_4 = ((self.embedding_size - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_conv_4 = math.floor(out_conv_4)
        out_pool_4 = ((out_conv_4 - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_pool_4 = math.floor(out_pool_4)

        # Returns "flattened" vector (input for fully connected layer)
        return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * self.out_size

    def forward(self, x):
        # Sequence of tokes is filtered through an embedding layer
        x = self.embedding(x)

        # Convolution layer 1 is applied
        x1 = self.conv_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)

        # Convolution layer 2 is applied
        x2 = self.conv_2(x)
        x2 = torch.relu(x2)
        x2 = self.pool_2(x2)

        # Convolution layer 3 is applied
        x3 = self.conv_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)

        # Convolution layer 4 is applied
        x4 = self.conv_4(x)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)

        # The output of each convolutional layer is concatenated into a unique vector
        union = torch.cat((x1, x2, x3, x4), 2)
        union = union.reshape(union.size(0), -1)

        # The "flattened" vector is passed through a fully connected layer
        out = self.fc(union)
        # Dropout is applied
        out = self.dropout(out)
        # Activation function is applied
        out = torch.sigmoid(out)

        return out.squeeze()

#model = TextClassifier(seq_len=seq_len, num_words=num_words, embedding_size=embedding_size,
                       #out_size=out_size, stride=stride)
#print(model)


# TRAINING
epochs = 10
batch_size = 12
learning_rate = 0.001

class DatasetMaper(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def train(model, x_train, y_train, x_test, y_test, batch_size, learning_rate, epochs):

    # Initialize dataset mapper
    train = DatasetMaper(x_train, y_train)
    test = DatasetMaper(x_test, y_test)

    # Initialize loaders
    loader_train = DataLoader(train, batch_size=batch_size)
    loader_test = DataLoader(test, batch_size=batch_size)

    # Define optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    # Starts training phase
    for epoch in range(epochs):
        # Set model in training model
        model.train()
        predictions = []
        # Starts batch training
        for x_batch, y_batch in loader_train:
            #print(x_batch)
            #print(y_batch)
            x_batch = torch.tensor(x_batch).to(torch.long)
            y_batch = torch.tensor(y_batch).to(torch.float)

            # Feed the model
            y_pred = model(x_batch)

            # Loss calculation
            loss = F.binary_cross_entropy(y_pred, y_batch)

            # Clean gradientes
            optimizer.zero_grad()

            # Gradients calculation
            loss.backward()

            # Gradients update
            optimizer.step()

            # Save predictions
            predictions += list(y_pred.detach().numpy())
            print(list(y_pred.detach().numpy()))

        # Evaluation phase
        test_predictions = evaluation(model, loader_test)

        # Metrics calculation
        train_accuracy, train_true_pos, train_true_neg, train_fal_pos, train_fal_neg = \
            calculate_accuracy(y_train, predictions)
        test_accuracy, test_true_pos, test_true_neg, test_fal_pos, test_fal_neg = \
            calculate_accuracy(y_test, test_predictions)
        print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (
        epoch + 1, loss.item(), train_accuracy, test_accuracy))

        print("True Positives: %d, True Negatives: %d, False Positives: %d, False Negatives: %d" % (
            test_true_pos, test_true_neg, test_fal_pos, test_fal_neg))
        #rows = []
        #labeled_tweet_cols = ['Test Predictions', 'Truth']
        #predictions_df = pd.DataFrame(rows, columns=labeled_tweet_cols)
        #predictions_data = predictions_df.to_csv('C:\\Users\\kalex\\PycharmProjects\\190\\proj v1\\data\\prediction_data.csv', index=True)

def evaluation(model, loader_test):

    # Set the model in evaluation mode
    model.eval()
    predictions = []

    # Start evaluation phase
    with torch.no_grad():
        for x_batch, y_batch in loader_test:
            x_batch = torch.tensor(x_batch).to(torch.long)
            y_batch = torch.tensor(y_batch).to(torch.float)
            y_pred = model(x_batch)
            predictions += list(y_pred.detach().numpy())
    return predictions

def calculate_accuracy(ground_truth, predictions):
    # Metrics calculation
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for true, pred in zip(ground_truth, predictions):
        if (pred >= 0.5) and (true == 1):
            true_positives += 1
        elif (pred < 0.5) and (true == 0):
            true_negatives += 1
        else:
            if (pred >= 0.5) and (true == 0):
                false_positives += 1
            elif (pred < 0.5) and (true == 1):
                false_negatives += 1
    # Return accuracy
    return (true_positives + true_negatives) / len(ground_truth), \
           true_positives, true_negatives, false_positives, false_negatives

#train(model, x_train, y_train, x_test, y_test, batch_size=batch_size, learning_rate=learning_rate, epochs=epochs)
#torch.save(model, 'C:\\Users\\kalex\\PycharmProjects\\190\\proj v1\\model.pt')

model = torch.load('C:\\Users\\kalex\\PycharmProjects\\190\\proj v1\\model.pt')
#print(model)


## PREDICT
def predict(model, tweet, vocabulary):
    x_raw = [x.lower() for x in tweet]
    x_raw = [tweet_tokenizer.tokenize(x) for x in x_raw]

    x_tokenized = list()
    for sentence in x_raw:
        temp_sentence = list()
        for word in sentence:
            if word in vocabulary.keys():
                temp_sentence.append(vocabulary[word])
        x_tokenized.append(temp_sentence)
    #print(x_tokenized)

    # Pad to length 240 with index 0
    pad_idx = 0
    x_padded = list()

    for sentence in x_tokenized:
        while len(sentence) < seq_len:
            sentence.insert(len(sentence), pad_idx)
        x_padded.append(sentence)

    x_padded = np.array(x_padded)
    #print(x_padded)
    x_batch = torch.tensor(x_padded).to(torch.long)
    #print(x_batch)

    model.eval()
    with torch.no_grad():
        pred = model(x_batch)
    prediction = pred.detach().numpy()
    #print(prediction)
    return prediction

#def representative(handle):
dict = {}
for x in vocabulary:
    num = predict(model, x, vocabulary)
    ave = np.max(num)
    dict[x] = ave
    print(ave)
    #(num)
    #print("Word: %s" % (x))
    #print("Predictions: %d" % (num))
    #print(x)

print(dict)

from collections import Counter
#import operator
#key_max = max(dict.items(), key=operator.itemgetter(1))[0]
#key_min = min(dict.items(), key=operator.itemgetter(1))[0]
key_max = sorted(dict, key=dict.get, reverse=True)[:100]
key_min = sorted(dict, key=dict.get, reverse=False)[:100]
print(key_max)
print(key_min)
#t = sorted(dict.iteritems(), key=lambda x:x-x[1])[:100]
k = Counter(dict)
high = k.most_common(100)
low = k.least_common(100)
print("Keys: Values (Highest Weighted [Democrat?])")
for i in high:
    print(i[0], " :", i[1], " ")
print("Keys: Values (Lowest Weighted [Republican?])")
for i in low:
    print(i[0], " :", i[1], " ")



## FIGURES
import matplotlib.pyplot as plt

data = {'True Positives (Democrats)':40259, 'True Negatives (Republicans)':72133,
        'False Positives':10766, 'False Negatives':12169}
classification = list(data.keys())
number = list(data.values())
fig = plt.figure(figsize=(10,5))
plt.bar(classification, number, color='black', width=0.4)
plt.ylabel('Number of Classifications')
plt.xlabel('Type of Classification')
plt.title('4 Layer CNN')
#plt.show()
