import pandas as pd
import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn.functional as F

# THINGS I NEED TO DEFINE: num_words, seq_len, in_features_fc()
######################################
# LOAD DATA

# only first 100 rows for now
'''
def load_data(self):
    text_data = pd.read_csv(r'../proj template/data/representatives-1.csv', nrows=100)
    text_df = pd.DataFrame(text_data, columns=['user_name', 'user_screen_name', 'text'])
    # print(text_df)

    label_table = pd.read_csv(r'../proj template/data/rep_accounts.csv')
    label_df = pd.DataFrame(label_table, columns=['Handle', 'Party'])
    # print(label_df)

    # x values, which are the tweets
    self.x_raw = text_df['text'].values

    def getLabel(tweet):
        row_of_tweet = text_df.loc[text_df['text'] == tweet]
        handle_from_tweet = row_of_tweet['user_screen_name']
        row_from_handle = label_df.loc[label_df['Handle' == handle_from_tweet]]
        party = row_from_handle['Party']
        return party
'''

text_data = pd.read_csv(r'../proj template/data/representatives-1.csv', nrows=100)
text_df = pd.DataFrame(text_data, columns=['user_name', 'user_screen_name', 'text'])
print(text_df)


label_table = pd.read_csv(r'../proj template/data/rep_accounts.csv')
label_df = pd.DataFrame(label_table, columns=['Handle', 'Party'])
# print(label_df)

# x values, which are the tweets
x_raw = text_df['text'].values

def getLabel(tweet):
    row_of_tweet = text_df.loc[text_df['text'] == tweet]
    handle_from_tweet = row_of_tweet['user_screen_name']
    row_from_handle = label_df.loc[label_df['Handle' == handle_from_tweet]]
    party = row_from_handle['Party']
    return party


"""# function to get party label from tweet
def getLabel(tweet):
    row_of_tweet = text_df.loc[text_df['text'] == tweet]
    handle_from_tweet = row_of_tweet['user_screen_name']
    row_from_handle = label_df.loc[label_df['Handle' == handle_from_tweet]]
    party = row_from_handle['Party']
    return party"""

# y values, which are the party labels # doesn't work, maybe need to do for each
# y = getLabel(x_raw)

###############################################
# PREPROCESSING

# clean data
#def clean_text(self):
    # x_raw = [x.lower() for x in x_raw] # all lowercase

# tokenize data
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
def text_tokenization(self):
    self.x_raw = [word_tokenize(x) for x in self.x_raw]

# builds vocab
import nltk
def build_vocabulary(self):
    self.vocabulary = dict()
    fdist = nltk.FreqDist()

    for sentence in self.x_raw:
        for word in sentence:
            fdist[word] += 1

    #num_words = fdist[0]
    common_words = fdist.most_common(self.num_words)

    for idx, word in enumerate(common_words):
        self.vocabulary[word[0]] = (idx+1)

# word token to index representation
def word_to_index(self):
    self.x_tokenized = list()

    for sentence in self.x_raw:
        temp_sent = list()
        for word in sentence:
            if word in self.vocabulary.keys():
                temp_sent.append(self.vocabulary[word])
        self.x_tokenized.append(temp_sent)

# pads to same number of words
def padding_sentences(self):
    pad_idx = 0
    self.x_padded = list()

    for sentence in self.x_tokenized:
        while len(sentence) < self.seq_len:
            sentence.insert(len(sentence), pad_idx)
        self.x_padded.append(sentence)

    self.x_padded = np.array(self.x_padded)

# split into train and test
from sklearn.model_selection import train_test_split
def split_data(self):
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_padded, self.y, test_size=0.25, random_state=42)

###############################
# MODEL

# model uses different kernel sizes on the same sent, max pools, turned into one tensor, then classified
from torch import nn
class TextClassifier(nn.ModuleList):
    def __init__(self, params):
        super(TextClassifier, self).__init__()

        # parameters for preprocessing
        self.seq_len = params.seq_len
        self.num_words = params.num_words
        self.embedding_size = params.embedding_size

        # define dropout
        self.dropout = nn.Dropout(0.25)

        # CNN parameters
        # kernel sizes
        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        self.kernel_4 = 5

        # output size at each convolution
        self.out_size = params.out_size
        # num strides for each convolution
        self.stride = params.stride

        # embedding layer
        self.embedding = nn.Embedding(self.num_words+1, self.embedding_size, padding_idx=0)

        # convolution layers
        self.conv1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
        self.conv2 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)
        self.conv3 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_3, self.stride)
        self.conv4 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_4, self.stride)

        # max pool layers
        self.pool1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool4 = nn.MaxPool1d(self.kernel_4, self.stride)

        # fc layer
        self.fc = nn.Linear(self.in_features_fc(), 1)


# function to calculate input size for linear layer
def in_features_fc(self):
    '''Calculates the number of output features after Convolution + Max pooling

    Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
    Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1

    source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    '''
    # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
    out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
    out_conv_1 = math.floor(out_conv_1)
    out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
    out_pool_1 = math.floor(out_pool_1)

    # Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
    out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
    out_conv_2 = math.floor(out_conv_2)
    out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
    out_pool_2 = math.floor(out_pool_2)

    # Calcualte size of convolved/pooled features for convolution_3/max_pooling_3 features
    out_conv_3 = ((self.embedding_size - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
    out_conv_3 = math.floor(out_conv_3)
    out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
    out_pool_3 = math.floor(out_pool_3)

    # Calcualte size of convolved/pooled features for convolution_4/max_pooling_4 features
    out_conv_4 = ((self.embedding_size - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
    out_conv_4 = math.floor(out_conv_4)
    out_pool_4 = ((out_conv_4 - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
    out_pool_4 = math.floor(out_pool_4)

    # Returns "flattened" vector (input for fully connected layer)
    return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * self.out_size


# forward function
def forward(self, x):
    # tokens going through embedding layer
    x = self.embedding(x)

    # convolution, activation, and pooling
    x1 = self.conv1(x)
    x1 = torch.relu(x1)
    x1 = self.pool1(x1)

    x2 = self.conv2(x1)
    x2 = torch.relu(x2)
    x2 = self.pool2(x2)

    x3 = self.conv3(x2)
    x3 = torch.relu(x3)
    x3 = self.pool3(x3)

    x4 = self.conv4(x3)
    x4 = torch.relu(x4)
    x4 = self.pool4(x4)

    # concatenate into unique vector/ flatten
    union = torch.cat((x1, x2, x3, x4), 2)
    union = union.reshape(union.size(0), -1)

    # fc layer, dropout, and activation
    out = self.fc(union)
    out = self.dropout(out)
    out = torch.sigmoid(out)

    return out.squeeze()

#################
# TRAINING
from torch.utils.data import Dataset, DataLoader

class DatasetMaper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class Run:
    def train(model, data, params):
        # Initialize dataset maper
        train = DatasetMaper(data['x_train', data['y_train']])
        test = DatasetMaper(data['x_test', data['y_test']])

        # Initialize loaders
        loader_train = DataLoader(train, batch_size=params.batch_size)
        loader_test = DataLoader(test, batch_size=params.batch_size)

        # Define optimizer
        optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate)

        # Start training
        for epoch in range(params.epochs):
            # Set model in training model
            model.train()
            predictions = []
            # batch training
            for x_batch, y_batch in loader_train:
                y_batch = y_batch.type(torch.FloatTensor)

                # feed model
                y_pred = model(x_batch)

                # loss
                loss = F.binary_cross_entropy(y_pred, y_batch)

                # clean gradients
                optimizer.zero_grad()

                # calculate gradients
                loss.backward()

                # gradient update
                optimizer.step()

                # save predictions
                predictions += list(y_pred.detach().numpy())

            # Evaluation
            test_predictions = Run.evaluation(model, loader_test)

            # Metrics calculation
            train_accuracy = Run.calculate_accuracy(data['y_train'], predictions)
            test_accuracy = Run.calculate_accuracy(data['y_test', test_predictions])
            print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch+1, loss.item(), train_accuracy, test_accuracy))


    def evaluation(model, loader_test):
        # set model in eval mode
        model.eval()
        predictions = []

        # start eval
        with torch.no_grad():
            for x_batch, y_batch in loader_test:
                y_pred = model(x_batch)
                predictions += list(y_pred.detach().numpy())
            return predictions

    def calculate_accuracy(ground_truth, predictions):
        true_positives = 0
        true_negatives = 0

        # gets freq of true pos and true neg, threshold is 0.5
        threshold = 0.5
        for true, pred in zip(ground_truth, predictions):
            if (pred >= threshold) and (true == 1):
                true_positives += 1
            elif (pred < threshold) and (true == 0):
                true_negatives += 1
            else:
                pass
        return (true_positives+true_negatives) / len(ground_truth)




