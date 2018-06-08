import numpy as np
import pandas as pd
import LogReg as lr
import DataProcessor as dp
import LSTM as lstm
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from DataProcessor import Data
import Trainer
import Evaluator
import HANBAN_Trainer2
import HAN_Evaluator

# Column 1: the ID of the statement ([ID].json).
# Column 2: the label.
# Column 3: the statement.
# Column 4: the subject(s).
# Column 5: the speaker.
# Column 6: the speaker's job title.
# Column 7: the state info.
# Column 8: the party affiliation.
# Column 9-13: the total credit history count, including the current statement.
# 9: barely true counts.
# 10: false counts.
# 11: half true counts.
# 12: mostly true counts.
# 13: pants on fire counts.
# Column 14: the context (venue / location of the speech or statement)


columns = ['id', 'label', 'statement', 'subject', 'speaker', 'speaker_job', 'state', 'party', 'barely_true', 'false', 'half_true', 'mostly_true', 'pants_on_fire', 'context'] # add relplaced record!
raw_train = pd.read_csv("../dataset/liar_dataset/train.csv", sep=';', dtype=str, names=columns)
raw_val = pd.read_csv("../dataset/liar_dataset/valid.csv", sep=';', dtype=str, names=columns)
raw_test = pd.read_csv("../dataset/liar_dataset/test.csv", sep=';', dtype=str, names=columns)
#raw_test = pd.read_csv("../dataset/liar_dataset/test.csv", sep='\t', dtype=str, names=columns)

raw_train['label'] = raw_train['label'].map({'true': 'false', 'mostly-true': 'false', 'half-true': 'false', 'barely-true': 'true', 'false': 'true', 'pants-fire': 'true'})
raw_val['label'] = raw_val['label'].map({'true': 'false', 'mostly-true': 'false', 'half-true': 'false', 'barely-true': 'true', 'false': 'true', 'pants-fire': 'true'})
raw_test['label'] = raw_test['label'].map({'true': 'false', 'mostly-true': 'false', 'half-true': 'false', 'barely-true': 'true', 'false': 'true', 'pants-fire': 'true'})

print(list(raw_train))
print(raw_train['label'].unique())
print('shape:::', raw_train.shape)
print('shape:::', raw_val.shape)
print('shape:::', raw_test.shape)

print("Processing data ...")
train_x = raw_train['statement']
train_y = raw_train['label']
print("first: ", train_y[0], ", ", train_y[1])
val_x = raw_val['statement']
val_y = raw_val['label']
test_x = raw_test['statement']
test_y = raw_test['label']

# train_x, train_sent_lengths, train_seq_lengths = dp.process_hierarchical_data(train_x)
# val_x, val_sent_lengths, val_seq_lengths = dp.process_hierarchical_data(val_x)
# test_x, test_sent_lengths, test_seq_lengths = dp.process_hierarchical_data(test_x)
train_x = dp.process_data(train_x)
val_x = dp.process_data(val_x)
test_x = dp.process_data(test_x)
#print("first: ", train_y[0], ", ", train_y[1])

#translated = [[dp.translate_to_voc(x) for x in y] for y in val_x]
le = preprocessing.LabelEncoder()
train_y = le.fit_transform(train_y)
val_y = le.fit_transform(val_y)
test_y = le.fit_transform(test_y)
ohe = preprocessing.OneHotEncoder(sparse=False)
train_y = ohe.fit_transform(train_y.reshape(-1,1))
val_y = ohe.fit_transform(val_y.reshape(-1,1))
test_y = ohe.fit_transform(test_y.reshape(-1,1))
print("first: ", train_y[0], ", ", train_y[1])
''' possibe batch sizes: 1, 3, 7, 9, 21, 63, 163, 489, 1141, 1467, 3423, 10269 '''

#data = Data2((train_x, train_y, train_sent_lengths, train_seq_lengths), (val_x, val_y, val_sent_lengths, val_seq_lengths), (test_x, test_y, test_sent_lengths, test_seq_lengths), batch_size=163, fixed_padding=True)
data = Data(train_set_x=train_x, train_set_y=train_y, val_set_x=val_x, val_set_y=val_y, test_set_x=test_x, test_set_y=test_y, batch_size=163, fixed_padding=True)
#data = Data(train_set_x=train_x, train_set_y=train_y, val_set_x=val_x, val_set_y=val_y, test_set_x=test_x, test_set_y=test_y, batch_size=163, max_seq_length=30, fixed_padding=True)
voc, embds = dp.get_wordvec_dict()
#lstm.LSTM(data, embds)
print(data.train_set_x.shape)
#Trainer2.run(data, embds)
#print("seq length: ", data.seq_lengths_test[0])
#print("seq length: ", data.seq_lengths_test[703])
#x, y, l = data.fetch_test()
#print("shape: ", np.shape(x))
#Evaluator.run(data, embds)
#HAN_Evaluator.run_std(data, embds)
#HAN_Evaluator.get_confusion(data, embds)
#HAN_Evaluator.t_test(data, embds)

#HAN_Trainer2.run(data, embds)
#Trainer2.run(data, embds)

Evaluator.t_test(data, embds)
