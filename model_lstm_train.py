import pandas as pd
import numpy as np
import os
import io
import json
import pickle


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical


import warnings
warnings.filterwarnings('ignore')

def load_data(path):
    df = pd.read_csv(path)
    df.drop(['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither'],axis=1,inplace=True)
    df = df.rename(index=str, columns={"class": "label", "tweet": "tweet"})
    return df


def RNN(max_words,max_len):
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(3,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

#def parametters(epoch,class_weight,max_words,max_len):
    #epochs=epochs
    #class_weight=class_weight
    #max_words=max_words
    #max_len=max_len
    #return epochs,class_weight,max_words,max_len


df=load_data("labeled_data.csv")

X = df.tweet
Y = df.label
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = to_categorical(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)

#peut etre a faire dans une fonction a check

max_words = 1000
max_len = 260

tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

# saving
with open('tok.pickle', 'wb') as handle:
    pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)



model = RNN(max_words,max_len)
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.01),metrics=['accuracy'])

class_weight = {0: 50.,
                1: 1.,
                2: 20.}


model.fit(sequences_matrix,Y_train,batch_size=128,epochs=15, class_weight=class_weight)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")



