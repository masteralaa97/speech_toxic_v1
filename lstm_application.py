
import pandas as pd
import numpy as np
import json
import pickle


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

#load tokenizer
with open('tok.pickle', 'rb') as handle:
    tok = pickle.load(handle)



max_words = 1000
max_len = 260


data={'text':['kill yourself']}
test=pd.DataFrame(data)

r=test.head(20)
b=r["text"]

ts = tok.texts_to_sequences(b)
tss = sequence.pad_sequences(ts,maxlen=max_len)
pred=loaded_model.predict(tss)
Y_t = []
for i in pred:
  Y_t.append(np.argmax(i))

print(Y_t)  

if(Y_t==[2]):
    print("statue : valide")
if(Y_t==[1] or Y_t==[0]):
    print("message innapprorie veuillez reformulez")
