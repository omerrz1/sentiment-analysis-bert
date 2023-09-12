from datasets import load_dataset
from transformers import AutoTokenizer

import numpy as np
import pandas as pd
from transformers import TFAutoModelForSequenceClassification
import sys
import keras
import tensorflow as tf




tokeniser = AutoTokenizer.from_pretrained('distilbert-base-uncased')
imdb = load_dataset("imdb")

data = pd.DataFrame(imdb['train'])
print(data.head())

x = data['text'].tolist()
y = data['label'].tolist()

encoded_x = dict(tokeniser(x,max_length=100,padding=True,truncation=True,return_tensors='tf'))

dataset = tf.data.Dataset.from_tensor_slices((encoded_x,y))
dataset=dataset.batch(16)
dataset = dataset.shuffle(1024).prefetch(16).cache()

model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)




# NOTE ! 
# i used keras optimisers *(legacy for mac) and changed the learing rate to '5e-5'

model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=5e-5),metrics=['accuracy'])

model.fit(dataset, epochs=5)
model.save_pretrained('imdbmodel')

while True:
    inp = input('......')
    tokenized = tokeniser([inp], return_tensors="np", padding="longest")

    outputs = model(tokenized).logits

    classifications = np.argmax(outputs, axis=1)
    print(classifications)



