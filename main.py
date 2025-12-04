import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()

model = load_model('simplernn_imdb.h5')
model.summary()

reverse_word_index = {value:key for key,value in word_index.items()}

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review ])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


import streamlit as st

st.title("Movie sentiment analysis")
st.write('Enter a movie review to classfiy it as positive or negative')
user_input = st.text_area('Movie Review')
if st.button("Classify"):
    padded_review = preprocess_text(user_input)
    prediction = model.predict(padded_review)
    sentiment = 'Positive' if prediction[0][0]>0.50 else 'Negative'
    st.write(f'The Sentiment is {sentiment}')
else:
    st.write("Pls enter review")

    

