#step1: import libraries and load model
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}
model = load_model('simple_rnn_imdb.h5')

def decode_review(review):
    return ''.join([reverse_word_index.get( i-3,'?') for i in review])
def preprocess_text(text):
    words = text.lower()
    words = re.sub(r'[^\w\s]','',words)
    words = words.split()
    encoded_review = [word_index.get(i,2)+3 for i in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_review = preprocess_text(review)
    prediction = model.predict(preprocessed_review)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment,prediction[0][0]

#streamlit app
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

#user input 
user_input = st.text_area('Movie Review')
if st.button('Classify'): #if user clicks this button
    processed_input = preprocess_text(user_input)
    #make prediction
    prediction = model.predict(processed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    #Display the result
    score = round(float(prediction[0][0]), 2)
    st.write(f"Prediction Score: {score}")

    st.write(f"Sentiment: {sentiment}")
   
else:
    st.write("Please enter a review.")


