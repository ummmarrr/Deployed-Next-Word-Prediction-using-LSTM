# app/model.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def load_model_from_file(model_path):
    # Load the trained Keras model
    model = load_model(model_path)
    
    # Load tokenizer and max_sequence_len if saved separately
    tokenizer_path = model_path.replace('.h5', 'tokenizer.pickle')
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    max_sequence_len = model.input_shape[1]
    
    return model, tokenizer, max_sequence_len

def predict_next_word(model, tokenizer, max_sequence_len, text):
    # Tokenize the input text
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Pad the sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    # Predict probabilities for the next word
    predicted = model.predict(token_list, verbose=0)
    
    # Get the index with the highest probability
    predicted_index = predicted.argmax(axis=-1)[0]
    
    # Convert index back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    
    return "N/A"
