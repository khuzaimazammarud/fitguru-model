from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import json


app = FastAPI()

# Define the Pydantic model for request body
class TextIn(BaseModel):
    text: str

# # Load the dataset and train the model when the application starts
def load_and_train_model():
    file_path = 'fitness_sentences (1).csv'  # Update this to your file path
    data = pd.read_csv(file_path)

    data['IsFitnessRelated'] = data['IsFitnessRelated'].astype(int)
    sentences = data['Sentence'].values
    labels = data['IsFitnessRelated'].values

    training_sentences, testing_sentences, training_labels, testing_labels = train_test_split(
        sentences, labels, test_size=0.2, random_state=42)

    # Text preprocessing parameters
    vocab_size = 10000
    embedding_dim = 16
    max_length = 20
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(training_padded, training_labels, epochs=30, validation_data=(testing_padded, testing_labels), verbose=2)
    
    return model, tokenizer

# Load your model and tokenizer
model, tokenizer = load_and_train_model()

@app.post("/predict")
async def predict(text_in: TextIn):
    sequences = tokenizer.texts_to_sequences([text_in.text])
    padded = pad_sequences(sequences, maxlen=20, padding='post', truncating='post')
    prediction = model.predict(padded)
    threshold = 0.5
    is_related = prediction[0, 0] > threshold
    is_related = bool(is_related)
    score = float(prediction[0, 0])
    return {"score": score, "is_fitness_related": is_related}