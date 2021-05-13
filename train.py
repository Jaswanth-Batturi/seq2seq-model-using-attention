import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from attention import AttentionLayer
import re
import json
import pickle
from matplotlib import pyplot 

with open('data/embedding.pickle', 'rb') as f:
    word_embedding_matrix = pickle.load(f)

with open('data/vocabulary.pickle', 'rb') as f:
    int_to_vocab = pickle.load(f)

with open('data/index.pickle', 'rb') as f:
    vocab_to_int = pickle.load(f)


with open("data/data.json",'r') as f:
        js = json.load(f)
        X_train = np.asarray(js['X_train'], dtype=np.float32)
        y_train = np.asarray(js['y_train'], dtype=np.float32)
        X_test = np.asarray(js['X_test'], dtype=np.float32)
        y_test = np.asarray(js['y_test'], dtype=np.float32)

latent_dim = 50
max_len_text = 100
max_len_summary = 10

# Encoder 
encoder_inputs = Input(shape=(max_len_text,)) 
enc_emb = Embedding(len(word_embedding_matrix), latent_dim,trainable=True, weights=[word_embedding_matrix])(encoder_inputs) 

#LSTM 1
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True) 
encoder_outputs, state_h, state_c = encoder_lstm1(enc_emb) 

#LSTM 2 
# encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True) 
# encoder_output, state_h, state_c = encoder_lstm2(encoder_output1) 

# #LSTM 3 
# encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True) 
# encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2) 

# Set up the decoder. 
decoder_inputs = Input(shape=(None,)) 
dec_emb_layer = Embedding(len(word_embedding_matrix), latent_dim,trainable=True,  weights=[word_embedding_matrix]) 
dec_emb = dec_emb_layer(decoder_inputs) 

#LSTM using encoder_states as initial state
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True) 
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c]) 

#Attention Layer
attn_layer = AttentionLayer(name='attention_layer') 
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs]) 

# Concat attention output and decoder LSTM output 
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#Dense layer
decoder_dense = TimeDistributed(Dense(len(word_embedding_matrix), activation='softmax')) 
decoder_outputs = decoder_dense(decoder_concat_input) 

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 
model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

history=model.fit([X_train,y_train[:,:-1]], y_train.reshape(y_train.shape[0],y_train.shape[1], 1)[:,1:] ,epochs=1, batch_size=64, callbacks=[es], validation_data=([X_test,y_test[:,:-1]], y_test.reshape(y_test.shape[0],y_test.shape[1], 1)[:,1:]))

model.save_weights('model/weights.h5')

print("model saved...")