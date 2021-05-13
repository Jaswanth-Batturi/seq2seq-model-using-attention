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
Model([encoder_inputs, decoder_inputs], decoder_outputs).load_weights('model/weights.h5')  

# encoder inference
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])



# decoder inference
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_len_text,latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

#attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat)

# Final decoder model
decoder_model = Model(
[decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
[decoder_outputs2] + [state_h2, state_c2])

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Choose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = vocab_to_int['<START>']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq, e_out, e_h, e_c])

        # Sample a token
        #print(output_tokens)
        #print(output_tokens[0, -1, :])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = int_to_vocab[sampled_token_index]

        if (sampled_token!='<EOS>'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == '<EOS>' or len(decoded_sentence.split()) >= (max_len_summary-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

def seq2summary(input_seq):
    newString=''
    for i in input_seq:
      if((i!=0 and i!=vocab_to_int['<START>']) and i!=vocab_to_int['<EOS>']):
        newString=newString+int_to_vocab[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
      if(i!=0):
        newString=newString+int_to_vocab[i]+' '
    return newString

for i in range(0,5):
    print(i+1,"Review:",seq2text(X_test[i]))
    print("Original summary:",seq2summary(y_test[i]))
    print("Predicted summary:",decode_sequence(X_test[i].reshape(1,max_len_text)))
    print("\n")
