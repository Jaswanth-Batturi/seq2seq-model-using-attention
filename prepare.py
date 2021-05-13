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
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import json
import pickle

reviews = pd.read_csv("./data/amazon_reviews.csv")
reviews = reviews.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time'], axis=1)
reviews = reviews.dropna()
reviews =  reviews.reset_index(drop=True)
reviews.head()
reviews.shape

contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def clean_text(text, remove_stopwords):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
    
    # Convert words to lower case
    text = text.lower()
    
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'<br >', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = deEmojify(text)
    
    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text

clean_texts = []
for text in reviews.Text:
    clean_texts.append(clean_text(str(text), remove_stopwords=False))
print("Texts are complete.")

clean_summaries = []
for summary in reviews.Summary:
    clean_summaries.append(clean_text(str(summary), remove_stopwords=False))
print("Summaries are complete.")

def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1
                
'''Find the number of times each word was used and the size of the vocabulary'''
word_counts = {}

count_words(word_counts, clean_summaries)
count_words(word_counts, clean_texts)
            
print("Size of Vocabulary:", len(word_counts))

embeddings_index = {}

with open('./data/glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings:', len(embeddings_index))

missing_words = 0
threshold = 0

for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index:
            missing_words += 1
            
missing_ratio = round(missing_words/len(word_counts),4)*100
            
#print("Number of words missing:", missing_words)
#print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))

#dictionary to convert words to integers
vocab_to_int = {} 

value = 0
for word, count in word_counts.items():
    #if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>","<PAD>","<START>", "<EOS>"]   

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

with open('data/vocabulary.pickle', 'wb') as f:
    pickle.dump(int_to_vocab, f)

with open('data/index.pickle', 'wb') as f:
    pickle.dump(vocab_to_int, f)

# usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

# print("Total number of unique words:", len(word_counts))
# print("Number of words we will use:", len(vocab_to_int))
# print("Percent of words we will use: {}%".format(usage_ratio))

embedding_dim = 50
nb_words = len(vocab_to_int)

# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        # If word not in CN, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        #embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Check if value matches len(vocab_to_int)
print(len(word_embedding_matrix))

with open('data/embedding.pickle', 'wb') as f:
    pickle.dump(word_embedding_matrix, f)

def create_lengths(text):
    '''Create a data frame of the sentence lengths from a text'''
    lengths = []
    for sentence in text:
        lengths.append(len(sentence.split()))
    return pd.DataFrame(lengths, columns=['counts'])

# Text analysing
lengths_texts = create_lengths(clean_texts)
print("Texts:")
print(lengths_texts.describe())

# Summary analysing
lengths_summaries = create_lengths(clean_summaries)
print("Summaries:")
print(lengths_summaries.describe())


def convert_to_ints(text, word_count, unk_count, summary, max_length):
    '''Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts'''
    ints = []
    for sentence in text:
        sentence_ints = []
        if summary:
            sentence_ints.append(vocab_to_int["<START>"])
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1


        while not summary and len(sentence_ints) < max_length:
            sentence_ints.append(vocab_to_int["<PAD>"])
        
        if summary and len(sentence_ints) >= max_length:
            sentence_ints[max_length-1] = vocab_to_int["<EOS>"]
        else:
            sentence_ints.append(vocab_to_int["<EOS>"])
        
        while summary and len(sentence_ints) < max_length:
            sentence_ints.append(vocab_to_int["<PAD>"])

        sentence_ints = sentence_ints[:max_length]
        ints.append(np.array(sentence_ints))

    return np.array(ints), word_count, unk_count

# Apply convert_to_ints to clean_summaries and clean_texts
word_count = 0
unk_count = 0
max_len_text = 100
max_len_summary = 10

int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, False, max_len_text)
int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count, True, max_len_summary)

unk_percent = round(unk_count/word_count,4)*100

# print("Total number of words in headlines:", word_count)
# print("Total number of UNKs in headlines:", unk_count)
# print("Percent of words that are UNK: {}%".format(unk_percent))

def create_sizes(tokens):
    '''Create a data frame of the sentence lengths from a text'''
    lengths = []
    for sentence in tokens:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])

lengths_texts = create_sizes(int_texts)
lengths_summaries = create_sizes(int_summaries)

# print("After processing:\n")
# print("Texts:")
# print(lengths_texts.describe())
# print()
# print("Summaries:")
# print(lengths_summaries.describe())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(int_texts, int_summaries, test_size=0.2, random_state=42)

print("Train Size :", len(X_train))
print("Test Size :", len(X_test))

with open("data/data.json",'w') as f:
        js = {}
        js['X_train'] = X_train.tolist()
        js['y_train'] = y_train.tolist()
        js['X_test'] = X_test.tolist()
        js['y_test'] = y_test.tolist()
        json.dump(js,f)

print("Data preparation finished...")