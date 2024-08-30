# Import necessary modules and functions from NLTK for text processing and language modeling
from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten
import nltk
from nltk.lm.models import MLE

# Download NLTK's punkt
nltk.download('punkt')

# Define two sample sentences as lists of words
text = [['I', 'need', 'to', 'book', 'ticket', 'to', 'Australia'],
        ['I', 'want', 'to', 'read', 'a', 'book', 'of', 'Shakespeare']]

# Create bigrams from the first sentence
list(bigrams(text[0]))

# Create n-grams (in this case, trigrams) from the second sentence
list(ngrams(text[1], n=3))

# Import the os module to work with file paths
import os

# Get the current working directory
cwd = os.getcwd()
print(cwd)


import pandas as pd

# Read a CSV file ('realdonaldtrump.csv') into a DataFrame
df = pd.read_csv(cwd + '/realdonaldtrump.csv')

# Display the first few rows of the DataFrame
df.head()

# Tokenize the 'content' column of the DataFrame into a list of sentences
from nltk import word_tokenize, sent_tokenize
trump_corpus = df['content'].apply(nltk.word_tokenize).tolist()

# Import a function for preparing data for language modeling
from nltk.lm.preprocessing import padded_everygram_pipeline

# Preprocess the tokenized text for 3-grams language modeling
n = 3
train_data, padded_sents = padded_everygram_pipeline(n, trump_corpus)

# Import a detokenizer to convert tokens back into sentences
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Create an MLE (Maximum Likelihood Estimation) language model with n=3
trump_model = MLE(n)

# Fit the model on the training data
trump_model.fit(train_data, padded_sents)

# Define a detokenizer function to convert tokens back into sentences
detokenize = TreebankWordDetokenizer().detokenize

# Function to generate sentences using the trained language model
def generate_sent(model, num_words, random_seed=42):
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)

#Generate sentences using the trump model with different random seeds
print(generate_sent(trump_model, num_words=20, random_seed=42))
print(generate_sent(trump_model, num_words=10, random_seed=0))
print(generate_sent(trump_model, num_words=10, random_seed=55))
