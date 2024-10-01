import pandas as pd
import numpy as np
from gensim.models import LdaModel
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import multiprocessing as mp

# Function to prepare text data
def prepare_text_data(df):
    nltk.download('stopwords')
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = stopwords.words('english')
    # Assuming 'lemmatized' column contains the processed text
    data = df['lemmatized'].apply(lambda x: x.strip('[]').replace("'", "").split(', '))
    texts = [[word for word in document if word not in stop_words] for document in data]
    return texts

# Function to create a dictionary and corpus
def create_dictionary_corpus(texts):
    dictionary = corpora.Dictionary(texts)
    # Filter out extremes to limit the dictionary size and reduce memory usage
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus

# Function to train LDA model with optimized parameters
def train_lda_model(corpus, dictionary, num_topics=10, passes=5, chunksize=2000, alpha='auto', eta='auto'):
    # Set the number of passes and chunksize for efficient computation
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, 
                         passes=passes, chunksize=chunksize, alpha=alpha, eta=eta, random_state=100)
    return lda_model

# Function to compute coherence score
def compute_coherence(lda_model, texts, dictionary):
    coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()

# Function to print the topics
def print_topics(lda_model):
    topics = lda_model.print_topics(num_words=10)
    for topic in topics:
        print(f"Topic: {topic[0]} \nWords: {topic[1]}")

# Main function to be executed
def main():
    df = pd.read_csv('/Users/nomantahir/Desktop/ve/data/processed_books_ratings.csv')
    
    texts = prepare_text_data(df)
    dictionary, corpus = create_dictionary_corpus(texts)
    
    # Reduce number of passes and chunksize for faster execution
    lda_model = train_lda_model(corpus, dictionary, num_topics=10, passes=5, chunksize=1000, alpha='symmetric', eta='symmetric')
    coherence_score = compute_coherence(lda_model, texts, dictionary)

    # Save the LDA model and dictionary to a specific directory
    lda_model.save('/Users/nomantahir/Desktop/ve/venv/lda_model_optimized.model')
    dictionary.save('/Users/nomantahir/Desktop/ve/venv/dictionary_optimized.gensim')

    print(f'Coherence Score: {coherence_score}')
    print_topics(lda_model)  # Call to print topics

if __name__ == '__main__':
    mp.freeze_support()  # This is required for Windows
    main()
