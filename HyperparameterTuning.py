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
    
    # Parameters for tuning
    num_topics_list = [5, 10, 15, 20, 25]  # Different numbers of topics to evaluate
    alpha_list = [0.01, 0.1, 1]  # Different alpha values to evaluate
    
    # DataFrame to store coherence values
    coherence_df = pd.DataFrame(columns=['alpha', 'num_topics', 'coherence_value'])
    
    # Iterate over different combinations of num_topics and alpha
    for num_topics in num_topics_list:
        for alpha in alpha_list:
            print(f"Training model with num_topics={num_topics}, alpha={alpha}")
            
            # Train the model
            lda_model = train_lda_model(corpus, dictionary, num_topics=num_topics, passes=5, chunksize=1000, alpha=alpha, eta='auto')
            coherence_score = compute_coherence(lda_model, texts, dictionary)
            
            # Append results to DataFrame using pd.concat()
            new_row = pd.DataFrame({'alpha': [alpha], 'num_topics': [num_topics], 'coherence_value': [coherence_score]})
            coherence_df = pd.concat([coherence_df, new_row], ignore_index=True)
            print(f'Coherence Score: {coherence_score}')
    
    # Save the coherence values to CSV
    coherence_df.to_csv('/Users/nomantahir/Desktop/ve/venv/coherence_scores.csv', index=False)
    
    # Display the results
    print(coherence_df)

    # Save the LDA model and dictionary for the best configuration (if needed)
    best_config = coherence_df.loc[coherence_df['coherence_value'].idxmax()]
    print(f"Best Configuration: num_topics={best_config['num_topics']}, alpha={best_config['alpha']}")
    lda_model = train_lda_model(corpus, dictionary, num_topics=int(best_config['num_topics']), passes=5, chunksize=1000, alpha=best_config['alpha'], eta='auto')
    lda_model.save('/Users/nomantahir/Desktop/ve/venv/lda_model_best.model')
    dictionary.save('/Users/nomantahir/Desktop/ve/venv/dictionary_best.gensim')
    
    # Print topics for the best model
    print_topics(lda_model)

if __name__ == '__main__':
    mp.freeze_support()  # This is required for Windows
    main()
