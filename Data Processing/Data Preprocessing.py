import pandas as pd
import spacy
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
import string
from tqdm import tqdm  # Import tqdm

# Load datasets
books_data_path = '/Users/nomantahir/Desktop/ve/data/books_data.csv'
ratings_data_path = '/Users/nomantahir/Desktop/ve/data/books_rating.csv'

books_data = pd.read_csv(books_data_path)
ratings_data = pd.read_csv(ratings_data_path)

# Print column names to verify
print("Columns in books_data:", books_data.columns)
print("Columns in ratings_data:", ratings_data.columns)

# Normalize titles and clean data
books_data['Title'] = books_data['Title'].str.strip().str.lower()
ratings_data['Title'] = ratings_data['Title'].str.strip().str.lower()

books_data.drop_duplicates(subset=['Title'], inplace=True)
ratings_data.drop_duplicates(subset=['Title'], inplace=True)

books_data.dropna(subset=['Title'], inplace=True)
ratings_data.dropna(subset=['Title'], inplace=True)

# Merge both datasets on the 'Title' column
full_data = pd.merge(books_data, ratings_data, on='Title', how='inner')

# Initialize NLTK and Spacy
nltk.download('stopwords', quiet=True)
stop_words = stopwords.words('english') + list(string.punctuation)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Preprocessing functions
def tokenize(text):
    return simple_preprocess(text, deacc=True)  # deacc=True removes punctuations

def remove_stopwords(texts):
    return [[word for word in doc if word not in stop_words] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in tqdm(texts, desc='Lemmatizing', unit='sentences'):  # Add progress bar for lemmatization
        doc = nlp(" ".join(sent))  # Re-join words to process with Spacy
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Apply tokenization, stopwords removal, and lemmatization if 'review/text' column exists
if 'review/text' in full_data.columns:
    # Apply tokenization
    full_data['tokens'] = full_data['review/text'].apply(tokenize)
    
    # Apply stopwords removal
    full_data['tokens_nostops'] = full_data['tokens'].apply(lambda x: remove_stopwords([x])[0])
    
    # Apply lemmatization with progress bar
    full_data['lemmatized'] = lemmatization(full_data['tokens_nostops'])
    
    # Display the first few entries of processed text
    print(full_data['lemmatized'].head())
else:
    print("No 'review/text' column found in merged data.")

print(full_data.head(10))

output_path = '/Users/nomantahir/Desktop/ve/data/processed_books_ratings.csv'
full_data.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")

