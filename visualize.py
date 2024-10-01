import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# Function to load model and dictionary
def load_model_and_dictionary(model_path, dictionary_path):
    lda_model = LdaModel.load(model_path)
    dictionary = Dictionary.load(dictionary_path)
    return lda_model, dictionary

# Function to prepare and visualize LDA model
def visualize_lda_model(lda_model, dictionary, corpus):
    lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_vis, '/Users/nomantahir/Desktop/ve/venv/lda_visualization.html')
    print("Visualization saved as HTML.")

def main():
    # Paths to your model and dictionary
    model_path = '/Users/nomantahir/Desktop/ve/venv/lda_model.model'
    dictionary_path = '/Users/nomantahir/Desktop/ve/venv/dictionary.gensim'

    # Load the model and dictionary
    lda_model, dictionary = load_model_and_dictionary(model_path, dictionary_path)

    # Load processed data for corpus generation
    df = pd.read_csv('/Users/nomantahir/Desktop/ve/data/processed_books_ratings.csv')
    # Assume 'lemmatized' column contains the processed text
    texts = df['lemmatized'].apply(lambda x: x.strip('[]').replace("'", "").split(', ')).tolist()
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Visualize the LDA model
    visualize_lda_model(lda_model, dictionary, corpus)

if __name__ == "__main__":
    main()
