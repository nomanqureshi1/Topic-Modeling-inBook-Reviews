import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pandas as pd

# Function to load model and dictionary
def load_model_and_dictionary(model_path, dictionary_path):
    lda_model = LdaModel.load(model_path)
    dictionary = Dictionary.load(dictionary_path)
    return lda_model, dictionary

# Function to prepare text data
def prepare_text_data(df, column_name):
    # Example to prepare text data from a dataframe column
    texts = df[column_name].apply(lambda x: x.strip('[]').replace("'", "").split(', '))
    return texts

# Function to visualize LDA model
def visualize_lda_model(lda_model, dictionary, texts):
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_vis, '/Users/nomantahir/Desktop/ve/venv/lda_visualization.html')

def main():
    # Paths to your model and dictionary
    model_path = '/Users/nomantahir/Desktop/ve/venv/lda_model.model'
    dictionary_path = '/Users/nomantahir/Desktop/ve/venv/dictionary.gensim'

    # Load the model and dictionary
    lda_model, dictionary = load_model_and_dictionary(model_path, dictionary_path)

    # Load your data
    df = pd.read_csv('/Users/nomantahir/Desktop/ve/data/processed_books_ratings.csv')
    texts = prepare_text_data(df, 'lemmatized')  # Adjust 'lemmatized' if your column name is different

    # Visualize the LDA model
    visualize_lda_model(lda_model, dictionary, texts)

if __name__ == "__main__":
    main()
