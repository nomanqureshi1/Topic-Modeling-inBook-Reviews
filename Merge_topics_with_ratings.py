import pandas as pd
from gensim.models.ldamodel import LdaModel

# Load the trained LDA model
lda_model = LdaModel.load('/Users/nomantahir/Desktop/ve/venv/lda_model_best.model')

# Load the dataset (assuming it has user ratings and some text to infer topic proportions)
df = pd.read_csv('/Users/nomantahir/Desktop/ve/data/processed_books_ratings.csv')

# Assuming 'texts' is the column with book texts to infer topic proportions
texts = df['lemmatized'].apply(lambda x: x.strip('[]').replace("'", "").split(', '))

# Get the topic distribution for each document
corpus = [lda_model.id2word.doc2bow(text) for text in texts]
topic_proportions = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]

# Convert topic proportions to a DataFrame
topics_df = pd.DataFrame([[prob for _, prob in doc] for doc in topic_proportions],
                         columns=[f'topic_{i}' for i in range(lda_model.num_topics)])

# Merge the topic proportions with user ratings
df_with_topics = pd.concat([df, topics_df], axis=1)

# Save the merged DataFrame
df_with_topics.to_csv('/Users/nomantahir/Desktop/ve/venv/processed_books_ratings_with_topics.csv', index=False)

print("Topic proportions merged with user ratings and saved.")
