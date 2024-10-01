import pandas as pd
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# Load the LDA model and dictionary (adjust paths based on your files)
lda_model = LdaModel.load('/Users/nomantahir/Desktop/ve/venv/lda_model_best.model')
dictionary = Dictionary.load('/Users/nomantahir/Desktop/ve/venv/dictionary_best.gensim')

# Define the number of top words to extract per topic
num_words = 30

# Prepare a list to hold the data for the CSV
data = []

# Extract topics and word frequencies
for topic_id in range(lda_model.num_topics):
    # Get the top words and their probabilities for this topic
    top_words = lda_model.show_topic(topic_id, num_words)
    for word, prob in top_words:
        data.append({
            'topic': topic_id,
            'word': word,
            'frequency': prob
        })

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_output_path = '/Users/nomantahir/Desktop/ve/venv/word_frequencies.csv'
df.to_csv(csv_output_path, index=False)

print(f"Word frequencies saved to {csv_output_path}")
