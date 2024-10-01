import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset with topics and reviews
df = pd.read_csv('/Users/nomantahir/Desktop/ve/venv/processed_books_ratings_with_topics.csv')

# Initialize VADER sentiment analyzer
nltk.download('vader_lexicon')  # Download the VADER lexicon
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis on the reviews
df['sentiment'] = df['review/text'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Calculate the average sentiment score for each topic
topic_columns = [f'topic_{i}' for i in range(25)]
average_sentiment_by_topic = df[topic_columns + ['sentiment']].groupby(topic_columns).mean()

# Visualize the relationship between sentiment and topics using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[topic_columns + ['sentiment']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation between Topics and Sentiment')
plt.savefig('/Users/nomantahir/Desktop/ve/venv/correlation_topics_sentiment_heatmap.png')
plt.show()

# Correlate topics with ratings
df['review/score'] = pd.to_numeric(df['review/score'], errors='coerce')

# Calculate correlation between topics and ratings
correlation_matrix = df[topic_columns + ['review/score']].corr()

# Visualize the relationship between topics and ratings
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu')
plt.title('Correlation between Topics and Review Scores')
plt.savefig('/Users/nomantahir/Desktop/ve/venv/correlation_topics_ratings_heatmap.png')
plt.show()

# Save the correlation matrix to CSV
correlation_matrix.to_csv('/Users/nomantahir/Desktop/ve/venv/topic_sentiment_correlation.csv')

print("Sentiment and topic correlation analysis completed. Heatmaps generated and saved.")
