import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/nomantahir/Desktop/ve/venv/processed_books_ratings_with_topics.csv')

# Select the topic columns
topic_columns = [col for col in df.columns if col.startswith('topic_')]

# Correlation between topics and review score
topic_score_correlation = df[topic_columns + ['review/score']].corr()['review/score'].drop('review/score')

# Correlation between topics and ratings count (popularity)
topic_popularity_correlation = df[topic_columns + ['ratingsCount']].corr()['ratingsCount'].drop('ratingsCount')

# Plotting the correlation between topics and review score and ratings count
plt.figure(figsize=(14, 8))

# Plot for correlation with review score
plt.subplot(2, 1, 1)
topic_score_correlation.plot(kind='bar', color='teal')
plt.title('Correlation Between Topics and Review Score')
plt.xlabel('Topics')
plt.ylabel('Correlation with Review Score')
plt.xticks(rotation=90)

# Plot for correlation with ratings count
plt.subplot(2, 1, 2)
topic_popularity_correlation.plot(kind='bar', color='orange')
plt.title('Correlation Between Topics and Ratings Count (Popularity)')
plt.xlabel('Topics')
plt.ylabel('Correlation with Ratings Count')
plt.xticks(rotation=90)

# Save and show the plot
plt.tight_layout()
plt.savefig('/Users/nomantahir/Desktop/ve/venv/topic_score_ratings_correlation.png')
plt.show()
