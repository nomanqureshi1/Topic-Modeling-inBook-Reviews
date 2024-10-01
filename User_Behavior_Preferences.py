import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset with topics and processed book ratings
df = pd.read_csv('/Users/nomantahir/Desktop/ve/venv/processed_books_ratings_with_topics.csv')

# Load the topic-sentiment correlation file (if useful for further analysis)
sentiment_correlation = pd.read_csv('/Users/nomantahir/Desktop/ve/venv/topic_sentiment_correlation.csv')

# Load the topic labels (for better visualization)
topic_labels_df = pd.read_csv('/Users/nomantahir/Desktop/ve/venv/topic_words_labels.csv')

# Fill missing 'description' values with an empty string for safety
df['description'] = df['description'].fillna('')

# Adding a new column 'review_length' based on the length of description text
df['review_length'] = df['description'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

# Define a threshold for detailed vs brief reviews
review_threshold = df['review_length'].median()

# Categorize reviews into 'detailed' and 'brief' based on the threshold
df['review_type'] = np.where(df['review_length'] > review_threshold, 'detailed', 'brief')

# Ensure we're only selecting the last 25 topic columns, which are numeric
topic_columns = [col for col in df.columns if col.startswith('topic_')]

# Add topic labels from 'Unnamed: 0' column in the topic_labels_df file
topic_labels = topic_labels_df['Unnamed: 0'].tolist()

# ------------------- Part 1: Detailed vs Brief Review Analysis ------------------- #

# Average topic proportions for detailed and brief reviews
detailed_reviews = df[df['review_type'] == 'detailed'][topic_columns].mean()
brief_reviews = df[df['review_type'] == 'brief'][topic_columns].mean()

# Plotting comparison of topic prevalence in detailed vs brief reviews
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(topic_columns))

plt.bar(index, detailed_reviews, bar_width, label='Detailed Reviews', alpha=0.7, color='b')
plt.bar(index + bar_width, brief_reviews, bar_width, label='Brief Reviews', alpha=0.7, color='g')

plt.xlabel('Topics')
plt.ylabel('Average Topic Proportion')
plt.title('Topic Prevalence in Detailed vs Brief Reviews')
plt.xticks(index + bar_width / 2, topic_labels, rotation=90)
plt.legend()

# Save and show the plot
plt.tight_layout()
plt.savefig('/Users/nomantahir/Desktop/ve/venv/topic_detailed_vs_brief_with_labels.png')
plt.show()

# ------------------- Part 2: Trends in Topics Over Time ------------------- #

# Assuming 'publishedDate' or another column represents review time
df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce')
df['review_year'] = df['publishedDate'].dt.year

# Filter out unreasonable years (for example, keeping only years after 1900)
df = df[df['review_year'] >= 1900]

# Filter out rows with missing years
df = df.dropna(subset=['review_year'])

# Calculate the mean topic proportions by year
topic_trends = df.groupby('review_year').mean()[topic_columns]

# Plot heatmap for topic trends over time
plt.figure(figsize=(12, 8))
sns.heatmap(topic_trends.T, cmap='coolwarm', cbar_kws={'label': 'Average Topic Proportion'}, yticklabels=topic_labels)
plt.title('Topic Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Topics')

# Save and show the heatmap
plt.tight_layout()
plt.savefig('/Users/nomantahir/Desktop/ve/venv/topic_trends_over_time_with_labels_filtered.png')
plt.show()
