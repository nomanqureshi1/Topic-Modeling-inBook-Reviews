import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset with topics and user ratings
df = pd.read_csv('/Users/nomantahir/Desktop/ve/venv/processed_books_ratings_with_topics.csv')

# Convert 'review/score' to numeric
df['review/score'] = pd.to_numeric(df['review/score'], errors='coerce')

# Drop rows with missing values in 'review/score'
df = df.dropna(subset=['review/score'])

# List of topic columns
topic_columns = [col for col in df.columns if 'topic_' in col]

# Calculate the average review score for each topic
average_ratings = df[topic_columns].multiply(df['review/score'], axis=0).mean()

# Plot the average ratings for each topic
plt.figure(figsize=(10, 6))
average_ratings.plot(kind='bar', color='skyblue')
plt.title('Average Review Scores for Each Topic')
plt.xlabel('Topics')
plt.ylabel('Average Review Score')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plt.savefig('/Users/nomantahir/Desktop/ve/venv/topic_average_rating_plot.png')

# Show the plot
plt.show()

print("Average review score plot saved.")
