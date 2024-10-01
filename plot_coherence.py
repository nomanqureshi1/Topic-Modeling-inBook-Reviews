import pandas as pd
import matplotlib.pyplot as plt

# Load the coherence scores from the CSV file
csv_file_path = '/Users/nomantahir/Desktop/ve/venv/coherence_scores.csv'  # Path to your CSV file
coherence_df = pd.read_csv(csv_file_path)

# Plotting the coherence values
def plot_coherence(coherence_df):
    plt.figure(figsize=(10, 6))
    for alpha in coherence_df['alpha'].unique():
        df = coherence_df[coherence_df['alpha'] == alpha]
        plt.plot(df['num_topics'], df['coherence_value'], marker='o', label=f'alpha={alpha}')

    plt.title('Coherence Score for Various Topic and Alpha Values')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Value')
    plt.legend(title='Alpha')
    plt.grid(True)
    plt.savefig('/Users/nomantahir/Desktop/ve/venv/coherence_plot.png')  # Save the plot to this path
    plt.show()

# Call the plot function
plot_coherence(coherence_df)
