import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Load the dataset with topics and genres
df = pd.read_csv('/Users/nomantahir/Desktop/ve/venv/processed_books_ratings_with_topics.csv')

# Calculate the average topic proportions for each genre
topic_columns = [f'topic_{i}' for i in range(25)]
df_genres = df[['categories'] + topic_columns]

# Group by genres and calculate the mean for each topic
genre_topic_avg = df_genres.groupby('categories').mean().reset_index()

# Create a dropdown menu for selecting genres
fig = make_subplots(rows=1, cols=1)

# Add bar traces for each genre dynamically
for genre in genre_topic_avg['categories'].unique():
    genre_data = genre_topic_avg[genre_topic_avg['categories'] == genre]
    
    trace = go.Bar(
        x=topic_columns,
        y=genre_data[topic_columns].values.flatten(),
        name=genre,
        visible=(genre == genre_topic_avg['categories'].unique()[0])  # Only show the first genre initially
    )
    fig.add_trace(trace)

# Create dropdown options for each genre
dropdown_buttons = [
    dict(
        args=[{"visible": [g == genre for g in genre_topic_avg['categories'].unique()]}],
        label=genre,
        method="update"
    ) for genre in genre_topic_avg['categories'].unique()
]

# Update layout for the dropdown and chart settings
fig.update_layout(
    title="Topic Proportions Across Book Genres",
    xaxis_title="Topics",
    yaxis_title="Average Topic Proportion",
    yaxis=dict(tickmode="linear"),
    updatemenus=[go.layout.Updatemenu(
        active=0,
        buttons=dropdown_buttons,
        x=0.1,
        xanchor="left",
        y=1.15,
        yanchor="top",
        direction="down",
        showactive=True,
        pad={"r": 10, "t": 10},
    )]
)

# Show the figure
fig.show()

# Save the interactive plot
fig.write_html('/Users/nomantahir/Desktop/ve/venv/topic_genre_interactive_barplot.html')
