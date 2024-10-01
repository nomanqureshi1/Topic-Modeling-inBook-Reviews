import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Load the word frequencies from the CSV file
df = pd.read_csv('/Users/nomantahir/Desktop/ve/venv/word_frequencies.csv')

# Create a figure
fig = make_subplots(rows=1, cols=1)

# Create a dropdown menu for selecting topics
topic_dropdown = []

# Add scatter traces for each topic
for topic in df['topic'].unique():
    # Filter data for the current topic
    topic_data = df[df['topic'] == topic]
    
    # Create a bar trace for the topic
    trace = go.Bar(
        x=topic_data['frequency'],
        y=topic_data['word'],
        name=f'Topic {topic}',
        orientation='h',
        visible=(topic == df['topic'].unique()[0])  # Only show the first topic initially
    )
    
    # Add the trace to the figure
    fig.add_trace(trace)

    # Add topic to the dropdown menu
    topic_dropdown.append(
        dict(
            label=f'Topic {topic}',
            method='update',
            args=[{'visible': [t == topic for t in df['topic'].unique()]},
                  {'title': f'Significant Words for Topic {topic}'}]
        )
    )

# Create a text annotation next to the dropdown menu
annotation = dict(
    text="Select any topic from 0 to 24 using a ruller in dropbox",  # Text to display
    x=0.05,  # Adjust x position
    y=1.15,  # Adjust y position
    xref="paper",  # Use paper coordinates
    yref="paper",  # Use paper coordinates
    showarrow=False,  # Don't show an arrow
    font=dict(size=16)  # Increase font size
)

# Update layout with adjusted dropdown position and added annotation
fig.update_layout(
    title='Significant Words for Topic 0',
    xaxis_title='Frequency',
    yaxis_title='Words',
    yaxis=dict(tickmode='linear', automargin=True),
    showlegend=False,
    annotations=[annotation],  # Add annotation to the layout
    updatemenus=[dict(
        active=0,
        buttons=topic_dropdown,
        x=0.3,  # Adjusted x position to make space for text
        xanchor='left',
        y=1.15,  # Adjust y position
        yanchor='top',
        direction='down',
        font=dict(size=16),  # Increase font size for dropdown
        pad=dict(l=20, r=20, t=20, b=20),  # Increase padding around dropdown
        borderwidth=3,  # Add border width for better appearance
        bordercolor='black',  # Border color for dropdown box
        bgcolor='lightgrey'  # Background color for the dropdown box
    )]
)

# Set a larger margin for the entire layout
fig.update_layout(
    margin=dict(l=100, r=100, t=100, b=100)
)

# Save the interactive plot
fig.write_html('/Users/nomantahir/Desktop/ve/venv/topic_scroll_interactive_plot.html')

# Show the figure
fig.show()
