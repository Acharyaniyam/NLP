import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyzeSentiment(text):
    sentiment = analyzer.polarity_scores(text)
    compound = sentiment['compound']

    if compound >= 0.05:
        sentiment_class = 'positive'
    elif compound <= -0.05:
        sentiment_class = 'negative'
    else:
        sentiment_class = 'neutral'
    
    return sentiment, sentiment_class

# Function to process JSON data
def process_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.json_normalize(data)
    
    # Convert timestamp to datetime
    df['post_date'] = pd.to_datetime(df['post_date'])

    # Analyze sentiment
    df['sentiment'], df['sentiment_class'] = zip(*df['body_text'].apply(analyzeSentiment))
    
    return df

# Process each subreddit's posts
subreddits = ['bodyweightfitness', 'fitness', 'health', 'keto', 'loseit', 'medicine', 'mentalhealth', 'mentalhealthsupport', 'nutrition']

for subreddit in subreddits:
    print(subreddit)
    post_file = f"{subreddit}/{subreddit}_posts.json"
    posts_df = process_json(post_file)
    
    # Group by time interval (e.g., monthly) and calculate sentiment
    posts_sentiment_over_time = posts_df.groupby([pd.Grouper(key='post_date', freq='M'), 'sentiment_class']).size().unstack().fillna(0)

    # Normalize the sentiment counts to get proportions
    posts_sentiment_over_time = posts_sentiment_over_time.div(posts_sentiment_over_time.sum(axis=1), axis=0)

    # Plot the sentiment for the subreddit
    plt.figure(figsize=(12, 6))
    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment not in posts_sentiment_over_time.columns:
            continue
        plt.plot(posts_sentiment_over_time.index, posts_sentiment_over_time[sentiment], label=sentiment.capitalize())
    
    
    plt.title(f'Sentiment Over Time in r/{subreddit} Posts')
    plt.xlabel('Date')
    plt.ylabel('Proportion of Sentiment')
    plt.legend()
    plt.savefig(f"Time-Series Visualizations/{subreddit}.png")
    plt.clf()
