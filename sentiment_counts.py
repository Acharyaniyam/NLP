import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
import re

# Download the tokenizer models
nltk.download('punkt')

# List of subreddits to iterate through
subreddits = ['health', 'AskDocs', 'medicine', 'nutrition', 'fitness', 'loseit', 'keto', 'mentalhealth', 'mentalhealthsupport', 'bodyweightfitness']

def removeURLS(text):
    """
    Removes URLs from the given text using regular expressions.
    
    Parameters:
    text (str): The text from which URLs will be removed.
    
    Returns:
    str: The text with URLs removed.
    """
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def tokenizeText(text):
    """
    Tokenizes the input text into words and removes non-alphabetic tokens.
    
    Parameters:
    text (str): The text to tokenize.
    
    Returns:
    str: The tokenized text joined into a single string.
    """
    tokens = word_tokenize(text)
    filtered_tokens = [token.lower() for token in tokens if token.isalpha()]
    return ' '.join(filtered_tokens)

def sentimentVisualize(subreddit_names):
    """
    Visualizes the distribution of sentiments (positive, negative, neutral) across multiple subreddits.
    
    Parameters:
    subreddit_names (list): A list of subreddit names to visualize.
    
    Returns:
    None
    """
    sentiment_data = []

    for subreddit_name in subreddit_names:
        file_path = f"{subreddit_name}/{subreddit_name}_posts.json"

        try:
            df = pd.read_json(file_path)
            
            # Tokenize and clean text data for sentiment analysis
            df['body_text'] = df['body_text'].apply(lambda x: tokenizeText(removeURLS(x)))
            
            # Get the count of each sentiment category
            sentiment_counts = df['sentiment_class'].value_counts().to_dict()
            
            # Ensure all sentiment categories are represented in the counts
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment not in sentiment_counts:
                    sentiment_counts[sentiment] = 0

            sentiment_data.append({
                'subreddit': subreddit_name,
                'Positive': sentiment_counts['positive'],
                'Negative': sentiment_counts['negative'],
                'Neutral': sentiment_counts['neutral']
            })
        except FileNotFoundError:
            print(f"File not found for subreddit: {subreddit_name}")

    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_df = sentiment_df.melt(id_vars='subreddit', var_name='Sentiment', value_name='Counts')

    plt.figure(figsize=(16, 8))
    barplot = sns.barplot(data=sentiment_df, x='subreddit', y='Counts', hue='Sentiment', palette='viridis')

    # Add annotations on each bar
    for p in barplot.patches:
        barplot.annotate(
            format(int(p.get_height()), ','),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 8),
            textcoords='offset points'
        )

    plt.xlabel('Subreddits')
    plt.ylabel('Counts')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    plt.tight_layout()

    plt.savefig(f"sentiment_counts.png")

if __name__ == "__main__":
    sentimentVisualize(subreddits)
