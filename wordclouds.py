import os
import numpy as np
import PIL.Image
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize

# Download the tokenizer models
nltk.download('punkt')

# List of subreddits to iterate through
subreddits = ['AskDocs', 'medicine', 'nutrition', 'fitness', 'loseit', 'keto', 'mentalhealth', 'mentalhealthsupport', 'bodyweightfitness']

def removeURLS(text):
    """
    Removes URLs from a given text using regular expressions.

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
    # Filter out non-alphabetic tokens and convert to lowercase
    filtered_tokens = [token.lower() for token in tokens if token.isalpha()]
    return ' '.join(filtered_tokens)

def loadPostsBySentiment(subreddit_name):
    """
    Loads posts from a JSON file for a given subreddit and categorizes them by sentiment.

    Parameters:
    subreddit_name (str): The name of the subreddit to load posts from.

    Returns:
    dict: A dictionary with keys 'positive', 'negative', and 'neutral', containing concatenated text for each sentiment.
    """
    file_path = f"{subreddit_name}/{subreddit_name}_posts.json"

    try:
        df = pd.read_json(file_path)
        # Create a dictionary to store concatenated text for each sentiment category
        sentiment_texts = {
            'positive': " ".join(df[df['sentiment_class'] == 'positive']['body_text']),
            'negative': " ".join(df[df['sentiment_class'] == 'negative']['body_text']),
            'neutral': " ".join(df[df['sentiment_class'] == 'neutral']['body_text'])
        }
    except FileNotFoundError:
        sentiment_texts = {'positive': '', 'negative': '', 'neutral': ''}

    return sentiment_texts

def createWordcloud(text, subreddit_name, sentiment):
    """
    Generates a word cloud from the provided text, saves it as an image file using a mask, and saves it to a specific directory.

    Parameters:
    text (str): The text to generate the word cloud from.
    subreddit_name (str): The name of the subreddit for which the word cloud is being generated.
    sentiment (str): The sentiment category ('positive', 'negative', or 'neutral') for the word cloud.

    Returns:
    None
    """
    clean_text = removeURLS(text)
    tokenized_text = tokenizeText(clean_text)
    image_mask = np.array(PIL.Image.open("reddit_logo.png"))
    output_path = f"{subreddit_name}/{subreddit_name}_{sentiment}_wordcloud.png"

    wc = WordCloud(stopwords=STOPWORDS, 
                   mask=image_mask, 
                   background_color="white",
                   min_font_size=5).generate(tokenized_text)
    
    wc.to_file(output_path)
    # plt.imshow(wc, interpolation='bilinear')
    # plt.axis("off")
    # plt.show()

if __name__ == "__main__":
    for subreddit in subreddits:
        sentiment_texts = loadPostsBySentiment(subreddit)
        for sentiment, text in sentiment_texts.items():
            if text:  # Only generate word cloud if there is text
                createWordcloud(text, subreddit, sentiment)
