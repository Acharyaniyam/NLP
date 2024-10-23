import threading
import praw
from datetime import datetime
import pandas as pd
import concurrent.futures
import os
import praw.exceptions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Lock object for controlling access to shared resources across threads
progress_lock = threading.Lock()

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Credentials
reddit = praw.Reddit(
    client_id="nr_SgHYSQdNS43iFNz21-w",
    client_secret="l0BTV4udzVvStvlZcIbZ1OrKLlcd3g",
    user_agent="INFO-440-001"
)

# List of subreddits to iterate through
subreddits = ['health', 'AskDocs', 'medicine', 'nutrition', 'fitness', 'loseit', 'keto', 'mentalhealth', 'mentalhealthsupport', 'bodyweightfitness']

# List to store posts that failed to be processed
failed_posts = []

def analyzeSentiment(text):
    """
    Analyzes the sentiment of a given text using the VADER sentiment analyzer.

    Parameters:
    text (str): The text to be analyzed.

    Returns:
    tuple: A dictionary containing the sentiment scores and a string representing the classified sentiment ('positive', 'negative', or 'neutral').
    """
    # Get the sentiment scores
    sentiment = analyzer.polarity_scores(text)
    compound = sentiment['compound']

    # Classify the sentiment based on the compound score
    if compound >= 0.05:
        sentiment_class = 'positive'
    elif compound <= -0.05:
        sentiment_class = 'negative'
    else:
        sentiment_class = 'neutral'
    
    return sentiment, sentiment_class

def fetchPosts(subreddit_name):
    """
    Fetches the top 500 posts from a specified subreddit, analyzes the sentiment of each post, 
    and saves the data to a JSON file.

    Parameters:
    subreddit_name (str): The name of the subreddit to fetch posts from.

    Returns:
    list: A list of dictionaries containing the post data, including sentiment scores and classifications.
    """
    try:
        instance = reddit.subreddit(subreddit_name)
        post_data = []
        
        # Iterate over top 500 posts in the subreddit
        for post in instance.top(limit=500):
            date = datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            sentiment, sentiment_class = analyzeSentiment(post.selftext)
            data = {
                'post_id': post.id,
                'post_name': post.title,
                'user': post.author.name if post.author else 'N/A',
                'body_text': post.selftext,
                'num_comments': post.num_comments,
                'post_date': date,
                'sentiment': sentiment,
                'sentiment_class': sentiment_class
            }
            post_data.append(data)
        
        df = pd.DataFrame(post_data)

        os.makedirs(subreddit_name, exist_ok=True)
        file_path = f"{subreddit_name}/{subreddit_name}_posts.json"
        df.to_json(file_path, orient='records', indent=4)

        with progress_lock:
            print(f"Fetched r/{subreddit_name} data and saved to {file_path}.")
        
        return post_data

    except Exception as e:
        print(f"Failed to fetch {subreddit_name}: {e}")
        return None

def fetchComments(subreddit_name):
    """
    Fetches comments for each post in a specified subreddit and saves them to a JSON file.

    Parameters:
    subreddit_name (str): The name of the subreddit to fetch comments from.

    Returns:
    None
    """
    post_data = pd.read_json(f"{subreddit_name}/{subreddit_name}_posts.json")
    comment_path = f"{subreddit_name}/Comments"

    # Iterate over each post in the subreddit
    for _, post in post_data.iterrows():
        try:
            post_id = post['post_id']
            submission = reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)

            comment_data = []

            # Collect all direct comments for each post
            for top_level_comment in submission.comments.list():
                date = datetime.utcfromtimestamp(top_level_comment.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                data = {
                    'post_id': post_id,
                    'comment_id': top_level_comment.id,
                    'name': top_level_comment.author.name if top_level_comment.author else 'N/A',
                    'comment_body': top_level_comment.body,
                    'comment_date': date
                }
                comment_data.append(data)
            
            df = pd.DataFrame(comment_data)

            os.makedirs(comment_path, exist_ok=True)
            file_path = f"{comment_path}/{post_id}_comments.json"
            df.to_json(file_path, orient='records', indent=4)
        
        except Exception as e:
            print(f"{post_id} in r/{subreddit_name}: {e}")
            failed_posts.append((subreddit_name, post_id))

def failedComments(subreddit_name, post_id):
    """
    Attempts to fetch comments for a specific post that previously failed to be processed.

    Parameters:
    subreddit_name (str): The name of the subreddit where the post is located.
    post_id (str): The ID of the post for which comments are being fetched.

    Returns:
    None
    """
    comment_path = f"{subreddit_name}/Comments"

    try:
        submission = reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)

        comment_data = []

        # Collect all comments for the post
        for top_level_comment in submission.comments.list():
            date = datetime.utcfromtimestamp(top_level_comment.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            data = {
                'post_id': post_id,
                'comment_id': top_level_comment.id,
                'name': top_level_comment.author.name if top_level_comment.author else 'N/A',
                'comment_body': top_level_comment.body,
                'comment_date': date
            }
            comment_data.append(data)

        df = pd.DataFrame(comment_data)

        os.makedirs(comment_path, exist_ok=True)
        file_path = f"{comment_path}/{post_id}_comments.json"
        df.to_json(file_path, orient='records', indent=4)

        with progress_lock:
            print(f"Reran fetching Post ID: {post_id}")
    
    except Exception as e:
        print(f"Reran {post_id} in r/{subreddit_name}: {e}")

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        post = list(executor.map(fetchPosts, subreddits))
        comments = list(executor.map(fetchComments, subreddits))
        print(f"Failed Posts: {len(failed_posts)}")

        for subreddit_name, post_id in failed_posts:
            executor.submit(failedComments, subreddit_name, post_id)