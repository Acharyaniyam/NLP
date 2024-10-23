#Sahil Adhikari, sa3933
## This script visualizes the overall distribution of sentiments across specified subreddits.

import pandas as pd
import matplotlib.pyplot as plt

subreddits = ['health', 'AskDocs', 'medicine', 'nutrition', 'fitness', 'loseit', 'keto', 'mentalhealth', 'mentalhealthsupport', 'bodyweightfitness']

def aggregate_sentiments(subreddit_names):
    total_sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

    for subreddit_name in subreddit_names:
        file_path = f"{subreddit_name}/{subreddit_name}_posts.json"
        
        df = pd.read_json(file_path)
        sentiment_counts = df['sentiment_class'].value_counts().to_dict()

        for sentiment in ['positive', 'negative', 'neutral']:
            total_sentiment_counts[sentiment] += sentiment_counts.get(sentiment, 0)

    return total_sentiment_counts

def plot_sentiment_distribution(sentiment_counts, plot_type='bar'):
    labels = sentiment_counts.keys()
    sizes = sentiment_counts.values()

    if plot_type == 'bar':
        plt.figure(figsize=(10, 6))
        plt.bar(labels, sizes, color=['#2ca02c', '#d62728', '#1f77b4'])
        plt.xlabel('Sentiment')
        plt.ylabel('Counts')
        plt.title('Overall Sentiment Distribution Across All Subreddits')
        for i, count in enumerate(sizes):
            plt.text(i, count + 5, str(count), ha='center')
    elif plot_type == 'pie':
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#2ca02c', '#d62728', '#1f77b4'])
        plt.title('Overall Sentiment Distribution Across All Subreddits')

    plt.tight_layout()
    plt.savefig("overall_sentiment_distribution.png")
    plt.show()

if __name__ == "__main__":
    total_sentiment_counts = aggregate_sentiments(subreddits)
    plot_sentiment_distribution(total_sentiment_counts, plot_type='bar')  # or 'pie' for a pie chart

# Metrics Explanation:
# - Positive Sentiment (2781 posts): The majority of posts are positive, indicating that users generally 
#   share encouraging or optimistic content within these subreddits.
# - Negative Sentiment (1131 posts): A significant portion of posts are negative, reflecting challenges, 
#   frustrations, or negative experiences discussed by the community.
# - Neutral Sentiment (1088 posts): A considerable number of posts are neutral, suggesting that many users 
#   share factual or informational content without strong emotional tone.

# Conclusion:
# The sentiment analysis shows that the community interactions are predominantly positive, with a substantial 
# amount of negative and neutral content. This suggests that users engage in a wide range of discussions, 
# from sharing positive experiences to discussing challenges and providing neutral information.
