import pandas as pd
from transformers import pipeline

# Function to return sentiment and confidence score
def apply_sentiment_analysis(text):
    try:
        result = sentiment_analysis_model(text)[0]
        return result['label'], result['score'] # return Sentiment, Confidence Score
    except Exception as e:
        return "Error", 0  # if something goes wrong, return Error, 0 confidence score

if __name__ == "__main__":

    # initialize each subfolders of subreddits
    subfolders = ['AskDocs', 'bodyweightfitness', 'fitness', 'health', 'keto', 'loseit', 'medicine', 'mentalhealth', 'mentalhealthsupport', 'nutrition']

    # Load BERT sentiment analysis model
    sentiment_analysis_model = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

    # for each subfolder/subreddit we want to apply sentiment analysis to their posts
    for each_subfolder in subfolders:
        #initializing the paths
        path_to_subfolder = f'{each_subfolder}/'
        path_to_json = path_to_subfolder + f'{each_subfolder}_posts.json'

        # read in json
        data = pd.read_json(path_to_json)

        # get each posts' text
        body_texts = data['body_text'].tolist()

        # Apply sentiment analysis to each posts' text
        sentiment_results = list(map(apply_sentiment_analysis, body_texts))

        # Create new columns in the Json file for Sentiment and Confidence Score
        data['Sentiment'] = [result[0] for result in sentiment_results]
        data['Confidence'] = [result[1] for result in sentiment_results]

        # Save the results to a JSON file in their respective subfolder paths
        output_file = path_to_subfolder + f'{each_subfolder}_BERT_sentiment.json'
        data.to_json(output_file, orient='records', indent=4)

        # Checking if program runs after each subfolder
        print(f"Sentiment analysis completed for {each_subfolder}")