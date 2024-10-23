# Sentiment Analysis on Health Topics in Reddit Communities

## Project Overview
This project is a **Sentiment Analysis** of various health-related discussions on Reddit. The analysis focuses on posts from health-related subreddits such as r/health, r/AskDocs, and r/mentalhealth, with the goal of understanding public sentiment on health topics. The project also includes **Time Series Sentiment Analysis** to track changes in sentiment over time.

## Contributors
- **Niyam Acharya**
- **Richardson Chhin**
- **Sahil Adhikari**
- **Benson Zhang**

## Key Features
### Data Collection:
Reddit data was collected using the Reddit API through the PRAW library, targeting key subreddits focused on health.

### Sentiment Analysis:
Sentiment analysis was performed using NLP tools like **VADER** and **BERT** to classify posts and comments as positive, negative, or neutral.

### Time Series Analysis:
Sentiment data was tracked over a timeline to analyze shifts in sentiment from 2014 to 2024, which provided insights into how public opinion on health topics evolved over time.

### Data Visualization:
Various visualizations, such as **Word Clouds**, **Sentiment Distribution Graphs**, **Bar Charts**, **Pie Charts**, and **Line Graphs** for time series, were created to represent the data insights.

## Folder and File Structure
### Folders:
- `loseit/`: Sentiment data related to weight loss and fitness.
- `medicine/`: Data from healthcare-related discussions.
- `mentalhealth/`: Sentiment data from mental health conversations.
- `nutrition/`: Sentiment data from nutrition discussions.
- `Time-Series Visualizations/`: Time-series graphs visualizing sentiment trends over time.

### Key Scripts:
- `main for final project.py`: The main script handling sentiment analysis tasks.
- `overall_sentiment_distribution.py`: Script to analyze and visualize the overall sentiment distribution.
- `sentiment_counts.py`: Script for computing sentiment counts from the dataset.
- `time-series sentiment analysis.py`: Script to conduct time-series analysis on sentiment data.
- `wordclouds.py`: Script to generate word clouds for visualizing frequent words in posts.

### Key Outputs:
- `overall_sentiment_distribution.png`: A graphical representation of sentiment distribution across subreddits.
- `sentiment_counts.png`: Bar chart of sentiment counts for health-related subreddits.

## Methodology
### Data Cleaning:
Text data from Reddit posts was preprocessed to remove URLs, special characters, and stop words. The text was then tokenized using NLTK.

### Sentiment Analysis:
**VADER** (Valence Aware Dictionary and sEntiment Reasoner) and **BERT** were used for sentiment classification. Posts and comments were classified as positive, neutral, or negative based on the sentiment score.

### Time Series Sentiment Analysis:
The project tracked sentiment changes over time, with a focus on significant periods like the COVID-19 pandemic, which showed an increase in negative sentiment.

### Data Visualization:
Word clouds and sentiment graphs were used to illustrate key trends and frequently mentioned terms.

## Challenges
- **Data Volume**: Managing large datasets from multiple subreddits slowed down processing.
- **Sentiment Misclassification**: Sentiment analysis tools sometimes struggled with detecting nuances like sarcasm, leading to occasional misclassification.
- **Tool Integration**: Different team members working in varying environments created integration issues during tool setup.

## Lessons Learned
- **Team Collaboration**: Effective communication and regular check-ins helped the team stay on track despite the challenges of data volume and tool integration.
- **Sentiment Analysis Complexity**: The complexity of accurately classifying neutral sentiments highlighted the need for more advanced models.
- **Time Management**: This project reinforced the importance of setting realistic deadlines and checking progress regularly.

## Future Directions
- **Broader Data Scope**: Extending the analysis to include more subreddits and other social media platforms like Twitter and YouTube.
- **Deeper Sentiment Analysis**: Incorporating more advanced sentiment analysis techniques to handle sarcasm and mixed emotions more accurately.
- **Temporal Shifts**: Exploring predictive models to analyze future sentiment trends based on historical data.

