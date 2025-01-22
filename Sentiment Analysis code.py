import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the dataset
df = pd.read_csv('D:/Google Classroom/5th TRIMESTER/TSMA/reviews.csv')

# Display the first few rows of the dataframe
df.head()

# Drop irrelevant columns
df = df.drop(['Total_thumbsup', 'Reply'], axis=1)

# Check for missing values in the DataFrame
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

# Check for duplicate records in the DataFrame
duplicates = df.duplicated().sum()
print("Number of duplicate records:", duplicates)

# Drop duplicate records if any
df = df.drop_duplicates()

# Convert text to lowercase and remove non-alphanumeric characters
df['Review'] = df['Review'].str.lower().apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

# Download the stopwords resource
nltk.download('stopwords')

# Define stop words
stopwords = set(nltk.corpus.stopwords.words('english'))

# Remove stop words from the reviews
df['Review'] = df['Review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))

# Download the VADER lexicon used for sentiment analysis
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis to each review text
df['sentiment_scores'] = df['Review'].apply(lambda x: sia.polarity_scores(x))

# Categorize the reviews into sentiment categories
df['sentiment_category'] = df['sentiment_scores'].apply(
    lambda x: 'positive' if x['compound'] >= 0.05 else ('negative' if x['compound'] <= -0.05 else 'neutral')
)

import matplotlib.pyplot as plt

# Count the number of reviews in each sentiment category
sentiment_counts = df['sentiment_category'].value_counts()

# Plot the bar chart
plt.figure(figsize=(8, 6))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'blue'])
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Reviews')
plt.title('Distribution of Sentiment Categories')
plt.xticks(rotation=45)
plt.show()

# Display the final dataframe
print(df.head())
