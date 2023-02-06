#step 1
import requests

# API endpoint for the New York Times Article Search API
url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

# Search parameters for articles containing the term "economic development"
params = {
    "q": "economic development",
    "api-key": "pVNXbP0D9tSdz8LWiZ0seWKu1tsv5srb"
}

# Send a GET request to the API and store the response
response = requests.get(url, params=params)

# Extract the articles from the response
articles = response.json()["response"]["docs"]

# Iterate over the articles and print their headline and lead_paragraph
for article in articles:
    print("Headline:", article["headline"]["main"])
    print("Lead Paragraph:", article["lead_paragraph"])

#step 2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Iterate over the articles
for article in articles:
    # Tokenize the lead_paragraph
    words = word_tokenize(article["lead_paragraph"])

    # Remove stop words and punctuation
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the words back into a single string
    preprocessed_text = ' '.join(words)

    # Store the preprocessed text
    article["preprocessed_text"] = preprocessed_text

    print("Preprocessed Text:", preprocessed_text)


#step 3
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

sentiment_analyzer = SentimentIntensityAnalyzer()

# Iterate over the articles
for article in articles:
    # Get the sentiment score from the SentimentIntensityAnalyzer
    sentiment_score = sentiment_analyzer.polarity_scores(article["preprocessed_text"])["compound"]

    # Assign the sentiment label based on the sentiment score
    if sentiment_score >= 0.05:
        sentiment_label = "positive"
    elif sentiment_score <= -0.05:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    # Store the sentiment label
    article["sentiment_label"] = sentiment_label

    print("Sentiment Label:", sentiment_label)

#step 4
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit the CountVectorizer to the preprocessed text
vectorized_text = vectorizer.fit_transform([article["preprocessed_text"] for article in articles])

# Convert the vectorized text to an array
bag_of_words = vectorized_text.toarray()

print("Bag of Words Shape:", bag_of_words.shape)

#step 5
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
labels = [1, -1, -1, 0, -1, 1, 1, 0, 1, 1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(bag_of_words, labels, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(X_train, y_train)

# Predict the sentiment of the testing data
y_pred = nb_classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

#step 6, analysis on unseen data
from textblob import TextBlob

# New, unseen data
new_data = [
    "The government's recent economic policies have had a positive impact on the country's growth.",
    "The government's recent economic policies have had a negative impact on the country's growth.",
    "Many people are angry that they can't make ends meet despite the announcment.",
    "The current state of the economy is uncertain and causing concern among investors.",
    "XYZ Corp is making strives in economic development",
    "XYZ Corp is falling behind the rest of the market",
    "Rising inflation is inlficting hardship on developers and consumers",
    "but now we're seeing taxpayer dollars go into an adversary, a Chinese corporation."
]

# Predict the sentiment of the new data
for text in new_data:
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        print(f"Text: {text}\nSentiment: Positive\n")
    elif sentiment < 0:
        print(f"Text: {text}\nSentiment: Negative\n")
    else:
        print(f"Text: {text}\nSentiment: Neutral\n")




