import streamlit as st
import re
import joblib
import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Load the trained model and vectorizer
model = joblib.load("rf_model.joblib") 
vectorizer = joblib.load("vectorizer.joblib")  

# Function to remove @user in tweet
def remove_pattern(input_txt, pattern):
    return re.sub(pattern, "", input_txt)

# Custom tokenizer (maintain hashtags)
def custom_tokenizer(tweet):
    # Regex to match words, numbers, hashtags, and mentions as individual tokens
    tokens = re.findall(r'\#\w+|\w+', tweet)
    return tokens

# Function to clean tweets
def clean_tweet(tweet):
    # Remove special characters, numbers, and punctuation
    tweet = re.sub(r"[^a-zA-Z\s#]", "", tweet)
    # Convert text to lowercase
    tweet = tweet.lower()
    # Remove extra spaces
    tweet = re.sub(r"\s+", " ", tweet).strip()
    return tweet

# Function to remove stopwords
def remove_stopwords(tweet):
    # creating stopwords in english
    stop_words = set(stopwords.words("english"))
    # tokenise words
    words = custom_tokenizer(tweet)
    # remove stopwords
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

# Function to Tokenise and Stem
def token_stemmer(tweet):
    #Tokenise tweets
    tweet = custom_tokenizer(tweet)
    #stem Tweets
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for token in tweet:
        if token.startswith('#'):
            # Keep hashtags intact
            stemmed_tokens.append(token)
        else:
            # Stem other words
            stemmed_tokens.append(stemmer.stem(token))
    
    return " ".join(stemmed_tokens)

# preprocess New Tweets
def predict_tweet(tweet):
    cleaned_tweet= remove_pattern(tweet, "@[A-Za-z0-9]+")
    cleaned_tweet= clean_tweet(tweet)
    cleaned_tweet= remove_stopwords(tweet)
    cleaned_tweet= token_stemmer(tweet)
    transformed_tweet = vectorizer.transform([tweet])  
    prediction = model.predict(transformed_tweet)  
    return "Hate Speech" if prediction[0] == 1 else "Not Hate Speech"

# Streamlit App UI
st.title("Hate Speech Detection App")
st.subheader("Enter a tweet to classify whether it contains hate speech or not")

# Input text box
user_input = st.text_area("Enter Tweet:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet to analyze.")
    else:
        result = predict_tweet(user_input)
        if result == "Hate Speech":
            st.error(f"Prediction: {result}")
        else:
            st.success(f"Prediction: {result}")

# Footer
st.markdown("Thank you!!!")
