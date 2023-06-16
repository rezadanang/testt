import streamlit as st
import joblib
import re
import nltk
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from PIL import Image
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
model = joblib.load('rf.pkl')

# Load the TF-IDF vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define function for sentiment analysis
def sentiment_analysis(text):
     # Preprocess the text
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuations
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Perform stemming using PySastrawi
    stemmer = StemmerFactory().create_stemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Join the tokens back to form the preprocessed text
    preprocessed_text = ' '.join(tokens)

    # Transform the preprocessed text into TF-IDF vector representation
    text_vector = vectorizer.transform([preprocessed_text]).toarray()
    
    # Make prediction using the model
    prediction = model.predict(text_vector)[0]
    
    # Map prediction to sentiment label
    # sentiment = "Positive" if prediction == 1 else "Negative"
    if prediction == 1:
        sentiment = "Positive"
        image = Image.open('./images/positive.png')
        st.image(image, width=150)
    elif prediction == 0:
        sentiment = "Neutral"
        image = Image.open('./images/neutral.png')
        st.image(image, width=150)
    else :
        sentiment = "Negative"
        image = Image.open('./images/negative.png')
        st.image(image, width=150)
    # return sentiment, prediction

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction: {prediction}")


# Streamlit app
def main():
    # Set page title
    st.title('Sentiment Analysis')

    st.sidebar.title('Project Skripsi')
    
    # Get user input text
    user_input = st.text_input("Enter text for sentiment analysis")
    
    # Perform sentiment analysis on button click
    if st.button('Predict'):
        # Check if input is provided
        # user_input = np.array(dtype=object)
        if user_input:
            # Perform sentiment analysis
            sentiment = sentiment_analysis(user_input)
            
            # # Display result
            # st.write(f"Sentiment: {sentiment}")
            
            
           
        else:
            st.write("Please enter text for sentiment analysis")

# Run the app
if __name__ == '__main__':
    main()
