# import streamlit as st
# import joblib
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer

# model = joblib.load('rf.pkl')

# def predict_sentiment(text):
#     # Lakukan preprocessing teks
#     preprocessed_text = preprocess_text(text)
    
#     # Vektorisasi teks
#     vectorized_text = vectorizer.transform([preprocessed_text])
    
#     # Melakukan prediksi sentimen
#     prediction = model.predict(vectorized_text)
    
#     # Mengembalikan hasil prediksi
#     return prediction[0]

# def main():
#     # Menambahkan judul aplikasi
#     st.title("Analisis Sentimen dengan Random Forest")
    
#     # Menambahkan input teks
#     text = st.text_input("Masukkan teks")
    
#     if st.button("Prediksi"):
#         # Melakukan prediksi sentimen
#         prediction = predict_sentiment(text)
        
#         # Menampilkan hasil prediksi
#         st.write(f"Hasil Prediksi: {prediction}")


# if __name__ == "__main__":
#     main()


import streamlit as st
import joblib
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
model = joblib.load('rff.pkl')

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
    text_vector = vectorizer.transform([preprocessed_text])
    
    # Make prediction using the model
    prediction = model.predict([text_vector])[0]
    
    # Map prediction to sentiment label
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return sentiment

# Streamlit app
def main():
    # Set page title
    st.title('Sentiment Analysis')
    
    # Get user input text
    user_input = st.text_input('Enter text for sentiment analysis')
    
    # Perform sentiment analysis on button click
    if st.button('Process'):
        # Check if input is provided
        if user_input:
            # Perform sentiment analysis
            sentiment = sentiment_analysis(user_input)
            
            # Display result
            st.write(f"Sentiment: {sentiment}")
        else:
            st.write("Please enter text for sentiment analysis")

# Run the app
if __name__ == '__main__':
    main()
