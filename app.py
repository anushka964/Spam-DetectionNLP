import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the Porter Stemmer
ps = PorterStemmer()

def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    try:
        text = nltk.word_tokenize(text)
        print(f"Tokenized Text: {text}")  # Debugging line
    except Exception as e:
        print(f"Error in Tokenization: {e}")
        return ""
    
    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]
    print(f"After removing non-alphanumeric words: {text}")  # Debugging line
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
    print(f"After removing stopwords and punctuation: {text}")  # Debugging line

    # Perform stemming
    text = [ps.stem(word) for word in text]
    print(f"After stemming: {text}")  # Debugging line

    # Join the list of words back into a string
    return " ".join(text)

# Load the trained model and vectorizer
tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Streamlit title
st.title("SMS Spam Detection Model")

# User input for SMS
input_sms = st.text_input("Enter the SMS")

# Prediction on button click
if st.button('Predict'):
    if not input_sms:
        st.warning("Please enter a message.")
    else:
        # 1. Preprocess the input SMS
        transformed_sms = transform_text(input_sms)
        print(f"Transformed SMS: {transformed_sms}")  # Debugging line
        
        # 2. Vectorize the transformed SMS
        vector_input = tk.transform([transformed_sms])  # Ensure it's wrapped in a list
        print(f"Vector input shape: {vector_input.shape}")  # Debugging line
        
        # 3. Predict whether the message is spam or not
        try:
            result = model.predict(vector_input)[0]
            print(f"Prediction result: {result}")  # Debugging line

            # 4. Display the result
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
        except ValueError as e:
            st.error(f"Error during prediction: {e}")
            print(f"Error during prediction: {e}")  # Debugging line
