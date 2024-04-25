import pickle
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_option_menu import option_menu

nltk.download('punkt')
nltk.download('stopwords')

loaded_model = joblib.load('model.pkl')

# Instantiate TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Set page config
st.set_page_config(
    page_title='Resume Classification Using NLP',
)

page_bg_img = """
<style>
  [data-testid="stAppViewContainer"] {
      background:linear-gradient(to right top, #051937, #004d7a, #008793, #00bf72, #a8eb12);
      background-size: cover;
      color: black;
  }
  [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
  }
  [data-testid="baseButton-headerNoPadding"]{
        color: black;
  }
  [data-testid="stDataFrame"] {
        
  }
  .title {
        color: black;
  }
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

def clean_resume(resume_texts):
    cleaned_resumes = []

    for text in resume_texts:
        # Perform cleaning operations on the resume text
        # Add your cleaning logic here
        cleaned_text = text.lower()  # Example: convert to lowercase

        # Append cleaned text to the list
        cleaned_resumes.append(cleaned_text)

    return cleaned_resumes

def home():
    st.subheader('Home')
    st.header('Resume Classification Using NLP')
    st.write('Business Objective')
    st.write('The business objective of our resume classification app powered by Natural Language Processing (NLP) is to streamline the recruitment process by efficiently categorizing and analyzing resumes. By leveraging advanced NLP techniques, our app aims to automatically extract key information from resumes, such as skills, experiences, and qualifications, and classify them into relevant categories or job roles. This enables recruiters and hiring managers to quickly identify top candidates, saving time and resources while ensuring a more thorough and accurate screening process. Our ultimate goal is to enhance hiring efficiency, reduce bias, and improve overall recruitment outcomes for organizations of all sizes.')

def visualize_data():
    data = pd.read_csv('output.csv')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(10, 6))
    sns.countplot(data['Category'])
    plt.title('Count of Different Categories')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot()
    

def main():
    menu_selection = option_menu(
        menu_title='Main Menu',
        options=['Home', 'Classify', 'Data Set', 'Data Visualization'],
        icons=['house', 'diagram-3', 'database-fill', 'bar-chart-2'],
        default_index=0,
        orientation='horizontal',
    )

    if menu_selection == 'Home':
        home()
    elif menu_selection == 'Classify':
        upload_file = st.file_uploader('Upload Resume', type=['pdf', 'txt', 'doc', 'docx'])

        if upload_file is not None:
            try:
                resume_bytes = upload_file.read()
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # if utf-8 decoding fails, try decoding with 'latin-1'
                resume_text = resume_bytes.decode('latin-1')

            clean_resumes = clean_resume([resume_text])
            transformed_resumes = tfidf_vectorizer.transform(clean_resumes)
            prediction_id = loaded_model.predict(transformed_resumes)[0]
            prediction_text = ''

            if prediction_id == 0:
                prediction_text = 'Intern'
            elif prediction_id == 1:
                prediction_text = 'PeopleSoft'
            elif prediction_id == 2:
                prediction_text = 'React Developer'
            elif prediction_id == 3:
                prediction_text = 'SQL Developer'
            elif prediction_id == 4:
                prediction_text = 'Workday'

            st.write("Predicted Category:", prediction_text)
            # Display the content of the uploaded document
            st.subheader("Uploaded Document Content")
            st.text_area("Resume Content", value=resume_text, height=400)

    elif menu_selection == 'Data Set':
        data = pd.read_csv('output.csv')
        st.dataframe(data)

    elif menu_selection == 'Data Visualization':
        visualize_data()

if __name__ == '__main__':
    main()
