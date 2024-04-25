import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random  # Import random module
from streamlit_option_menu import option_menu

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set the page configuration with the background gradient
st.set_page_config(
    page_title="Bankruptcy Classification App",
    page_icon=":chart_with_upwards_trend:",
    initial_sidebar_state="collapsed"  # or "auto"
)

page_bg_img = """
<style>
  [data-testid="stAppViewContainer"] {
      background: linear-gradient(90deg, rgba(76,161,175,1) 0%, rgba(196,224,229,1) 100%);
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

def predict(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk):
    # Perform prediction
    predictions = model.predict([[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]])
    probabilities = model.predict_proba([[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]])

    # Extract probability for the predicted class
    prediction_prob = probabilities[0][1] if predictions == 1 else probabilities[0][0]
    prediction_prob_percent = round(prediction_prob * 100, 2)
    prediction_text = 'Non-Bankrupt' if predictions == 1 else 'Bankrupt'

    # List of random tips
    tips = [
        "To improve your model's performance, consider collecting more diverse data.",
        "Regularly updating your model with new data can help it stay accurate over time.",
        "Try experimenting with different machine learning algorithms to see which one works best for your problem.",
        "Remember to evaluate your model's performance using appropriate metrics.",
        "Feature engineering can often lead to significant improvements in model accuracy.",
        "Consider using ensemble methods to combine the predictions of multiple models for better results.",
        "Keep in mind the ethical implications of your machine learning project and ensure fairness and transparency.",
        "Document your machine learning process thoroughly to facilitate reproducibility and collaboration.",
        "Don't forget to validate your model's performance on unseen data to ensure generalization.",
    ]

    # Display random tip
    random_tip = random.choice(tips)
    tip_message = f"Random Tip: {random_tip}"
    st.subheader(tip_message)

    return f'The company is under {prediction_text} category with a probability of {prediction_prob_percent}%'

def home():
    # Display home content
    st.subheader("Home")
    st.write("Welcome to the Bankruptcy Classification Project!")
    st.write('Business Objective')
    st.write('The main objective of the project is to provide insight into whether a company will go bankrupt or not based on certain scores that the company obtained by their performance to date.')
    st.write('Different Categories of scores include the following:')
    
def visualize_data(data):
    # Display data visualizations
    st.subheader("Data Visualization")
    st.write("Here are some visualizations of the dataset:")
    
    # Bar chart of the class distribution
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("Bar Chart of Bankruptcy Class Distribution:")
    class_distribution = data[' class'].value_counts()
    plt.bar(class_distribution.index, class_distribution.values)
    plt.xlabel('Bankruptcy Class')
    plt.ylabel('Frequency')
    st.pyplot()

    # Pairplot
    st.write("Pairplot:")
    sns.pairplot(data)
    st.pyplot()  # Corrected the function call to display the plot

def main():
    st.title('Bankruptcy Prediction App')

    # Sidebar
    with st.sidebar:
        menu_selection = option_menu(
            menu_title='Main Menu',
            options=['Home', 'Prediction', 'Data Set', 'Data Visualization'],   
            icons=['house', 'activity', 'database-fill', 'bar-chart'],
            default_index=0
        )

    if menu_selection == "Home":
        home()
    elif menu_selection == "Prediction":
        # Sliders for input features
        industrial_risk = st.slider('Industrial Risk', 0.0, 1.0, 0.0)
        management_risk = st.slider('Management Risk', 0.0, 1.0, 0.0)
        financial_flexibility = st.slider('Financial Flexibility', 0.0, 1.0, 0.0)
        credibility = st.slider('Credibility', 0.0, 1.0, 0.0)
        competitiveness = st.slider('Competitiveness', 0.0, 1.0, 0.0)
        operating_risk = st.slider('Operating Risk', 0.0, 1.0, 0.0)

        if st.button('Predict'):
            result = predict(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk)
            st.success(result)
    elif menu_selection == "Data Set":
        try:
            data = pd.read_excel('bank_p301.xlsx')
            st.dataframe(data)
        except FileNotFoundError:
            st.error("Data file not found. Please make sure 'bank_p301.xlsx' is in the current directory.")
    elif menu_selection == "Data Visualization":
        try:
            data = pd.read_excel('bank_p301.xlsx')
            visualize_data(data)
        except FileNotFoundError:
            st.error("Data file not found. Please make sure 'bank_p301.xlsx' is in the current directory.")

if __name__ == '__main__':
    main()
