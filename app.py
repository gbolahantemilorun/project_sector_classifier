import numpy as np
import pickle
from flask import Flask, render_template, request

# Create a Flask application
app = Flask(__name__)

# Load the trained model from the model.pkl file
with open('best_svc_model_tfidf.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the vectorizer from the vectorizer.pkl file (assuming you saved it during training)
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the project title entered by the user
    title = request.form['title']
    
    # Preprocess the project title
    preprocessed_title = vectorizer.transform([title])
    
    # Predict the sector using the loaded model
    predicted_sector = model.predict(preprocessed_title)[0]
    
    # Render the result template with the predicted sector
    return render_template('result.html', title=title, sector=predicted_sector)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)