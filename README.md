trafficChecker
trafficChecker is a machine learning-based tool for analyzing server requests and redirecting them depending on their security. The project utilizes natural language processing (NLP) to classify HTTP requests and determine whether they are normal or potentially malicious. It uses a pre-trained machine learning model to assess incoming requests and make real-time decisions about their safety.

Features
HTTP request classification using machine learning

Malicious request detection

Integration with a Flask API for real-time analysis

Classification model with a pre-trained TensorFlow model

Requirements
Python 3.x

TensorFlow

Flask

scikit-learn

pandas

numpy

Installation
Clone the repository:

bash
Копировать
Редактировать
git clone https://github.com/Plgdhd/trafficChecker.git
cd trafficChecker
Install the dependencies:

bash
Копировать
Редактировать
pip install -r requirements.txt
Run the Flask API:

bash
Копировать
Редактировать
python returner.py
Usage
Start the Flask server, and use the provided API endpoints to analyze HTTP requests.

The system uses a trained model to classify requests and determine whether they are secure.
