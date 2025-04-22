from flask import Flask, request
import tensorflow as tf
import numpy as np
import pickle
import requests

app = Flask(__name__)

model = tf.keras.models.load_model("http_model.h5")
with open("http_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

@app.route('/proxy', methods=['POST'])
def proxy():
    raw_data = request.get_data(as_text=True)
    sequences = tokenizer.texts_to_sequences([raw_data])
    x = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
    prediction = model.predict(x)[0]
    label = np.argmax(prediction) 
    
    if label == 1:
        #response = requests.post("http://localhost:8080/flights", data=raw_data)
        return "Normas good"
    else:
        with open("bad_requests.txt", "a", encoding="utf-8") as f:
            f.write(raw_data + "\n")
        return "Blocked bad request", 403

if __name__ == '__main__':
    app.run(port=5000)
