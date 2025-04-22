import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('http_requests.csv')
df['request_data'] = df['request_data'].str.strip()  # Очистка строк

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['request_data'])
sequences = tokenizer.texts_to_sequences(df['request_data'])
X = pad_sequences(sequences, maxlen=100)

labels = df['label'].astype('category')
Y = pd.get_dummies(labels).values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['label']),
    y=df['label']
)
class_weights_dict = dict(enumerate(class_weights))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=32, input_length=100),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), class_weight=class_weights_dict)
model.save("http_model.h5")
with open('http_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# preds = model.predict(X_test)
# pred_labels = preds.argmax(axis=1)
# true_labels = y_test.argmax(axis=1)

# print(classification_report(true_labels, pred_labels, target_names=labels.cat.categories))
