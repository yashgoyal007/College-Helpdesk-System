import json
import pickle
import numpy as np

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

from nlp import clean_text


with open("data/intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []

for intent in data["intents"]:
    tag = intent["tag"]

    if tag not in classes:
        classes.append(tag)

    for pattern in intent["patterns"]:
        word_list = clean_text(pattern)

        words.extend(word_list)
        documents.append((word_list, tag))

words = sorted(set(words))
classes = sorted(set(classes))

print("Number of words:", len(words))
print("Number of classes:", len(classes))

training = []

for document in documents:
    word_patterns = document[0]
    tag = document[1]

    bag = []

    for word in words:
        if word in word_patterns:
            bag.append(1)
        else:
            bag.append(0)

    output_row = [0] * len(classes)
    output_row[classes.index(tag)] = 1

    training.append([bag, output_row])

training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

model = Sequential()

model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(len(train_y[0]), activation="softmax"))

sgd = SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)

model.compile(
    loss="categorical_crossentropy",
    optimizer=sgd,
    metrics=["accuracy"]
)

print("\nTraining the model...")

history = model.fit(
    train_x,
    train_y,
    epochs=200,
    batch_size=5,
    verbose=1
)

model.save("model/chatbot_model.keras")

with open("model/words.pkl", "wb") as file:
    pickle.dump(words, file)

with open("model/classes.pkl", "wb") as file:
    pickle.dump(classes, file)


print("\nTraining completed successfully!")
print("Model saved as model/chatbot_model.keras")
print("Words saved as model/words.pkl")
print("Classes saved as model/classes.pkl")

