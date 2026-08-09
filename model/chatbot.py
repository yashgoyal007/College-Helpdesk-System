import json
import pickle
import random
import numpy as np

from tensorflow.keras.models import load_model
from model.nlp import clean_text

with open("data/intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

model = load_model("model/chatbot_model.keras")


with open("model/words.pkl", "rb") as file:
    words = pickle.load(file)

with open("model/classes.pkl", "rb") as file:
    classes = pickle.load(file)


def bag_of_words(sentence):
    sentence_words = clean_text(sentence)

    bag = []

    for word in words:
        if word in sentence_words:
            bag.append(1)
        else:
            bag.append(0)

    return np.array(bag)


def predict_intent(sentence):
    input_data = bag_of_words(sentence)
    input_data = np.array([input_data])

    prediction = model.predict(input_data, verbose=0)[0]

    index = np.argmax(prediction)

    tag = classes[index]
    confidence = prediction[index]

    return tag, confidence


def get_response(tag):
    direct_responses = {
        "classrooms": [
            "You can check your timetable for the assigned classroom and use the campus map to locate the building.",
            "Please check your timetable for the classroom number and campus map for the building location."
        ],

        "canteen": [
            "The college canteen provides food and refreshments to students. Check the campus notice board for current timings."
        ],

        "fees": [
            "Please check the student portal or contact the accounts department for the current fee details."
        ],

        "library": [
            "The library provides books and study resources for students. Please check the library notice for current timings and borrowing rules."
        ],

        "timings": [
            "College timings may vary by course and schedule. Please check the latest timetable or college notice for current timings.",
            "You can check the academic timetable or contact the college administration for the current working hours."
        ]
    }

    if tag in direct_responses:
        return random.choice(direct_responses[tag])

    for intent in intents["intents"]:
        if intent["tag"].lower() == tag.lower():
            return random.choice(intent["responses"])

    return "Sorry, I don't understand that question."


def keyword_intent(sentence):
    """
    Handles highly specific helpdesk questions
    before sending them to the ML model.
    """

    text = sentence.lower()

    # Classroom / classroom location
    classroom_words = [
        "classroom",
        "class room",
        "lecture room",
        "my class",
        "where is my class"
    ]

    if any(word in text for word in classroom_words):
        return "classrooms"


    # Canteen
    canteen_words = [
        "canteen",
        "food",
        "eat on campus"
    ]

    if any(word in text for word in canteen_words):
        return "canteen"


    # Library
    library_words = [
        "library",
        "books",
        "borrow books",
        "issue book"
    ]

    if any(word in text for word in library_words):
        return "library"


    # Fees
    fee_words = [
        "fee",
        "fees",
        "tuition",
        "semester fee"
    ]

    if any(word in text for word in fee_words):
        return "fees"


    # College timings
    timing_words = [
        "college timings",
        "college timing",
        "college hours",
        "college starts",
        "college start",
        "college closes",
        "college close",
        "college working hours",
        "college schedule"
    ]

    if any(word in text for word in timing_words):
        return "timings"


    return None


def get_answer(message):
    """
    Hybrid AI approach:
    1. Check specific college keywords.
    2. Otherwise use the trained ML model.
    """

    direct_intent = keyword_intent(message)

    if direct_intent:
        return get_response(direct_intent), direct_intent, 1.0


    tag, confidence = predict_intent(message)


    if confidence < 0.50:
        return (
            "Sorry, I don't understand that question. "
            "Please try asking in another way.",
            "unknown",
            float(confidence)
        )


    return get_response(tag), tag, float(confidence)


# Terminal testing
if __name__ == "__main__":

    print("College AI Helpdesk")
    print("Type 'quit' to exit.")

    while True:

        message = input("\nYou: ")

        if message.lower() == "quit":
            print("AI: Goodbye!")
            break

        response, intent, confidence = get_answer(message)

        print("AI:", response)
        print("Intent:", intent)
        print("Confidence:", round(confidence, 2))