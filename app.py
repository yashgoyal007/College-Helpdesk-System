from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from model.chatbot import get_answer

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return send_from_directory("frontend", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({
            "error": "Message is required"
        }), 400

    message = data["message"].strip()

    if not message:
        return jsonify({
            "error": "Message cannot be empty"
        }), 400

    response, intent, confidence = get_answer(message)

    return jsonify({
        "message": message,
        "response": response,
        "intent": intent,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)