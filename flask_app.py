# flask_app.py

from flask import Flask, render_template
from flask_socketio import SocketIO
from rag_gen import ask_llm

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def home():
    return render_template("index.html")

@socketio.on("ask")
def handle_question(json):
    question = json.get("question", "")
    if not question:
        socketio.emit("response", {"answer": "No question provided."})
        return
    answer = ask_llm(question)
    socketio.emit("response", {"answer": answer})

if __name__ == "__main__":
    print("Server running on http://127.0.0.1:5500/")
    socketio.run(app, port=5500, debug=False)
