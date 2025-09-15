from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from werkzeug.utils import secure_filename



# Load model
model = AutoModelForCausalLM.from_pretrained("./gemma-motivation-merged")
tokenizer = AutoTokenizer.from_pretrained("./gemma-motivation-merged")
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Initialize Flask
app = Flask(__name__)

# Serve HTML
@app.route("/")
def home():
    return render_template("chat.html")




# API endpoint
@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    user_input = request.json["message"]
    prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
    output = chatbot(prompt, max_new_tokens=100, do_sample=True)[0]["generated_text"]
    response = output.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
    suggestions = ["Motivational quote", "Breathing exercise"]


    return jsonify({
        "response": response,
        "suggestions":suggestions
    })

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        image_url = f"/{file_path}"
        return jsonify({"url": image_url})

    return jsonify({"error": "Invalid file type"}), 400


# Run the app
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
