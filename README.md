# AI-powered-chatbot-
An AI-powered chatbot built in Python that can engage in interactive conversations and provide intelligent responses
# AI-Powered Chatbot

A Python-based AI chatbot built with Flask and Hugging Face Transformers.  
It serves a simple web interface (`chat.html`) where users can interact with the bot in real time.  

---

## 🚀 Features
- Flask backend with a `/analyze_text` endpoint for chatbot responses  
- Simple and clean HTML frontend (`chat.html`)  
- Easy to extend with more suggestions or new models  
- Modular structure with clear separation of frontend and backend  

---

## 📂 Project Structure
```AI-powered-chatbot-/
│── app.py # Main Flask application
│── requirements.txt # Python dependencies
│── templates/
│ └── chat.html # Frontend UI for chatbot
│── static/ # Images, CSS, and static assets
│── .gitignore # Ignored files/folders
│── README.md # Project documentation
│── LICENSE # License file
```
---


## ⚡ How It Works

1. **User Interface (Frontend)**  
   - The chatbot interface is built with **`chat.html`** inside the `templates/` folder.  
   - When you visit [http://127.0.0.1:5000](http://127.0.0.1:5000), Flask serves this page in the browser.  
   - Users type their messages into the chat box.

2. **Flask Backend**  
   - The logic is in **`app.py`**.  
   - User messages are sent via a **POST request** to `/analyze_text`.  
   - Flask receives the input and forwards it to the chatbot model.

3. **AI Model**  
   - A Hugging Face Transformer model generates responses.  
   - Example:
     ```python
     output = chatbot(prompt, max_new_tokens=100, do_sample=True)[0]["generated_text"]
     ```

4. **Response**  
   - Flask returns the model’s output as JSON.  
   - `chat.html` displays the reply in the chat window.

✅ Flow: **User → Frontend → Flask → Model → Flask → Frontend → Response**

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nabhya1807/AI-powered-chatbot-.git
   cd AI-powered-chatbot-
2. Create the virtual environment
   python3 -m venv .venv
  source .venv/bin/activate    # Mac/Linux
  .venv\Scripts\activate       # Windows
3. Install dependencies
   pip install -r requirements.txt
4. Run the app
   python app.py

## 📜 License
This project is licensed under the MIT License.  
You’re free to use, modify, and distribute it with attribution.

