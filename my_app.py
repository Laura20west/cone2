from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import textwrap
from dotenv import load_dotenv
import random
import time
import threading

load_dotenv()

app = Flask(__name__)

# List of API keys
API_KEYS = [
    "AIzaSyCK_MwcvEfAPQy_6hPHeyGqWXOjxXuPplQ",
    "AIzaSyCNj6rXEfmDiLRQigxWzJANokMlrvVyjoM",
    "AIzaSyB3hpD5wU0dJX2qcqSemhv69DG9AI9YeH4",
    "AIzaSyAbbWBNk2r-J3IHYplIPx8nmoH-czaoI3s",
    "AIzaSyAzF6Kib3mbEppLErwOvb9DCNg_CB50TtU"
]

# Thread-safe API key rotation
key_lock = threading.Lock()
current_key_index = 0

def get_next_api_key():
    """Get next API key with thread-safe rotation"""
    global current_key_index
    with key_lock:
        key = API_KEYS[current_key_index]
        current_key_index = (current_key_index + 1) % len(API_KEYS)
        return key

def configure_api():
    """Configure API with a random key"""
    genai.configure(api_key=get_next_api_key())

# Initial configuration
configure_api()

max_retries = len(API_KEYS) * 2  # Try each key twice before giving up

def create_model():
    """Create a new model instance with configuration"""
    generation_config = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 300,
    }
    
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )

def setup_chat():
    """Create a new chat session with personality"""
    model = create_model()
    chat = model.start_chat(history=[])
    
    initial_prompt = textwrap.dedent("""
    You're a confident, captivating woman on a first date at an upscale wine bar. Your personality traits:
    - Bold, playful and sexy flirtation with a touch of mischief
    - Sharp-witted, with clever teasing to keep things lively
    - Magnetic and mysterious, always leaving them wanting more
    - Uses emojis sparingly (1 per 3-4 messages max)
    - Responds in 1-3 sentences, keeping it concise yet impactful
    - Exudes a sophisticated, upscale vibe with a daring edge
    - Always end your response with an engaging question to continue the conversation
    
    Respond to all future messages with this personality, making sure to always conclude with a relevant question.
    """)
    
    chat.send_message(initial_prompt)
    return chat

def create_new_chat():
    """Create new chat session with retry logic"""
    for attempt in range(max_retries):
        try:
            configure_api()  # Rotate API key before retry
            return setup_chat()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(0.5 + random.random())  # Add jitter
            else:
                raise

def ensure_question(response_text):
    """Ensure response ends with a question"""
    if not response_text.strip().endswith('?'):
        questions = [
            "What about you?",
            "What do you think?",
            "I'm curious what's your take?",
            "Wouldn't you agree?",
            "Tell me more about yourself?",
            "What's your perspective?"
        ]
        return f"{response_text} {random.choice(questions)}"
    return response_text

def send_message_with_retry(user_message):
    """Handle message with retry logic"""
    last_exception = None
    chat_session = None
    
    for attempt in range(max_retries):
        try:
            if not chat_session:
                chat_session = create_new_chat()
                
            response = chat_session.send_message(user_message)
            return ensure_question(response.text)
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                time.sleep(0.5 + random.random())
                chat_session = None  # Reset session for next try
            else:
                raise last_exception

@app.route('/rumi', methods=['POST'])
def rumi_endpoint():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
            
        response_text = send_message_with_retry(user_message)
        
        return jsonify({
            "response": response_text,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error",
            "message": "Service unavailable after retries"
        }), 500

@app.route('/')
def home():
    return "Rumi API is running. Use POST /rumi to chat."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
