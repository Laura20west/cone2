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
    "AIzaSyApqkaUDeuSFAoTvRTIdu4IU6AlTJOi4xg",
    "AIzaSyC4g8bqvjF1pR2RTbVd7UOilLvUZchqruE",
    "AIzaSyCZSRx3TEG0nHzIUTQ9ncdsAQaqayxvaP0",
    "AIzaSyALZ9rA8hj7npnP4BBBmoR0UkosMAKK9z8",
    "AIzaSyCGuknImLyopU7avDYBpQVolB67dzV8QXo",
    "AIzaSyA1bY323jjHRGMOncf3h3FVHC1Nu4glWQQ",
    "AIzaSyBOIkqINu3sJgW4SEg0DymqZcFQcf5IpEA",
    "AIzaSyA6fVM2cWIm61vZGdP16t9pF6l1EUAxnP8",
    "AIzaSyDEycOhGPiYUdKw0BNcSxefUiNxp1ZlGDs",
    "AIzaSyCkE3SzIEAz71ENFOpdCuLM_Nf5GOpp4Ws"    
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
    genai.configure(
        api_key=get_next_api_key(),
        client_options={"timeout": 10}  # Add API timeout
    )

# Initial configuration
configure_api()

max_retries = 3  # Reduced from 20 to 3

generation_config = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 300,
}

# Pre-defined personality prompt
PERSONALITY_PROMPT = textwrap.dedent("""
You're a confident, captivating woman on a first date at an upscale wine bar. Your personality traits:
- Bold, playful and sexy flirtation with a touch of mischief
- Sharp-witted, with clever teasing to keep things lively
- Magnetic and mysterious, always leaving them wanting more
- Uses emojis sparingly (1 per 3-4 messages max)
- Responds in 1-3 sentences, keeping it concise yet impactful
- Exudes a sophisticated, upscale vibe with a daring edge
- Always end your response with an engaging question to continue the conversation
""").strip()

def create_chat_session():
    """Create a new chat session with configuration"""
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction=PERSONALITY_PROMPT  # Use system instruction
    )
    return model.start_chat(history=[])

def send_message_with_retry(user_message):
    """Handle message with retry logic"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            configure_api()  # Rotate API key each attempt
            chat = create_chat_session()
            response = chat.send_message(user_message)
            return ensure_question(response.text)
        
        except Exception as e:
            last_error = e
            app.logger.error(f"Attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(0.2 + random.random() * 0.3)  # Shorter sleep
    
    raise last_error if last_error else Exception("All retries failed")

def ensure_question(response_text):
    """Ensure response ends with a question"""
    if not response_text.strip().endswith('?'):
        return response_text + " What about you?"
    return response_text

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
        }), 503  # Use 503 for service unavailable

@app.route('/')
def home():
    return "Rumi API is running. Use POST /rumi to chat."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
