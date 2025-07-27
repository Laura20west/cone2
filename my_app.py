from flask import Flask, request, jsonify
import google.generativeai as genai
import anthropic
import os
import textwrap
from dotenv import load_dotenv
import random
import time
import sys

load_dotenv()

app = Flask(__name__)

# API Keys Configuration
CLAUDE_API_KEYS = [
    "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",
    "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",
    "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"
]

GEMINI_API_KEYS = [
    "AIzaSyB3hpD5wU0dJX2qcqSemhv69DG9AI9YeH4",
"AIzaSyAbbWBNk2r-J3IHYplIPx8nmoH-czaoI3s"  # Replace with your actual API keys

]

# API Selection Tracking
current_api_type = "claude"  # Start with Claude
current_claude_key_index = 0
current_gemini_key_index = 0
max_retries = (len(CLAUDE_API_KEYS) + len(GEMINI_API_KEYS)) * 2  # Try each API twice

# Personality Prompt
PERSONALITY_PROMPT = textwrap.dedent("""
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

def get_next_api_config():
    """Rotate between Claude and Gemini APIs"""
    global current_api_type, current_claude_key_index, current_gemini_key_index
    
    if current_api_type == "claude":
        # Get next Claude key
        key = CLAUDE_API_KEYS[current_claude_key_index]
        current_claude_key_index = (current_claude_key_index + 1) % len(CLAUDE_API_KEYS)
        current_api_type = "gemini"  # Switch to Gemini next
        return {"type": "claude", "key": key}
    else:
        # Get next Gemini key
        key = GEMINI_API_KEYS[current_gemini_key_index]
        current_gemini_key_index = (current_gemini_key_index + 1) % len(GEMINI_API_KEYS)
        current_api_type = "claude"  # Switch to Claude next
        return {"type": "gemini", "key": key}

def initialize_claude_client(api_key):
    """Initialize Claude client with the given API key"""
    return anthropic.Client(api_key=api_key)

def initialize_gemini_client(api_key):
    """Configure Gemini with the given API key"""
    genai.configure(api_key=api_key)
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

def ensure_question(response_text):
    """Ensure the response ends with a question mark, modifying if needed"""
    if not response_text.strip().endswith('?'):
        questions = [
            "What about you?",
            "What do you think?",
            "I'm curious what's your take on this?",
            "Wouldn't you agree?",
            "Tell me more about yourself?",
            "What's your perspective on this?",
            "Care to share your thoughts?"
        ]
        return f"{response_text} {random.choice(questions)}"
    return response_text

def send_with_claude(client, message):
    """Send message using Claude API"""
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=300,
        system=PERSONALITY_PROMPT,
        messages=[
            {"role": "user", "content": message}
        ]
    )
    return ensure_question(response.content[0].text)

def send_with_gemini(model, message):
    """Send message using Gemini API"""
    chat = model.start_chat(history=[])
    chat.send_message(PERSONALITY_PROMPT)  # Set personality
    response = chat.send_message(message)
    return ensure_question(response.text)

def get_response(message):
    """Get response from either Claude or Gemini with retry logic"""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            api_config = get_next_api_config()
            
            if api_config["type"] == "claude":
                client = initialize_claude_client(api_config["key"])
                return send_with_claude(client, message)
            else:
                model = initialize_gemini_client(api_config["key"])
                return send_with_gemini(model, message)
                
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt + 1} with {api_config['type']} failed: {str(e)}", file=sys.stderr)
            if attempt < max_retries - 1:
                time.sleep(1 + random.random())  # Add jitter
    
    raise last_exception if last_exception else Exception("All API attempts failed")

@app.route('/rumi', methods=['POST'])
def rumi_endpoint():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
            
        response_text = get_response(user_message)
        
        return jsonify({
            "response": response_text,
            "status": "success",
            "api_used": current_api_type
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error",
            "message": "All API attempts failed"
        }), 500

@app.route('/')
def home():
    return "Rumi API is running. Use POST /rumi endpoint to chat."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
