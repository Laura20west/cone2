from flask import Flask, request, jsonify
import anthropic
import os
import textwrap
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configure Claude API
API_KEY = "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"
client = anthropic.Anthropic(api_key=API_KEY)

def setup_chat():
    system_prompt = textwrap.dedent("""
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
    
    return system_prompt

system_prompt = setup_chat()

def ensure_question(response_text):
    """Ensure the response ends with a question mark, modifying if needed"""
    if not response_text.strip().endswith('?'):
        # If no question exists, add a generic engaging question
        questions = [
            "What about you?",
            "What do you think?",
            "I'm curious what's your take on this?",
            "Wouldn't you agree?",
            "Tell me more about yourself?",
            "What's your perspective on this?",
            "Care to share your thoughts?"
        ]
        # Choose a random question or just pick the first one for simplicity
        return f"{response_text} {questions[0]}"
    return response_text

@app.route('/rumi', methods=['POST'])
def rumi_endpoint():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
            
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=300,
            temperature=1.0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        response_text = ensure_question(response.content[0].text)
        
        return jsonify({
            "response": response_text,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/')
def home():
    return "Rumi API is running. Use POST /rumi endpoint to chat."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
