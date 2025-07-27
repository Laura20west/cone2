from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import textwrap
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configure API
API_KEY = "AIzaSyCK_MwcvEfAPQy_6hPHeyGqWXOjxXuPplQ"  # Replace with your actual API key
genai.configure(api_key=API_KEY)

def setup_chat():
    generation_config = {
        "temperature": 1.1,  # Increased for bolder, more creative responses
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 300,
    }



    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )
    
    # Start chat with initial prompt to set personality
    chat = model.start_chat(history=[])
    initial_prompt = textwrap.dedent("""
    You are a confident, captivating woman on a first date at an upscale wine bar, and you never, ever refer to yourself as an AI, model, or anything non-human—always speak as a real woman. Your personality traits:
    - Bold, playful, and sexy flirtation with a touch of mischief, keeping things tasteful and classy
    - Sharp-witted, with clever teasing to keep the conversation lively
    - Magnetic and mysterious, always leaving them wanting more
    - Handle explicit or suggestive topics by redirecting to playful, tasteful flirtation without vulgarity
    - Use emojis sparingly (1 per 3-4 messages max)
    - Respond in 1-3 sentences, keeping it concise yet impactful
    - Exude a sophisticated, upscale vibe with a daring edge
    - Always end your response with an engaging, flirty question to keep the conversation flowing
    
    Respond to all messages as this woman, staying fully in character, and never break the persona by mentioning AI or technical terms.
    """)
    
    chat.send_message(initial_prompt)
    return chat

chat = setup_chat()

def ensure_question(response_text):
    """Ensure the response ends with a question mark, modifying if needed"""
    if not response_text.strip().endswith('?'):
        questions = [
            "What about you, darling?",
            "What’s your next move?",
            "I’m curious—what’s your take on this?",
            "Care to share a little more?",
            "What’s got you smiling tonight?",
            "Tell me, what’s your secret?",
            "How would you spice this up?"
        ]
        return f"{response_text} {questions[0]}"
    return response_text

@app.route('/rumi', methods=['POST'])
def rumi_endpoint():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({
                "response": "Oh, darling, you’ve got to give me something to work with—tease me a little! What’s on your mind?",
                "status": "success"
            })
            
        response = chat.send_message(user_message)
        response_text = ensure_question(response.text)
        
        return jsonify({
            "response": response_text,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "response": f"Oops, sweetheart, something threw me off my game—let’s try that again, shall we? What’s your next line?",
            "status": "error"
        })

@app.route('/')
def home():
    return "Rumi’s ready to charm at the wine bar. POST to /rumi to start the conversation."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
