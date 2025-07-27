from flask import Flask, request, jsonify
import requests
import json
import os

app = Flask(__name__)

# Configuration
OPENROUTER_API_KEY = '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'  # Will raise error if not set
MODEL = "anthropic/claude-3-haiku"
MAX_TOKENS = 70

def get_date_response(prompt):
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://your-render-app.onrender.com",  # Update with your actual URL
                "X-Title": "Rumi Date Simulator",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "Respond as a charming woman named Rumi on a first date. Use 1-2 short sentences max. Mix of playful, witty and genuinely interested. Never exceed 70 tokens."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.8
            }),
            timeout=10
        )
        
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
        
    except requests.exceptions.HTTPError as err:
        app.logger.error(f"HTTP Error: {err.response.status_code} - {err.response.text}")
        return None
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return None

@app.route('/rumi', methods=['POST'])
def rumi_endpoint():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'message' in request"}), 400
    
    response = get_date_response(data['message'])
    
    if response:
        return jsonify({"response": response})
    else:
        return jsonify({"error": "Failed to get response from AI. Check server logs."}), 502

@app.route('/')
def home():
    return "Rumi Date Simulator is running. POST to /rumi endpoint."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))  # Matches Render's expected port
