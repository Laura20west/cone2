from flask import Flask, request, jsonify
import requests
import json
import os

app = Flask(__name__)

# Configuration
OPENROUTER_API_KEY = ('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 ')  # Always use environment variables
MODEL = "anthropic/claude-3-haiku"
MAX_TOKENS = 70

def get_date_response(prompt):
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": request.host_url,
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
        
        # Print the full API response for debugging
        app.logger.info(f"OpenRouter API response: {response.text}")
        
        if response.status_code != 200:
            error_msg = f"API Error {response.status_code}"
            if response.text:
                error_data = response.json()
                error_msg += f": {error_data.get('error', {}).get('message', 'Unknown error')}"
            app.logger.error(error_msg)
            return None
            
        return response.json()['choices'][0]['message']['content']
        
    except Exception as e:
        app.logger.error(f"Request failed: {str(e)}")
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
        return jsonify({
            "response": response,
            "model": MODEL,
            "tokens": MAX_TOKENS
        })
    else:
        return jsonify({
            "error": "Failed to get response from AI",
            "solution": "Check server logs for details",
            "note": "This might be due to invalid API key or insufficient credits"
        }), 502

@app.route('/')
def home():
    return """
    <h1>Rumi Date Simulator API</h1>
    <p>Send POST requests to <code>/rumi</code> with JSON payload:</p>
    <pre>{"message": "Your question here"}</pre>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
