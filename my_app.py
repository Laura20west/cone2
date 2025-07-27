from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# Configuration
API_KEY = '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'
MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"

@app.route('/rumi', methods=['POST'])
def rumi_endpoint():
    try:
        data = request.json
        message = data.get('message', '')
        filtered_words = data.get('filteredWords', [])

        # Prepare the prompt for the AI
        system_prompt = """You are Rumi, an AI assistant. Respond to the user's message in a helpful, friendly manner. 
        Keep your responses concise and relevant. If any words were filtered (listed below), adapt your response accordingly.
        Filtered words: {filtered_words}""".format(filtered_words=", ".join(filtered_words))

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://myoperatorservice.com",
            "X-Title": "Rumi AI"
        }

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            return jsonify({"error": f"API error: {response.text}"}), 500

        ai_response = response.json()['choices'][0]['message']['content']
        return jsonify({"response": ai_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
