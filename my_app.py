import json
import random
import re
import requests
from flask import Flask, request, jsonify
from collections import defaultdict

app = Flask(__name__)

# Predefined keyword categories
CATEGORY_KEYWORDS = {
    "sex": ["fuck", "cock", "boobs", "pussy", "horny", "sex", "suck", "spank",
            "bondage", "threesome", "dick", "orgasm", "fucking", "nude", "naked",
            "blowjob", "handjob", "anal", "fetish", "kink", "sexy", "erotic", "masturbation"],
    "cars": ["car", "vehicle", "drive", "driving", "engine", "tire", "race", "speed",
             "motor", "wheel", "road", "highway", "license", "driver", "automobile"],
    "age": ["age", "old", "young", "birthday", "years", "aged", "elderly", "youth",
            "minor", "teen", "teenager", "adult", "senior", "centenarian"],
    "hobbies": ["toy", "fun", "hobbies", "game", "play", "playing", "collect",
                "activity", "leisure", "pastime", "sport", "craft", "art", "music", "reading"],
    "relationships": ["date", "dating", "partner", "boyfriend", "girlfriend",
                      "marriage", "marry", "crush", "love", "kiss", "romance",
                      "affection", "commitment", "proposal", "engagement"]
}

# Paraphrasing templates
PARAPHRASE_TEMPLATES = [
    lambda x: x.replace("good", "excellent"),
    lambda x: x.replace("good", "wonderful"),
    lambda x: x.replace("great", "amazing"),
    lambda x: x.replace("nice", "lovely"),
    lambda x: x.replace("happy", "delighted"),
    lambda x: x.replace("excited", "enthusiastic"),
    lambda x: re.sub(r"It's (.+)", r"The fact is that it's \1", x),
    lambda x: re.sub(r"That's (.+)", r"The reality is that it's \1", x),
    lambda x: re.sub(r"This is (.+)", r"What we have here is \1", x),
    lambda x: re.sub(r'\bis\b', 'appears to be', x),
    lambda x: re.sub(r'\bwas\b', 'seemed to be', x),
    lambda x: re.sub(r'\bcan\b', 'might be able to', x),
    lambda x: x.replace("love", "adore"),
    lambda x: x.replace("like", "enjoy"),
    lambda x: x.replace("want", "desire"),
    lambda x: x.replace("stuff", "things"),
    lambda x: x.replace("guy", "person"),
    lambda x: x.replace("kid", "child"),
    lambda x: re.sub(r"What is (.+)?", r"Can you tell me what \1 is?", x),
    lambda x: re.sub(r"How do (.+)?", r"What's the method for \1?", x),
    lambda x: re.sub(r"Why (.+)?", r"What's the reason that \1?", x),
    lambda x: x.replace("first", "initial"),
    lambda x: x.replace("before", "previously"),
    lambda x: x.replace("use", "utilize"),
    lambda x: x.replace("experienced", "encountered"),
    lambda x: x.replace("lady", "woman"),
    lambda x: x.replace("toy", "plaything")
]

class ContextValidator:
    """Validates if response matches user context"""
    @staticmethod
    def validate(user_input, response):
        """Check if response shares keywords with user input"""
        user_words = set(re.findall(r'\w+', user_input.lower()))
        response_words = set(re.findall(r'\w+', response.lower()))
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                      "have", "has", "had", "do", "does", "did", "will", "would", "should",
                      "can", "could", "may", "might", "must", "shall", "i", "you", "he", "she",
                      "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
                      "its", "our", "their", "mine", "yours", "hers", "ours", "theirs", "this",
                      "that", "these", "those", "am", "to", "of", "in", "for", "on", "with", "at"}
        
        user_keywords = user_words - stop_words
        response_keywords = response_words - stop_words
        
        # Calculate keyword overlap
        overlap = user_keywords & response_keywords
        match_score = len(overlap) / max(1, len(user_keywords))
        
        return match_score > 0.3, match_score

class BlueMessageManager:
    """Manages blue messages with robust JSON handling"""
    def __init__(self, api_url="https://cone3.onrender.com/get_messages"):
        self.api_url = api_url
        self.blue_messages = defaultdict(list)
        self.all_blue_messages = []
        self.load_messages()
    
    def load_messages(self):
        """Fetch and categorize blue messages with robust JSON parsing"""
        try:
            response = requests.get(self.api_url)
            response.raise_for_status()
            
            # Handle potential string responses
            raw_data = response.text
            if isinstance(raw_data, str):
                try:
                    messages = json.loads(raw_data)
                except json.JSONDecodeError:
                    print("API returned invalid JSON string")
                    messages = None
            else:
                messages = response.json()

            # Validate message format
            if not messages or not isinstance(messages, list):
                print(f"Invalid API response format: {type(messages)}")
                raise ValueError("API response is not a list")

            for msg in messages:
                # Skip non-dictionary items
                if not isinstance(msg, dict):
                    print(f"Skipping non-dict message: {msg}")
                    continue
                    
                # Process valid blue messages
                if msg.get('bubble_color') == 'blue' and msg.get('content'):
                    content = msg['content']
                    self.all_blue_messages.append(content)
                    self.categorize_message(content)
            
            print(f"Loaded {len(self.all_blue_messages)} blue messages")
            
        except Exception as e:
            print(f"Error loading messages: {e}")
            # Use fallback messages
            fallback_messages = [
                "Will I be the first lady that you will use a toy on, or have you experienced that before?",
                "What kind of playthings do you enjoy exploring with new partners?",
                "Have you introduced toys in your previous intimate experiences?"
            ]
            for msg in fallback_messages:
                self.all_blue_messages.append(msg)
                self.categorize_message(msg)
    
    def categorize_message(self, content):
        """Categorize message based on keywords"""
        content_lower = content.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in content_lower for keyword in keywords):
                self.blue_messages[category].append(content)
    
    def get_context_match(self, user_input):
        """Find best matching blue message for user input"""
        # First, try category-based matching
        user_input_lower = user_input.lower()
        matched_categories = []
        
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in user_input_lower for keyword in keywords):
                matched_categories.append(category)
        
        # If we found matching categories, select from those
        if matched_categories:
            # Prioritize categories with most keyword matches
            category_scores = []
            for category in matched_categories:
                keywords = CATEGORY_KEYWORDS[category]
                score = sum(1 for keyword in keywords if keyword in user_input_lower)
                category_scores.append((category, score))
            
            # Sort by score descending
            category_scores.sort(key=lambda x: x[1], reverse=True)
            top_category = category_scores[0][0]
            
            if self.blue_messages[top_category]:
                return random.choice(self.blue_messages[top_category])
        
        # Fallback to keyword similarity across all messages
        best_match = None
        best_score = 0
        user_words = set(re.findall(r'\w+', user_input_lower))
        
        for message in self.all_blue_messages:
            msg_words = set(re.findall(r'\w+', message.lower()))
            overlap = user_words & msg_words
            score = len(overlap) / max(1, len(user_words))
            
            if score > best_score:
                best_score = score
                best_match = message
        
        return best_match or random.choice(self.all_blue_messages)

class Paraphraser:
    """Handles message paraphrasing"""
    def __init__(self):
        self.templates = PARAPHRASE_TEMPLATES
    
    def paraphrase(self, text, iterations=3):
        """Apply paraphrasing transformations"""
        if not text.strip():
            return text
        
        for _ in range(iterations):
            template = random.choice(self.templates)
            try:
                new_text = template(text)
                if new_text != text:
                    text = new_text
            except:
                continue
        
        return text

# Initialize components
message_manager = BlueMessageManager()
paraphraser = Paraphraser()
context_validator = ContextValidator()

@app.route('/chat', methods=['POST'])
def chat_handler():
    """Main chat endpoint with context validation"""
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({"error": "Empty message"}), 400
        
        # Get context-matched blue message
        blue_message = message_manager.get_context_match(user_input)
        
        # Paraphrase the response
        paraphrased = paraphraser.paraphrase(blue_message)
        
        # Validate context match
        is_context_match, match_score = context_validator.validate(user_input, paraphrased)
        
        return jsonify({
            "response": paraphrased,
            "original_blue_message": blue_message,
            "context_match": is_context_match,
            "match_score": round(match_score, 2),
            "user_input": user_input
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/blue_messages', methods=['GET'])
def get_blue_messages():
    """Endpoint to view loaded blue messages"""
    return jsonify({
        "count": len(message_manager.all_blue_messages),
        "messages": message_manager.all_blue_messages,
        "categorized": {k: len(v) for k, v in message_manager.blue_messages.items()}
    })

@app.route('/')
def health_check():
    return jsonify({
        "status": "active",
        "blue_messages_loaded": len(message_manager.all_blue_messages),
        "categories": list(CATEGORY_KEYWORDS.keys())
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
