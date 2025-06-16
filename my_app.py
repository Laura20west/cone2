import json
import random
import re
import requests
import os
from flask import Flask, request, jsonify
from collections import defaultdict

app = Flask(__name__)

# Predefined keyword categories
CATEGORY_KEYWORDS = {
    "sex": ["fuck", "cock", "boobs", "pussy", "horny", "sex", "suck", "spank",
            "bondage", "threesome", "dick", "orgasm", "fucking", "nude", "naked",
            "blowjob", "handjob", "anal", "fetish", "kink", "sexy", "erotic", "masturbation"],
    "care": ["tease", "care", "day", "busy", "tired", "exhausted", "bed", "sleep",
            "rest"],
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
    lambda x: x.replace("toy", "plaything"),
    lambda x: x.replace("bad", "terrible"),
    lambda x: x.replace("big", "enormous"),
    lambda x: x.replace("small", "tiny"),
    lambda x: x.replace("fast", "rapid"),
    lambda x: x.replace("slow", "gradual"),
    lambda x: x.replace("easy", "simple"),
    lambda x: x.replace("hard", "difficult"),
    lambda x: x.replace("old", "ancient"),
    lambda x: x.replace("new", "fresh"),
    lambda x: x.replace("important", "crucial"),
    lambda x: x.replace("smart", "intelligent"),
    lambda x: x.replace("pretty", "beautiful"),
    lambda x: x.replace("ugly", "unattractive"),
    lambda x: x.replace("funny", "hilarious"),
    lambda x: x.replace("sad", "melancholy"),
    lambda x: x.replace("angry", "furious"),
    lambda x: x.replace("tired", "exhausted"),
    lambda x: x.replace("hungry", "starving"),
    lambda x: x.replace("cold", "freezing"),
    lambda x: x.replace("hot", "scorching"),
    lambda x: x.replace("right", "correct"),
    lambda x: x.replace("wrong", "incorrect"),
    lambda x: x.replace("sure", "certain"),
    lambda x: x.replace("maybe", "perhaps")
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
    """Manages blue messages with robust JSON handling and fallback to cone03.json"""
    def __init__(self, api_url="https://cone3.onrender.com/get_messages"):
        self.api_url = api_url
        self.blue_messages = defaultdict(list)
        self.all_blue_messages = []
        self.fallback_messages = []
        self.load_fallback_messages()
        self.load_messages()
    
    def load_fallback_messages(self):
        """Load messages from cone03.json fallback file"""
        try:
            if os.path.exists("cone03.json"):
                with open("cone03.json", "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.fallback_messages = [msg["content"] for msg in data if isinstance(msg, dict) and "content" in msg]
                        print(f"Loaded {len(self.fallback_messages)} fallback messages from cone03.json")
                    else:
                        print("cone03.json does not contain a list")
            else:
                print("cone03.json not found")
        except Exception as e:
            print(f"Error loading fallback messages: {e}")
    
    def load_messages(self):
        """Fetch and categorize blue messages with robust JSON parsing"""
        try:
            response = requests.get(self.api_url)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Extract messages list from response
            messages = data.get('messages', [])
            if not isinstance(messages, list):
                print(f"Invalid messages format: {type(messages)}")
                raise ValueError("'messages' is not a list")
            
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
            
            print(f"Loaded {len(self.all_blue_messages)} blue messages from API")
            
            # If we have less than 20 messages, add fallback messages
            if len(self.all_blue_messages) < 20 and self.fallback_messages:
                print(f"Adding {len(self.fallback_messages)} fallback messages (total API messages: {len(self.all_blue_messages)})")
                for msg in self.fallback_messages:
                    self.all_blue_messages.append(msg)
                    self.categorize_message(msg)
            
        except Exception as e:
            print(f"Error loading API messages: {e}")
            # Use fallback messages if API fails
            if self.fallback_messages:
                print(f"Using {len(self.fallback_messages)} fallback messages due to API error")
                for msg in self.fallback_messages:
                    self.all_blue_messages.append(msg)
                    self.categorize_message(msg)
            else:
                # Ultimate fallback if everything fails
                ultimate_fallback = [
                    "Will I be the first lady that you will use a toy on, or have you experienced that before?",
                    "What kind of playthings do you enjoy exploring with new partners?",
                    "Have you introduced toys in your previous intimate experiences?"
                ]
                for msg in ultimate_fallback:
                    self.all_blue_messages.append(msg)
                    self.categorize_message(msg)
    
    def categorize_message(self, content):
        """Categorize message based on keywords"""
        content_lower = content.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in content_lower for keyword in keywords):
                self.blue_messages[category].append(content)
    
    def get_context_match(self, user_input):
        """Find best matching blue message for user input with fallback to cone03.json"""
        # First try with the main messages
        match = self._get_match_from_messages(user_input, self.all_blue_messages)
        
        # If no good match found, try with fallback messages (if they weren't already included)
        if (not match or 
            (match and not self._has_keyword_match(user_input, match)) and self.fallback_messages:
            fallback_match = self._get_match_from_messages(user_input, self.fallback_messages)
            if fallback_match and self._has_keyword_match(user_input, fallback_match):
                return fallback_match
        
        return match or random.choice(self.all_blue_messages or self.fallback_messages)
    
    def _get_match_from_messages(self, user_input, messages):
        """Helper method to find best match from a given message list"""
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
                category_msgs = [msg for msg in messages if msg in self.blue_messages[top_category]]
                if category_msgs:
                    return random.choice(category_msgs)
        
        # Fallback to keyword similarity across all messages
        best_match = None
        best_score = 0
        user_words = set(re.findall(r'\w+', user_input_lower))
        
        for message in messages:
            msg_words = set(re.findall(r'\w+', message.lower()))
            overlap = user_words & msg_words
            score = len(overlap) / max(1, len(user_words))
            
            if score > best_score:
                best_score = score
                best_match = message
        
        return best_match
    
    def _has_keyword_match(self, user_input, message):
        """Check if message has at least one keyword match with user input"""
        user_input_lower = user_input.lower()
        message_lower = message.lower()
        
        # Check against all category keywords
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in user_input_lower and keyword in message_lower for keyword in keywords):
                return True
        
        return False

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
        "categorized": {k: len(v) for k, v in message_manager.blue_messages.items()},
        "fallback_messages_available": len(message_manager.fallback_messages)
    })

@app.route('/')
def health_check():
    return jsonify({
        "status": "active",
        "blue_messages_loaded": len(message_manager.all_blue_messages),
        "fallback_messages_available": len(message_manager.fallback_messages),
        "categories": list(CATEGORY_KEYWORDS.keys())
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
