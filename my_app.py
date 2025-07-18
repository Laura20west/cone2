import json
import random
import re
import requests
import os
from datetime import datetime, timedelta
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

# Enhanced paraphrasing templates
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
    lambda x: x.replace("maybe", "perhaps"),
    lambda x: re.sub(r"\b(\w+)ed\b", lambda m: m.group(1) + "d" if random.random() > 0.5 else m.group(0), x),
    lambda x: x + " you know" if random.random() > 0.7 else x,
    lambda x: x.replace("I", "one") if random.random() > 0.6 else x,
    lambda x: re.sub(r"\bthe\b", "", x) if random.random() > 0.4 else x,
    lambda x: x.replace("you", "someone") if random.random() > 0.5 else x,
    lambda x: re.sub(r"\.$", "?", x) if random.random() > 0.3 else x,
    lambda x: re.sub(r"\b(\w+)ing\b", lambda m: m.group(1) + "in'" if random.random() > 0.4 else m.group(0), x),
    lambda x: "Well, " + x[0].lower() + x[1:] if not x.startswith("Well") and random.random() > 0.5 else x,
    lambda x: "Actually, " + x if not x.startswith("Actually") and random.random() > 0.4 else x,
    lambda x: re.sub(r"\b(\w+)\b", lambda m: m.group(1).upper() if random.random() > 0.8 else m.group(1), x)
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
    """Manages blue messages with caching and fallback"""
    def __init__(self, api_url="https://cone3.onrender.com/get_messages"):
        self.api_url = api_url
        self.blue_messages = defaultdict(list)
        self.all_blue_messages = []
        self.fallback_messages = []
        self.last_loaded = None
        self.cache_expiry = timedelta(minutes=5)  # Cache for 5 minutes
        self._load_fallback_messages()
        self._load_api_messages()
    
    def _load_fallback_messages(self):
        """Load messages from local fallback file"""
        try:
            if os.path.exists("cone03.json"):
                with open("cone03.json", "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.fallback_messages = [msg["content"] for msg in data if isinstance(msg, dict) and "content" in msg]
                        print(f"Loaded {len(self.fallback_messages)} fallback messages")
        except Exception as e:
            print(f"Error loading fallback: {str(e)}")
            self.fallback_messages = [
                "What kind of experiences are you curious about?",
                "How do you feel about trying something new?",
                "Tell me more about your interests."
            ]
    
    def _load_api_messages(self):
        """Fetch messages from API with proper error handling"""
        try:
            response = requests.get(self.api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            messages = data.get('messages', [])
            
            if not isinstance(messages, list):
                raise ValueError("API response is not a list")
            
            self.all_blue_messages = []
            self.blue_messages = defaultdict(list)
            
            for msg in messages:
                if isinstance(msg, dict) and msg.get('bubble_color') == 'blue' and msg.get('content'):
                    content = msg['content']
                    self.all_blue_messages.append(content)
                    self._categorize_message(content)
            
            print(f"Successfully loaded {len(self.all_blue_messages)} messages from API")
            self.last_loaded = datetime.now()
            return True
            
        except Exception as e:
            print(f"API load failed: {str(e)}")
            # Use fallback if API fails
            if self.fallback_messages:
                print("Using fallback messages")
                self.all_blue_messages = self.fallback_messages.copy()
                for msg in self.fallback_messages:
                    self._categorize_message(msg)
            return False
    
    def _categorize_message(self, content):
        """Categorize message based on keywords"""
        content_lower = content.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in content_lower for keyword in keywords):
                self.blue_messages[category].append(content)
    
    def reload_messages_if_needed(self):
        """Reload messages only if cache expired"""
        if not self.last_loaded or (datetime.now() - self.last_loaded) > self.cache_expiry:
            print("Cache expired, reloading messages...")
            return self._load_api_messages()
        return False
    
    def get_context_match(self, user_input):
        """Find best matching blue message for user input"""
        if not self.all_blue_messages:
            return random.choice(self.fallback_messages)
        
        # Try category-based matching first
        user_input_lower = user_input.lower()
        matched_categories = []
        
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in user_input_lower for keyword in keywords):
                matched_categories.append(category)
        
        if matched_categories:
            selected_category = random.choice(matched_categories)
            if self.blue_messages[selected_category]:
                return random.choice(self.blue_messages[selected_category])
        
        # Fallback to keyword similarity
        user_words = set(re.findall(r'\w+', user_input_lower))
        best_match = None
        best_score = 0
        
        for message in self.all_blue_messages:
            msg_words = set(re.findall(r'\w+', message.lower()))
            common = user_words & msg_words
            score = len(common)
            
            if score > best_score:
                best_score = score
                best_match = message
        
        return best_match or random.choice(self.all_blue_messages)

class Paraphraser:
    """Handles message paraphrasing with variation tracking"""
    def __init__(self):
        self.templates = PARAPHRASE_TEMPLATES
        self.recent_phrases = set()
        self.max_recent = 20
    
    def paraphrase(self, text):
        """Apply multiple paraphrasing transformations"""
        if not text.strip():
            return text
        
        # Generate multiple variations
        variations = []
        for _ in range(3):  # Generate 3 variations
            current = text
            # Apply 2-4 random transformations
            for _ in range(random.randint(2, 4)):
                template = random.choice(self.templates)
                try:
                    transformed = template(current)
                    if transformed != current:
                        current = transformed
                except:
                    continue
            variations.append(current)
        
        # Select the most unique variation not recently used
        for variation in sorted(variations, key=lambda x: self._uniqueness_score(text, x), reverse=True):
            if variation not in self.recent_phrases:
                self._track_phrase(variation)
                return variation
        
        # If all variations are recent, return the most different one
        best_variation = max(variations, key=lambda x: self._uniqueness_score(text, x))
        self._track_phrase(best_variation)
        return best_variation
    
    def _uniqueness_score(self, original, variation):
        """Calculate how different variation is from original"""
        orig_words = set(re.findall(r'\w+', original.lower()))
        var_words = set(re.findall(r'\w+', variation.lower()))
        return len(var_words - orig_words)
    
    def _track_phrase(self, phrase):
        """Track recent phrases to avoid repetition"""
        self.recent_phrases.add(phrase)
        if len(self.recent_phrases) > self.max_recent:
            self.recent_phrases.pop()

# Initialize components
message_manager = BlueMessageManager()
paraphraser = Paraphraser()
context_validator = ContextValidator()

@app.route('/sinners', methods=['POST'])
def chat_handler():
    """Main chat endpoint with caching and fallback"""
    try:
        # Only reload messages if cache expired
        message_manager.reload_messages_if_needed()
        
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
            "original": blue_message,
            "context_match": is_context_match,
            "match_score": round(match_score, 2),
            "user_input": user_input,
            "cache_status": "fresh" if message_manager.last_loaded and 
                            (datetime.now() - message_manager.last_loaded) < message_manager.cache_expiry 
                            else "reloaded"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/blue_messages', methods=['GET'])
def get_blue_messages():
    """Endpoint to view loaded blue messages"""
    return jsonify({
        "count": len(message_manager.all_blue_messages),
        "categories": {k: len(v) for k, v in message_manager.blue_messages.items()},
        "last_loaded": message_manager.last_loaded.isoformat() if message_manager.last_loaded else None,
        "using_fallback": len(message_manager.all_blue_messages) == len(message_manager.fallback_messages)
    })

@app.route('/')
def health_check():
    return jsonify({
        "status": "active",
        "message": "Sinner's endpoint is ready",
        "paraphrase_templates": len(PARAPHRASE_TEMPLATES),
        "cached_messages": len(message_manager.all_blue_messages),
        "cache_expiry_minutes": message_manager.cache_expiry.total_seconds() / 60
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
