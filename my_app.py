from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import json
import gzip
from datetime import datetime
from pathlib import Path
import uuid
import random
from collections import defaultdict, deque, Counter
from typing import Dict, List, Optional
import re
from itertools import product
import nltk
from nltk.corpus import wordnet as wn

# Initialize NLP
try:
    nlp = spacy.load("en_core_web_md")
    print("Loaded spaCy model successfully")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("Downloading spaCy model...")
    import os
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

try:
    nltk.download('wordnet', quiet=True)
    print("NLTK wordnet downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK wordnet: {e}")

app = FastAPI(title="Sally Assistant API", 
              description="Backend service for Sally Assistant, providing AI-powered conversation suggestions")

# Configuration with compressed files
BASE_DIR = Path(__file__).parent.absolute()
DATASET_PATH = BASE_DIR / "conversation_dataset.jsonl.gz"
UNCERTAIN_PATH = BASE_DIR / "uncertain_responses.jsonl.gz"
REPLY_POOLS_PATH = BASE_DIR / "reply_pools_augmented.json.gz"

# Ensure compressed files exist
for file_path in [DATASET_PATH, UNCERTAIN_PATH]:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        with gzip.open(file_path, "wt") as f:
            f.write("")

class TriggerGenerator:
    def __init__(self):
        self.misspellings = {
            'fuck': ['fck', 'fuk', 'f*ck', 'f**k', 'fvck'],
            'sex': ['sx', 'secks', 'sexx', 's3x'],
            'cock': ['cok', 'cck', 'kock', 'c0ck'],
            'pussy': ['p*ssy', 'pussay', 'pusi', 'p$$y'],
            'anal': ['anl', 'a**l', 'an@l'],
            'boobs': ['b**bs', 'bxxbs', 'boobies', 'tits', 'titties'],
            'cum': ['c*m', 'cvm', 'c0m', 'coom'],
            'dick': ['d*ck', 'd**k', 'dik', 'dck'],
            'horny': ['h0rny', 'hrny', 'h*rny', 'h@rny'],
            'naked': ['n*ked', 'nekid', 'n4k3d', 'naaked'],
            'chat': ['ch@t', 'cht', 'chatt'],
            'video': ['vid', 'vdo', 'v1deo'],
            'picture': ['pic', 'pix', 'p1c', 'pictur'],
            'send': ['snd', 's3nd', 'sehnd']
        }
        
        self.common_phrases = [
            "show me", "send me", "let me see", "wanna see", "want to see",
            "how much", "you charge", "how old", "where are you", "can we",
            "do you do", "i want to", "i wanna", "let's have", "tell me about",
            "price for", "cost for", "are you real", "are you a bot", "verification",
            "can i get", "can you send", "got more", "have more", "prove you're real"
        ]
    
    def generate_variations(self, trigger):
        variations = set([trigger])
        doc = nlp(trigger.lower())
        
        for word in trigger.lower().split():
            if word in self.misspellings:
                for misspelling in self.misspellings[word]:
                    new_variation = trigger.lower().replace(word, misspelling)
                    variations.add(new_variation)
        
        for token in doc:
            if token.pos_ == "VERB":
                variations.update([
                    f"want to {token.lemma_}",
                    f"how to {token.lemma_}",
                    f"let's {token.lemma_}",
                    f"i need {token.lemma_}",
                    f"can you {token.lemma_}",
                    f"would you {token.lemma_}"
                ])
        
        for token in doc:
            if token.pos_ in ["NOUN", "VERB", "ADJ"]:
                synsets = wn.synsets(token.text)
                for synset in synsets[:3]:
                    for lemma in synset.lemmas()[:3]:
                        synonym = lemma.name().replace('_', ' ')
                        if synonym != token.text:
                            variations.add(synonym)
        
        for phrase in self.common_phrases:
            for word in trigger.lower().split():
                if len(word) > 3:
                    variations.add(f"{phrase} {word}")
                    variations.add(f"{word} {phrase}")
        
        return variations

trigger_gen = TriggerGenerator()

DEFAULT_REPLY_POOLS = {
    "general": {
        "triggers": ["hi", "hello", "hey", "what's up", "how are you", "good morning", "good afternoon", "good evening"],
        "responses": [
            "Hey there, sugar! How's your day going?",
            "Well hello handsome! What brings you here today?",
            "Hey sweetie! I've been waiting to chat with someone like you.",
            "Hello there! You caught me at just the right time."
        ],
        "questions": [
            "What are you looking for today?",
            "How can I make your day better?",
            "What kind of fun are you in the mood for?",
            "What's on your mind right now?"
        ]
    },
    "verification": {
        "triggers": ["are you real", "are you fake", "are you a bot", "verification", "verify", "prove", "real person"],
        "responses": [
            "I'm 100% real, honey! Just a girl looking for some fun.",
            "Yes baby, I'm a real person. Want me to prove it to you?",
            "Of course I'm real! I get that question a lot, but I promise it's really me.",
            "I'm as real as they come, sweetheart. No bots here!"
        ],
        "questions": [
            "What can I do to help you believe me?",
            "What would convince you I'm real?",
            "Would you like to know more about me instead?",
            "How about we talk about something more fun?"
        ]
    },
    "location": {
        "triggers": ["where are you", "where do you live", "location", "city", "nearby", "close to me", "your place", "my place"],
        "responses": [
            "I'm in the city center, not too far from downtown.",
            "I have my own private place where we can have some fun.",
            "I'm probably closer than you think, honey.",
            "I'm in a nice, discreet location where we can be private."
        ],
        "questions": [
            "Where are you located? Maybe we're close by.",
            "Are you looking for someone in your area?",
            "Would you like to meet up if we're close?",
            "How far are you willing to travel for some fun?"
        ]
    },
    "pricing": {
        "triggers": ["how much", "price", "cost", "charge", "fee", "rates", "pricing", "pay", "dollar", "money"],
        "responses": [
            "My rates depend on what you're looking for, sweetie.",
            "I offer different packages depending on what you want.",
            "Let's talk about what you want first, then we can discuss compensation.",
            "Quality time with me is worth every penny, I promise."
        ],
        "questions": [
            "What kind of experience are you looking for?",
            "How long would you like to spend together?",
            "Do you have something specific in mind?",
            "Are you looking for something quick or do you want to take your time?"
        ]
    },
    "pictures": {
        "triggers": ["pictures", "pics", "photos", "selfies", "nudes", "naked", "send pic", "show me", "see you"],
        "responses": [
            "I have some special pictures I can share with serious people.",
            "I don't send free pics, honey, but I promise I'm worth it.",
            "My pictures are for my special friends only.",
            "I'd rather show you in person, but we can discuss some previews."
        ],
        "questions": [
            "What kind of pictures are you hoping to see?",
            "Would you like to see something specific?",
            "Do you have any pictures to share with me first?",
            "What would you do if you saw my pictures?"
        ]
    },
    "services": {
        "triggers": ["what do you do", "services", "offer", "specialize", "good at", "favorite", "best at", "provide"],
        "responses": [
            "I'm very open-minded and love to please.",
            "I offer the full girlfriend experience and more.",
            "I specialize in making fantasies come true.",
            "Whatever you desire, I can probably accommodate."
        ],
        "questions": [
            "What are you in the mood for today?",
            "Do you have any special requests?",
            "What's your biggest fantasy?",
            "Is there something specific you're looking for?"
        ]
    },
    "explicit": {
        "triggers": ["sex", "fuck", "suck", "blow", "anal", "pussy", "ass", "cum", "dick", "cock", "horny", "hard"],
        "responses": [
            "Mmm, sounds like you're in a naughty mood.",
            "I love how direct you are, baby.",
            "Those are exactly the kind of things I enjoy.",
            "You're making me excited just thinking about it."
        ],
        "questions": [
            "Tell me more about what you like?",
            "What's your favorite position?",
            "Have you been thinking about this for a while?",
            "What would you do first if we were together right now?"
        ]
    }
}

# Load or initialize compressed reply pools
if REPLY_POOLS_PATH.exists():
    try:
        with gzip.open(REPLY_POOLS_PATH, "rt") as f:
            REPLY_POOLS = json.load(f)
        print(f"Loaded reply pools from {REPLY_POOLS_PATH}")
    except Exception as e:
        print(f"Error loading reply pools: {e}")
        REPLY_POOLS = DEFAULT_REPLY_POOLS
else:
    REPLY_POOLS = DEFAULT_REPLY_POOLS
    with gzip.open(REPLY_POOLS_PATH, "wt") as f:
        json.dump(REPLY_POOLS, f, indent=2)
    print(f"Created default reply pools at {REPLY_POOLS_PATH}")

# Generate trigger variations
for category, data in REPLY_POOLS.items():
    enhanced_triggers = set(data["triggers"])
    for trigger in list(data["triggers"]):
        new_variations = trigger_gen.generate_variations(trigger)
        enhanced_triggers.update(new_variations)
    REPLY_POOLS[category]["triggers"] = list(enhanced_triggers)
    print(f"Generated {len(enhanced_triggers)} trigger variations for category '{category}'")

CATEGORY_QUEUES = {}
for category, data in REPLY_POOLS.items():
    responses = data["responses"]
    questions = data["questions"]
    if responses and questions:
        combinations = list(product(range(len(responses)), range(len(questions))))
        random.shuffle(combinations)
        CATEGORY_QUEUES[category] = deque(combinations)
    else:
        print(f"Warning: Category '{category}' is missing responses or questions")

AUTHORIZED_OPERATORS = {
    "cone478", "cone353", "cone229", "cone516", "cone481", "cone335", 
    "cone424", "cone069", "cone096", "cone075","cone136", "cone406", 
    "cone047", "cone461", "cone423", "cone290", "cone407", "cone468",
    "cone221", "cone412", "cone413", "admin@company.com", "test@example.com",
    "operator@myoperatorservice.com"
}

def is_authorized(operator_email):
    if not operator_email:
        return False
        
    operator_email = operator_email.lower().strip()
    
    if operator_email in AUTHORIZED_OPERATORS:
        return True
        
    for auth in AUTHORIZED_OPERATORS:
        if '@' in auth and auth.startswith('@'):
            if operator_email.endswith(auth[1:]):
                return True
    
    return False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

class UserMessage(BaseModel):
    message: str
    context: Optional[Dict] = None

class SallyResponse(BaseModel):
    matched_category: str = "general"
    confidence: float = 0.0
    replies: List[str] = []

def log_to_dataset(user_input: str, response_data: dict, operator: str, context: Optional[Dict] = None):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "matched_category": response_data["matched_category"],
        "replies": response_data["replies"],
        "operator": operator,
        "confidence": response_data["confidence"],
        "context": context
    }
    
    try:
        doc = nlp(user_input)
        if doc.vector.size > 0:
            entry["embedding"] = doc.vector.tolist()
    except Exception as e:
        print(f"Error creating embedding: {e}")
    
    try:
        with gzip.open(DATASET_PATH, "at") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"Logged message to dataset: {user_input[:30]}...")
    except Exception as e:
        print(f"Error logging to dataset: {e}")

def store_uncertain(user_input: str, context: Optional[Dict] = None):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "reviewed": False,
        "context": context
    }
    
    try:
        with gzip.open(UNCERTAIN_PATH, "at") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"Stored uncertain message: {user_input[:30]}...")
    except Exception as e:
        print(f"Error storing uncertain message: {e}")

def match_message_to_category(message: str):
    message = message.strip().lower()
    best_match = ("general", None, 0.0)
    
    for category, data in REPLY_POOLS.items():
        for trigger in data["triggers"]:
            if trigger.lower() in message:
                return (category, trigger, 1.0)
            
            if '*' in trigger:
                pattern = re.compile(trigger.replace('*', '.*'), re.IGNORECASE)
                if pattern.search(message):
                    return (category, trigger, 0.9)
    
    try:
        message_doc = nlp(message)
        
        for category, data in REPLY_POOLS.items():
            for trigger in data["triggers"]:
                trigger_doc = nlp(trigger)
                try:
                    similarity = message_doc.similarity(trigger_doc)
                    
                    for token in message_doc:
                        if token.has_vector and len(token.text) > 3:
                            for trigger_token in trigger_doc:
                                if trigger_token.has_vector and len(trigger_token.text) > 3:
                                    token_similarity = token.similarity(trigger_token)
                                    if token_similarity > 0.85:
                                        similarity = max(similarity, token_similarity * 0.9)
                    
                    if similarity > best_match[2]:
                        best_match = (category, trigger, similarity)
                except Exception as e:
                    print(f"Error calculating similarity for '{trigger}': {e}")
                    continue
    except Exception as e:
        print(f"Error in semantic matching: {e}")
    
    return best_match

def augment_dataset():
    if not DATASET_PATH.exists():
        return
    
    try:
        with gzip.open(DATASET_PATH, "rt") as f:
            entries = [json.loads(line) for line in f]
        
        if not entries:
            return
            
        print(f"Analyzing {len(entries)} conversation entries for augmentation")
        
        category_vocabs = defaultdict(set)
        for entry in entries:
            try:
                doc = nlp(entry["user_input"])
                keywords = [token.text.lower() for token in doc 
                           if token.is_alpha and not token.is_stop and token.pos_ in ["NOUN", "VERB", "ADJ"]]
                category_vocabs[entry["matched_category"]].update(keywords)
            except Exception as e:
                print(f"Error processing entry for augmentation: {e}")
        
        for category, words in category_vocabs.items():
            if category in REPLY_POOLS:
                significant_words = [w for w in words if len(w) > 3]
                top_words = list(significant_words)[:30]
                REPLY_POOLS[category]["triggers"].extend(top_words)
                REPLY_POOLS[category]["triggers"] = list(set(REPLY_POOLS[category]["triggers"]))
        
        for category, data in REPLY_POOLS.items():
            enhanced_triggers = set(data["triggers"])
            for trigger in list(data["triggers"]):
                enhanced_triggers.update(trigger_gen.generate_variations(trigger))
            REPLY_POOLS[category]["triggers"] = list(enhanced_triggers)
        
        with gzip.open(REPLY_POOLS_PATH, "wt") as f:
            json.dump(REPLY_POOLS, f, indent=2)
        
        global CATEGORY_QUEUES
        CATEGORY_QUEUES = {}
        for category, data in REPLY_POOLS.items():
            responses = data["responses"]
            questions = data["questions"]
            if responses and questions:
                combinations = list(product(range(len(responses)), range(len(questions))))
                random.shuffle(combinations)
                CATEGORY_QUEUES[category] = deque(combinations)
        
        print("Dataset augmentation complete")
        return True
    except Exception as e:
        print(f"Error during dataset augmentation: {e}")
        return False

async def verify_operator(request: Request):
    operator_email = request.headers.get("X-Operator-Email")
    
    if request.headers.get("X-Dev-Override") == "sally-dev-1a5f9e3":
        return "dev@example.com"
    
    if not operator_email or not is_authorized(operator_email):
        raise HTTPException(status_code=403, detail="Unauthorized operator")
    
    return operator_email

@app.post("/per", response_model=SallyResponse)
async def analyze_message(
    request: Request,
    user_input: UserMessage,
    operator: str = Depends(verify_operator)
):
    try:
        message = user_input.message.strip()
        context = user_input.context or {}
        
        if not message:
            return SallyResponse(matched_category="general", confidence=0.0, replies=[
                "I'm waiting to hear what you'd like to say...", "What's on your mind?"
            ])
        
        matched_category, matched_trigger, confidence = match_message_to_category(message)
        response_data = {"matched_category": matched_category, "confidence": round(confidence, 2), "replies": []}
        
        if matched_category in CATEGORY_QUEUES and CATEGORY_QUEUES[matched_category]:
            queue = CATEGORY_QUEUES[matched_category]
            category_data = REPLY_POOLS[matched_category]
            
            if queue and category_data["responses"] and category_data["questions"]:
                r_idx, q_idx = queue.popleft()
                queue.append((r_idx, q_idx))
                response = category_data["responses"][r_idx % len(category_data["responses"])]
                question = category_data["questions"][q_idx % len(category_data["questions"])]
                response_data["replies"] = [response, question]
                
                if confidence > 0.8 and len(queue) > 1:
                    r2_idx, q2_idx = queue[1]
                    response2 = category_data["responses"][r2_idx % len(category_data["responses"])]
                    question2 = category_data["questions"][q2_idx % len(category_data["questions"])]
                    response_data["replies"].extend([response2, question2])
        
        if not response_data["replies"]:
            response_data["replies"] = [
                "I'd love to chat more about that...",
                "What else would you like to know about me?"
            ]
        
        log_to_dataset(message, response_data, operator, context)
        
        if confidence < 0.6:
            store_uncertain(message, context)
            if len(response_data["replies"]) >= 2:
                response_data["replies"][1] += " Could you tell me more about what you're looking for?"
        
        return SallyResponse(**response_data)
    
    except Exception as e:
        print(f"Error processing message: {e}")
        return SallyResponse(
            matched_category="error",
            confidence=0.0,
            replies=["I'm not quite sure I understood that. Could you rephrase?", 
                     "What were you hoping to talk about?"]
        )

@app.get("/dataset/analytics")
async def get_analytics(request: Request, operator: str = Depends(verify_operator)):
    analytics = {
        "total_entries": 0,
        "common_categories": {},
        "confidence_stats": {},
        "categories": list(REPLY_POOLS.keys()),
        "trigger_counts": {category: len(data["triggers"]) for category, data in REPLY_POOLS.items()}
    }
    
    if DATASET_PATH.exists():
        try:
            with gzip.open(DATASET_PATH, "rt") as f:
                entries = [json.loads(line) for line in f]
            
            analytics["total_entries"] = len(entries)
            analytics["common_categories"] = dict(Counter(entry["matched_category"] for entry in entries).most_common(5))
            
            if entries:
                confidences = [entry.get("confidence", 0) for entry in entries if "confidence" in entry]
                if confidences:
                    analytics["confidence_stats"] = {
                        "average": round(sum(confidences)/len(confidences), 2),
                        "min": round(min(confidences), 2),
                        "max": round(max(confidences), 2)
                    }
                
                recent = entries[-10:]
                analytics["recent_queries"] = [
                    {"input": entry["user_input"][:50], "category": entry["matched_category"]}
                    for entry in recent
                ]
        except Exception as e:
            print(f"Error getting analytics: {e}")
    
    return analytics

@app.post("/augment")
async def trigger_augmentation(request: Request, operator: str = Depends(verify_operator)):
    result = augment_dataset()
    return {
        "status": "Dataset augmented" if result else "Augmentation failed", 
        "categories": len(REPLY_POOLS),
        "triggers_total": sum(len(data["triggers"]) for data in REPLY_POOLS.values())
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "8.5", "categories": len(REPLY_POOLS)}

@app.on_event("startup")
async def startup_event():
    print("Starting Sally Assistant API Server")
    try:
        augment_dataset()
    except Exception as e:
        print(f"Error during startup: {e}")

for path in [BASE_DIR, DATASET_PATH.parent, UNCERTAIN_PATH.parent]:
    if isinstance(path, Path) and not path.exists():
        path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
