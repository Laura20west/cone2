from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import json
from datetime import datetime
from pathlib import Path
import uuid
import random
from collections import defaultdict, deque
import nltk
from nltk.corpus import wordnet as wn
from typing import Dict, List, Optional
import re
from itertools import product

# Initialize NLP
nlp = spacy.load("en_core_web_md")
nltk.download('wordnet')

app = FastAPI()

# Configuration
DATASET_PATH = Path("conversation_dataset.jsonl")
UNCERTAIN_PATH = Path("uncertain_responses.jsonl")
REPLY_POOLS_PATH = Path("reply_pools_augmented.json")

class TriggerGenerator:
    def __init__(self):
        self.misspellings = {
            'fuck': ['fck', 'fuk', 'f*ck', 'f**k', 'fvck'],
            'sex': ['sx', 'secks', 'sexx', 's3x'],
            'cock': ['cok', 'cck', 'kock', 'c0ck'],
            'pussy': ['p*ssy', 'pussay', 'pusi', 'p$$y'],
            'anal': ['anl', 'a**l', 'an@l'],
            'boobs': ['b**bs', 'bxxbs', 'boobies', 'tits', 'titties'],
        }
    
    def generate_variations(self, trigger):
        variations = set()
        doc = nlp(trigger)
        
        # Misspellings
        for word in trigger.split():
            if word.lower() in self.misspellings:
                variations.update(self.misspellings[word.lower()])
        
        # Phrasal patterns
        for token in doc:
            if token.pos_ == "VERB":
                variations.update([
                    f"want to {token.lemma_}",
                    f"how to {token.lemma_}",
                    f"let's {token.lemma_}",
                    f"I need {token.lemma_}"
                ])
        
        # Semantic variations
        for token in doc:
            if token.has_vector:
                similar_words = [
                    w.text for w in nlp.vocab 
                    if w.has_vector and token.similarity(w) > 0.6
                ]
                variations.update(similar_words)
        
        return variations

trigger_gen = TriggerGenerator()

# Load or initialize reply pools with auto-generation
if REPLY_POOLS_PATH.exists():
    with open(REPLY_POOLS_PATH, "r") as f:
        REPLY_POOLS = json.load(f)
    # Generate trigger variations
    for category, data in REPLY_POOLS.items():
        enhanced_triggers = set(data["triggers"])
        for trigger in data["triggers"]:
            enhanced_triggers.update(trigger_gen.generate_variations(trigger))
        REPLY_POOLS[category]["triggers"] = list(enhanced_triggers)
else:
    REPLY_POOLS = {
        "general": {
            "triggers": [],
            "responses": ["Honey, let's talk about something more exciting..."],
            "questions": ["What really gets you going?"]
        }
    }

# Initialize response queues
CATEGORY_QUEUES = {}
for category, data in REPLY_POOLS.items():
    responses = data["responses"]
    questions = data["questions"]
    combinations = list(product(range(len(responses)), range(len(questions))))
    random.shuffle(combinations)
    CATEGORY_QUEUES[category] = deque(combinations)

# Security config
AUTHORIZED_OPERATORS = {"cone478", "cone353", "cone229", "cone516", 
                       "cone481", "cone335", "cone424", "cone069", "cone096", 
                       "cone075","cone136", "cone406", "cone047", "cone461", 
                       "cone423", "cone290", "cone407", "cone468",
                       "cone221", "cone412", "cone413", "admin@company.com"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

class UserMessage(BaseModel):
    message: str

class SallyResponse(BaseModel):
    matched_word: str
    matched_category: str
    confidence: float
    response: str
    question: str

def log_to_dataset(user_input: str, response_data: dict, operator: str):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "matched_category": response_data["matched_category"],
        "response": response_data["response"],
        "question": response_data["question"],
        "operator": operator,
        "confidence": response_data["confidence"],
        "embedding": nlp(user_input).vector.tolist()
    }
    
    with open(DATASET_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

def store_uncertain(user_input: str):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "reviewed": False
    }
    
    with open(UNCERTAIN_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

def augment_dataset():
    if not DATASET_PATH.exists():
        return
    
    with open(DATASET_PATH, "r") as f:
        entries = [json.loads(line) for line in f]
    
    # Auto-discover new categories
    category_vocabs = defaultdict(set)
    for entry in entries:
        doc = nlp(entry["user_input"])
        category_vocabs[entry["matched_category"]].update(
            [token.text.lower() for token in doc if token.is_alpha]
        )
    
    # Create new categories
    for category, words in category_vocabs.items():
        if category not in REPLY_POOLS:
            REPLY_POOLS[category] = {
                "triggers": list(words),
                "responses": [],
                "questions": []
            }
    
    # Enhance triggers
    for category, data in REPLY_POOLS.items():
        new_triggers = set(data["triggers"])
        for trigger in data["triggers"]:
            new_triggers.update(trigger_gen.generate_variations(trigger))
        REPLY_POOLS[category]["triggers"] = list(new_triggers)
    
    with open(REPLY_POOLS_PATH, "w") as f:
        json.dump(REPLY_POOLS, f, indent=2)
    
    # Reinitialize queues
    global CATEGORY_QUEUES
    CATEGORY_QUEUES = {}
    for category, data in REPLY_POOLS.items():
        responses = data["responses"]
        questions = data["questions"]
        combinations = list(product(range(len(responses)), range(len(questions))))
        random.shuffle(combinations)
        CATEGORY_QUEUES[category] = deque(combinations)

async def verify_operator(request: Request):
    operator_email = request.headers.get("X-Operator-Email")
    if not operator_email or operator_email not in AUTHORIZED_OPERATORS:
        raise HTTPException(status_code=403, detail="Unauthorized operator")
    return operator_email

@app.post("/1A9I6F1O5R1C8O3N1E5145IA", response_model=SallyResponse)
async def analyze_message(
    request: Request,
    user_input: UserMessage,
    operator: str = Depends(verify_operator)
):  # <- Colon was missing here
    message = user_input.message.strip().lower()
    best_match = ("general", None, 0.0)
    
    for category, data in REPLY_POOLS.items():
        for trigger in data["triggers"]:
            # Handle wildcard patterns
            if '*' in trigger:
                pattern = re.compile(trigger.replace('*', '.*'), re.IGNORECASE)
                if pattern.fullmatch(message):
                    similarity = 1.0
            else:
                doc = nlp(message)
                trigger_doc = nlp(trigger)
                similarity = doc.similarity(trigger_doc)
            
            if similarity > best_match[2]:
                best_match = (category, trigger, similarity)
    
    response_data = {
        "matched_word": best_match[1] or "general",
        "matched_category": best_match[0],
        "confidence": round(best_match[2], 2),
        "response": "",
        "question": ""
    }
    
    # Get response pair
    category_data = REPLY_POOLS[best_match[0]]
    if category_data["responses"] and category_data["questions"]:
        queue = CATEGORY_QUEUES[best_match[0]]
        
        if queue:
            r_idx, q_idx = queue.popleft()
            response_data["response"] = category_data["responses"][r_idx]
            response_data["question"] = category_data["questions"][q_idx]
    
    # Fallback
    if not response_data["response"]:
        response_data.update({
            "response": "Honey, let's take this somewhere more private...",
            "question": "What's your deepest, darkest fantasy?"
        })
    
    log_to_dataset(message, response_data, operator)
    
    if response_data["confidence"] < 0.6:
        store_uncertain(message)
        response_data["question"] += " Could you rephrase that, baby?"
    
    return response_data

@app.get("/dataset/analytics")  # Properly aligned at app level
async def get_analytics():
    analytics = {
        "total_entries": 0,
        "common_categories": {},
        "confidence_stats": {}
    }
    
    if DATASET_PATH.exists():
        with open(DATASET_PATH, "r") as f:
            entries = [json.loads(line) for line in f]
        
        analytics["total_entries"] = len(entries)
        analytics["common_categories"] = Counter(entry["matched_category"] for entry in entries)
        
        if entries:
            confidences = [entry["confidence"] for entry in entries]
            analytics["confidence_stats"] = {
                "average": round(sum(confidences)/len(confidences), 2),
                "min": round(min(confidences), 2),
                "max": round(max(confidences), 2)
            }
    
    return analytics

@app.post("/augment")  # Properly aligned at app level
async def trigger_augmentation():
    augment_dataset()
    return {"status": "Dataset augmented", "new_pools": REPLY_POOLS}
