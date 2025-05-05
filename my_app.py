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
import gzip
import orjson  # Faster JSON serialization/deserialization
from functools import lru_cache

# Initialize NLP
nlp = spacy.load("en_core_web_md")
nltk.download('wordnet', quiet=True)

app = FastAPI()

# Configuration
DATASET_PATH = Path("conversation_dataset.jsonl.gz")
UNCERTAIN_PATH = Path("uncertain_responses.jsonl.gz")
REPLY_POOLS_PATH = Path("reply_pools_augmented.json.gz")


# Load or initialize reply pools with compression
def load_compressed_json(filepath):
    if filepath.exists():
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                return orjson.loads(f.read())
        except Exception as e:
            print(f"Error loading compressed file {filepath}: {e}")
            return {}
    return {}

def save_compressed_json(data, filepath):
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        f.write(orjson.dumps(data).decode('utf-8'))

# Load reply pools
REPLY_POOLS = load_compressed_json(REPLY_POOLS_PATH)
if not REPLY_POOLS:
    REPLY_POOLS = {
        "general": {
            "triggers": [],
            "responses": ["Honey, let's talk about something more exciting..."],
            "questions": ["What really gets you going?"]
        }
    }
    save_compressed_json(REPLY_POOLS, REPLY_POOLS_PATH)
else:
    # Ensure all categories have required fields
    for category in REPLY_POOLS.values():
        category.setdefault("triggers", [])
        category.setdefault("responses", [])
        category.setdefault("questions", [])

# Initialize response queues
CATEGORY_QUEUES = {}
for category, data in REPLY_POOLS.items():
    responses = data["responses"]
    questions = data["questions"]
    combinations = [(r_idx, q_idx) for r_idx in range(len(responses)) 
                   for q_idx in range(len(questions))]
    random.shuffle(combinations)
    CATEGORY_QUEUES[category] = deque(combinations)

# Security config - use a set for faster lookups
AUTHORIZED_OPERATORS = {
    "cone478", "cone353", "cone229", "cone516", 
    "cone481", "cone335", "cone424", "cone069", "cone096", 
    "cone075", "cone136", "cone406", "cone047", "cone461", 
    "cone423", "cone290", "cone407", "cone468",
    "cone221", "cone412", "cone413", "admin@company.com"
}

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
    matched_word: str
    matched_category: str
    confidence: float
    replies: List[str]

def log_to_dataset(user_input: str, response_data: dict, operator: str):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "matched_category": response_data["matched_category"],
        "response": response_data["replies"][0] if response_data["replies"] else None,
        "question": response_data["replies"][1] if len(response_data["replies"]) > 1 else None,
        "operator": operator,
        "confidence": response_data["confidence"],
        # Store minimal embedding information to save space
        "embedding": [round(x, 4) for x in nlp(user_input).vector.tolist()][:50]  # Truncate and round
    }
    
    # Append to compressed jsonl file
    with gzip.open(DATASET_PATH, "ab") as f:
        f.write(orjson.dumps(entry) + b"\n")

def store_uncertain(user_input: str):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "reviewed": False
    }
    
    with gzip.open(UNCERTAIN_PATH, "ab") as f:
        f.write(orjson.dumps(entry) + b"\n")

@lru_cache(maxsize=128)
def get_lemmatized_text(text):
    """Cache lemmatization results to save processing time"""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def augment_dataset():
    entries = []
    if DATASET_PATH.exists():
        try:
            with gzip.open(DATASET_PATH, "rt", encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entries.append(orjson.loads(line))
        except Exception as e:
            print(f"Error reading dataset: {e}")
    
    category_counts = defaultdict(int)
    for entry in entries:
        category_counts[entry["matched_category"]] += 1
    
    avg_count = sum(category_counts.values()) / len(category_counts) if category_counts else 0
    needs_augmentation = [k for k, v in category_counts.items() if v < avg_count * 0.5]
    
    for category in needs_augmentation:
        if category not in REPLY_POOLS:
            continue
            
        base_triggers = REPLY_POOLS[category]["triggers"]
        new_triggers = []
        
        for trigger in base_triggers:
            lemmatized = get_lemmatized_text(trigger)
            new_triggers.append(lemmatized)
            
            doc = nlp(trigger)
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    syns = [syn.lemmas()[0].name() for syn in wn.synsets(token.text)]
                    if syns:
                        new_triggers.append(trigger.replace(token.text, syns[0]))
        
        REPLY_POOLS[category]["triggers"] = list(set(REPLY_POOLS[category]["triggers"] + new_triggers))
    
    # Save compressed reply pools
    save_compressed_json(REPLY_POOLS, REPLY_POOLS_PATH)
    
    # Reinitialize queues after augmentation
    global CATEGORY_QUEUES
    CATEGORY_QUEUES = {}
    for category, data in REPLY_POOLS.items():
        responses = data["responses"]
        questions = data["questions"]
        combinations = [(r_idx, q_idx) for r_idx in range(len(responses)) 
                       for q_idx in range(len(questions))]
        random.shuffle(combinations)
        CATEGORY_QUEUES[category] = deque(combinations)

async def verify_operator(request: Request):
    operator_email = request.headers.get("X-Operator-Email", "").lower()
    if not operator_email or operator_email not in AUTHORIZED_OPERATORS:
        raise HTTPException(status_code=403, detail="Unauthorized operator")
    return operator_email

@app.post("/1A9I6F1O5R1C8O3N1E5145ID", response_model=SallyResponse)
async def analyze_message(
    request: Request,
    user_input: UserMessage,
    operator: str = Depends(verify_operator)
):
    message = user_input.message.strip()
    doc = nlp(message.lower())
    
    best_match = ("general", None, 0.0)
    
    # Enhanced matching with fallback
    for category, data in REPLY_POOLS.items():
        for trigger in data["triggers"]:
            trigger_doc = nlp(trigger)
            similarity = doc.similarity(trigger_doc)
            if similarity > best_match[2]:
                best_match = (category, trigger, similarity)
    
    # Word-based fallback
    if best_match[2] < 0.7:
        for token in doc:
            for category, data in REPLY_POOLS.items():
                if token.text in data["triggers"]:
                    best_match = (category, token.text, 0.8)
                    break
            if best_match[2] >= 0.7:
                break
    
    # Prepare response
    response = {
        "matched_word": best_match[1] or "general",
        "matched_category": best_match[0],
        "confidence": round(best_match[2], 2),
        "replies": []
    }
    
    # Get non-repeating response pair
    category_data = REPLY_POOLS[best_match[0]]
    if category_data["responses"] and category_data["questions"]:
        queue = CATEGORY_QUEUES[best_match[0]]
        
        if not queue:
            # Regenerate combinations if queue is empty
            combinations = [(r_idx, q_idx) for r_idx in range(len(category_data["responses"]))
                           for q_idx in range(len(category_data["questions"]))]
            random.shuffle(combinations)
            queue = deque(combinations)
            CATEGORY_QUEUES[best_match[0]] = queue

        if queue:
            taken = 0
            while taken < 2 and queue:
                r_idx, q_idx = queue.popleft()
                response["replies"].append(category_data["responses"][r_idx])
                response["replies"].append(category_data["questions"][q_idx])
                taken += 1
    
    # Fallback if no responses found
    if not response["replies"]:
        response["replies"] = [
            "Honey, let's take this somewhere more private...",
            "What's your deepest, darkest fantasy?"
        ]
    
    # Log interaction
    log_to_dataset(message, response, operator)
    
    # Active learning
    if response["confidence"] < 0.6:
        store_uncertain(message)
        if len(response["replies"]) > 1:
            response["replies"][1] += " Could you rephrase that, baby?"
    
    return response

@app.get("/dataset/analytics")
async def get_analytics():
    from collections import Counter
    
    analytics = {
        "total_entries": 0,
        "common_categories": {},
        "confidence_stats": {}
    }
    
    entries = []
    if DATASET_PATH.exists():
        try:
            with gzip.open(DATASET_PATH, "rt", encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entries.append(orjson.loads(line))
        except Exception as e:
            print(f"Error reading dataset for analytics: {e}")
    
    if entries:
        analytics["total_entries"] = len(entries)
        analytics["common_categories"] = dict(Counter(entry["matched_category"] for entry in entries))
        
        confidences = [entry["confidence"] for entry in entries]
        analytics["confidence_stats"] = {
            "average": round(sum(confidences)/len(confidences), 2),
            "min": round(min(confidences), 2),
            "max": round(max(confidences), 2)
        }
    
    return analytics

@app.post("/augment")
async def trigger_augmentation():
    augment_dataset()
    return {"status": "Dataset augmented", "new_pools": REPLY_POOLS}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
