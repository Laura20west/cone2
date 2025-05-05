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
import nltk
from nltk.corpus import wordnet as wn
from typing import Dict, List, Optional
from textblob import TextBlob
import re
from itertools import product

# Initialize NLP
nlp = spacy.load("en_core_web_md")
nltk.download('wordnet')

app = FastAPI()

# Configuration with GZIP compression
DATASET_PATH = Path("conversation_dataset.jsonl.gz")
UNCERTAIN_PATH = Path("uncertain_responses.jsonl.gz")
REPLY_POOLS_PATH = Path("reply_pools_augmented.json.gz")
USED_PAIRS_PATH = Path("used_pairs.json.gz")
GZIP_COMPRESSION_LEVEL = 3

def load_gzip_json(path: Path):
    """Load compressed JSON file with error handling"""
    try:
        if path.exists():
            with gzip.open(path, "rt") as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        return None

def save_gzip_json(data, path: Path):
    """Save data to compressed JSON file with atomic write"""
    temp_path = path.with_suffix(".tmp.gz")
    try:
        with gzip.open(temp_path, "wt", compresslevel=GZIP_COMPRESSION_LEVEL) as f:
            json.dump(data, f)
        temp_path.replace(path)
    except Exception as e:
        print(f"Error saving {path}: {str(e)}")
    finally:
        temp_path.unlink(missing_ok=True)

# Load or initialize reply pools
REPLY_POOLS = load_gzip_json(REPLY_POOLS_PATH) or {
    "general": {
        "triggers": [],
        "responses": ["Honey, let's talk about something more exciting..."],
        "questions": ["What really gets you going?"]
    },
    "positive": {
        "triggers": [],
        "responses": [
            "hi babe".
        ],
        "questions": [
            "hi?" 
        ]
    }
}

# Initialize response queues and usage tracking
CATEGORY_QUEUES = {}
USED_PAIRS = defaultdict(set)

def initialize_queues():
    global CATEGORY_QUEUES
    CATEGORY_QUEUES = {}
    for category, data in REPLY_POOLS.items():
        responses = data.get("responses", [])
        questions = data.get("questions", [])
        if responses and questions:
            combinations = list(product(range(len(responses)), range(len(questions))))
            random.shuffle(combinations)
            filtered = [(r, q) for r, q in combinations if (r, q) not in USED_PAIRS[category]]
            CATEGORY_QUEUES[category] = deque(filtered)
        else:
            CATEGORY_QUEUES[category] = deque()

# Load used pairs
if loaded_pairs := load_gzip_json(USED_PAIRS_PATH):
    USED_PAIRS.update(loaded_pairs)
initialize_queues()

# Security configuration
AUTHORIZED_OPERATORS = {
    "cone478", "cone353", "cone229", "cone516", "cone481", "cone335",
    "cone424", "cone069", "cone096", "cone075", "cone136", "cone406",
    "cone047", "cone461", "cone423", "cone290", "cone407", "cone468",
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

class SallyResponse(BaseModel):
    matched_words: List[str]
    matched_category: str
    confidence: float
    sentiment: float
    replies: List[str]

def log_to_dataset(user_input: str, response_data: dict, operator: str):
    try:
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": user_input,
            "matched_category": response_data["matched_category"],
            "response": response_data["replies"][0] if response_data["replies"] else None,
            "question": response_data["replies"][1] if len(response_data["replies"]) > 1 else None,
            "operator": operator,
            "confidence": response_data["confidence"],
            "sentiment": response_data["sentiment"],
            "embedding": nlp(user_input).vector.tolist()
        }
        with gzip.open(DATASET_PATH, "at", compresslevel=GZIP_COMPRESSION_LEVEL) as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Error logging to dataset: {str(e)}")

def store_uncertain(user_input: str):
    try:
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": user_input,
            "reviewed": False
        }
        with gzip.open(UNCERTAIN_PATH, "at", compresslevel=GZIP_COMPRESSION_LEVEL) as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Error storing uncertain response: {str(e)}")

def augment_dataset():
    try:
        if not DATASET_PATH.exists():
            return

        with gzip.open(DATASET_PATH, "rt") as f:
            entries = [json.loads(line) for line in f]

        # Auto-discover new categories
        category_vocabs = defaultdict(set)
        for entry in entries:
            doc = nlp(entry["user_input"])
            category = entry["matched_category"]
            category_vocabs[category].update([token.text.lower() for token in doc if token.is_alpha])

        # Create new categories with default responses
        for category, words in category_vocabs.items():
            if category not in REPLY_POOLS:
                REPLY_POOLS[category] = {
                    "triggers": list(words),
                    "responses": ["Honey, let's take this somewhere more private..."],
                    "questions": ["What's your deepest, darkest fantasy?"]
                }

        save_gzip_json(REPLY_POOLS, REPLY_POOLS_PATH)
        initialize_queues()
        
    except Exception as e:
        print(f"Augmentation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Augmentation failed")

async def verify_operator(request: Request):
    operator_email = request.headers.get("X-Operator-Email")
    if not operator_email or operator_email not in AUTHORIZED_OPERATORS:
        raise HTTPException(status_code=403, detail="Unauthorized operator")
    return operator_email

@app.post("/1A9I6F1O5R1C8O3N1E5145ID", response_model=SallyResponse)
async def analyze_message(
    request: Request,
    user_input: UserMessage,
    operator: str = Depends(verify_operator)
):
    try:
        message = user_input.message.strip().lower()
        doc = nlp(message)
        blob = TextBlob(message)
        sentiment = blob.sentiment.polarity

        # Find matching triggers
        matched_triggers = []
        best_category = "general"
        max_similarity = 0.0

        for category, data in REPLY_POOLS.items():
            for trigger in data["triggers"]:
                try:
                    if '*' in trigger:
                        pattern = re.compile(trigger.replace('*', '.*'), re.IGNORECASE)
                        if pattern.fullmatch(message):
                            similarity = 1.0
                            matched_triggers.append(trigger)
                    else:
                        trigger_doc = nlp(trigger)
                        similarity = doc.similarity(trigger_doc)
                        if similarity > 0.7:
                            matched_triggers.append(trigger)

                    # Update best category
                    category_doc = nlp(" ".join(data["triggers"]))
                    category_similarity = doc.similarity(category_doc) * (1 + abs(sentiment))
                    if category_similarity > max_similarity:
                        max_similarity = category_similarity
                        best_category = category

                except Exception as e:
                    print(f"Error processing trigger '{trigger}': {str(e)}")
                    continue

        # Prepare response
        response_data = {
            "matched_words": matched_triggers,
            "matched_category": best_category,
            "confidence": round(max_similarity, 2),
            "sentiment": sentiment,
            "replies": []
        }

        # Get response pair
        category_data = REPLY_POOLS.get(best_category, REPLY_POOLS["general"])
        queue = CATEGORY_QUEUES.get(best_category, deque())

        if queue:
            try:
                r_idx, q_idx = queue.popleft()
                response_data["replies"] = [
                    category_data["responses"][r_idx],
                    category_data["questions"][q_idx]
                ]
                USED_PAIRS[best_category].add((r_idx, q_idx))
                save_gzip_json(dict(USED_PAIRS), USED_PAIRS_PATH)
            except (IndexError, KeyError) as e:
                print(f"Error getting response pair: {str(e)}")
                response_data["replies"] = [
                    "Honey, let's take this somewhere more private...",
                    "What's your deepest, darkest fantasy?"
                ]

        # Fallback
        if not response_data["replies"]:
            response_data["replies"] = [
                "Honey, let's take this somewhere more private...",
                "What's your deepest, darkest fantasy?"
            ]

        # Active learning
        if response_data["confidence"] < 0.6:
            store_uncertain(message)
            if len(response_data["replies"]) > 1:
                response_data["replies"][1] += " Could you rephrase that, baby?"

        log_to_dataset(message, response_data, operator)
        return response_data

    except Exception as e:
        print(f"Error in analyze_message: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/dataset/analytics")
async def get_analytics():
    analytics = {
        "total_entries": 0,
        "common_categories": {},
        "confidence_stats": {},
        "sentiment_stats": {}
    }

    try:
        if DATASET_PATH.exists():
            with gzip.open(DATASET_PATH, "rt") as f:
                entries = [json.loads(line) for line in f]

            analytics["total_entries"] = len(entries)
            analytics["common_categories"] = Counter(entry["matched_category"] for entry in entries)

            if entries:
                confidences = [e.get("confidence", 0) for e in entries]
                sentiments = [e.get("sentiment", 0) for e in entries]

                analytics["confidence_stats"] = {
                    "average": round(sum(confidences)/len(confidences), 2),
                    "min": round(min(confidences), 2),
                    "max": round(max(confidences), 2)
                }

                analytics["sentiment_stats"] = {
                    "average": round(sum(sentiments)/len(sentiments), 2),
                    "positive": len([s for s in sentiments if s > 0]),
                    "negative": len([s for s in sentiments if s < 0]),
                    "neutral": len([s for s in sentiments if s == 0])
                }

    except Exception as e:
        print(f"Analytics error: {str(e)}")

    return analytics

@app.post("/augment")
async def trigger_augmentation():
    augment_dataset()
    return {"status": "Dataset augmented", "new_pools": REPLY_POOLS}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
