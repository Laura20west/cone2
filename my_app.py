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

# Initialize NLP
nlp = spacy.load("en_core_web_md")
nltk.download('wordnet')

app = FastAPI()

# Configuration with GZIP compression
DATASET_PATH = Path("conversation_dataset.jsonl.gz")
UNCERTAIN_PATH = Path("uncertain_responses.jsonl.gz")
REPLY_POOLS_PATH = Path("reply_pools_augmented.json.gz")
USED_PAIRS_PATH = Path("used_pairs.json.gz")
GZIP_COMPRESSION_LEVEL = 3  # Balanced compression level (1-9)

def load_gzip_json(path):
    """Load compressed JSON file with error handling"""
    try:
        if path.exists():
            with gzip.open(path, "rt") as f:
                return json.load(f)
    except (gzip.BadGzipFile, json.JSONDecodeError) as e:
        print(f"Error loading {path}: {e}")
    return None

def save_gzip_json(data, path):
    """Save data to compressed JSON file with atomic write"""
    temp_path = path.with_suffix(".tmp.gz")
    try:
        with gzip.open(temp_path, "wt", compresslevel=GZIP_COMPRESSION_LEVEL) as f:
            json.dump(data, f)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)

# Load or initialize reply pools with compression
REPLY_POOLS = load_gzip_json(REPLY_POOLS_PATH) or {
    "general": {
        "triggers": [],
        "responses": ["Honey, let's talk about something more exciting..."],
        "questions": ["What really gets you going?"]
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
        combinations = [(r_idx, q_idx) for r_idx in range(len(responses))
                      for q_idx in range(len(questions))]
        random.shuffle(combinations)
        CATEGORY_QUEUES[category] = deque(
            [(r, q) for r, q in combinations 
             if (r, q) not in USED_PAIRS[category]]
        )

# Load used pairs with compression
loaded_pairs = load_gzip_json(USED_PAIRS_PATH)
USED_PAIRS = defaultdict(set, loaded_pairs if loaded_pairs else {})
initialize_queues()

# Security configuration
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
    matched_words: List[str]
    matched_category: str
    confidence: float
    sentiment: float
    replies: List[str]

def log_to_dataset(user_input: str, response_data: dict, operator: str):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "matched_category": response_data["matched_category"].lower(),
        "response": response_data["replies"][0] if response_data["replies"] else None,
        "question": response_data["replies"][1] if len(response_data["replies"]) > 1 else None,
        "operator": operator,
        "confidence": response_data["confidence"],
        "sentiment": response_data["sentiment"],
        "embedding": nlp(user_input).vector.tolist()
    }
    
    with gzip.open(DATASET_PATH, "at", compresslevel=GZIP_COMPRESSION_LEVEL) as f:
        f.write(json.dumps(entry) + "\n")

def store_uncertain(user_input: str):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "reviewed": False
    }
    
    with gzip.open(UNCERTAIN_PATH, "at", compresslevel=GZIP_COMPRESSION_LEVEL) as f:
        f.write(json.dumps(entry) + "\n")

def augment_dataset():
    if DATASET_PATH.exists():
        try:
            with gzip.open(DATASET_PATH, "rt") as f:
                entries = [json.loads(line) for line in f]
            
            # Improved augmentation logic
            for entry in entries:
                category = entry["matched_category"]
                if category not in REPLY_POOLS:
                    REPLY_POOLS[category] = {
                        "triggers": [],
                        "responses": [],
                        "questions": []
                    }
                
                doc = nlp(entry["user_input"])
                # Add noun chunks as potential triggers
                for chunk in doc.noun_chunks:
                    if chunk.text not in REPLY_POOLS[category]["triggers"]:
                        REPLY_POOLS[category]["triggers"].append(chunk.text)
                
                # Add sentiment-based responses
                sentiment = entry.get("sentiment", 0)
                response = entry.get("response", "")
                question = entry.get("question", "")
                
                if sentiment > 0.5 and response:
                    REPLY_POOLS[category]["responses"].append(response)
                elif sentiment < -0.5 and question:
                    REPLY_POOLS[category]["questions"].append(question)
            
            save_gzip_json(REPLY_POOLS, REPLY_POOLS_PATH)
            initialize_queues()
        
        except Exception as e:
            print(f"Augmentation failed: {e}")
            raise HTTPException(status_code=500, detail="Augmentation process failed")

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
    message = user_input.message.strip()
    doc = nlp(message.lower())
    blob = TextBlob(message)
    
    # Sentiment analysis (-1 to 1)
    sentiment = blob.sentiment.polarity
    
    # Find all matching triggers
    matched_triggers = []
    for category, data in REPLY_POOLS.items():
        for trigger in data["triggers"]:
            trigger_doc = nlp(trigger)
            similarity = doc.similarity(trigger_doc)
            if similarity > 0.7:  # Higher threshold for multi-trigger detection
                matched_triggers.append(trigger)
    
    # Determine best category
    best_category = "general"
    max_similarity = 0.0
    for category, data in REPLY_POOLS.items():
        category_doc = nlp(" ".join(data["triggers"]))
        similarity = doc.similarity(category_doc) * (1 + abs(sentiment))
        if similarity > max_similarity:
            max_similarity = similarity
            best_category = category
    
    # Prepare response
    response = {
        "matched_words": matched_triggers,
        "matched_category": best_category,
        "confidence": round(max_similarity, 2),
        "sentiment": sentiment,
        "replies": []
    }
    
    # Get non-repeating response pair
    category_data = REPLY_POOLS.get(best_category, REPLY_POOLS["general"])
    if category_data["responses"] and category_data["questions"]:
        queue = CATEGORY_QUEUES.get(best_category, deque())
        
        best_score = -1
        best_pair = None
        
        # Find best matching response pair
        for _ in range(min(5, len(queue))):
            r_idx, q_idx = queue[0]
            response_text = category_data["responses"][r_idx]
            question_text = category_data["questions"][q_idx]
            
            # Calculate score based on sentiment alignment
            response_sentiment = TextBlob(response_text).sentiment.polarity
            sentiment_score = 1 - abs(sentiment - response_sentiment)
            
            # Calculate similarity score
            response_doc = nlp(response_text)
            similarity_score = doc.similarity(response_doc)
            
            total_score = sentiment_score * 0.6 + similarity_score * 0.4
            
            if total_score > best_score:
                best_score = total_score
                best_pair = queue.popleft()
            else:
                queue.rotate(-1)
        
        if best_pair:
            r_idx, q_idx = best_pair
            response["replies"].extend([
                category_data["responses"][r_idx],
                category_data["questions"][q_idx]
            ])
            USED_PAIRS[best_category].add((r_idx, q_idx))
            save_gzip_json(USED_PAIRS, USED_PAIRS_PATH)
            initialize_queues()
    
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
    analytics = {
        "total_entries": 0,
        "common_categories": {},
        "confidence_stats": {},
        "sentiment_stats": {}
    }
    
    if DATASET_PATH.exists():
        try:
            with gzip.open(DATASET_PATH, "rt") as f:
                entries = [json.loads(line) for line in f]
            
            analytics["total_entries"] = len(entries)
            analytics["common_categories"] = Counter(
                entry["matched_category"] for entry in entries
            )
            
            if entries:
                confidences = [entry["confidence"] for entry in entries]
                sentiments = [entry["sentiment"] for entry in entries]
                
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
            print(f"Analytics generation failed: {e}")
    
    return analytics

@app.post("/augment")
async def trigger_augmentation():
    augment_dataset()
    return {"status": "Dataset augmented", "new_pools": REPLY_POOLS}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
