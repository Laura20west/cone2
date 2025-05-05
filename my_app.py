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
import httpx

# Initialize NLP
nlp = spacy.load("en_core_web_md")
nltk.download('wordnet')

app = FastAPI()

# Configuration with GZIP compression
DATASET_PATH = Path("conversation_dataset.jsonl.gz")
UNCERTAIN_PATH = Path("uncertain_responses.jsonl.gz")
REPLY_POOLS_PATH = Path("reply_pools_augmented.json.gz")
GZIP_COMPRESSION_LEVEL = 3

# Sentiment configuration
SENTIMENT_CATEGORIES = {
    "positive": {"threshold": 0.3, "responses": [], "questions": []},
    "neutral": {"threshold": (-0.3, 0.3), "responses": [], "questions": []},
    "negative": {"threshold": -0.3, "responses": [], "questions": []}
}

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

# Load or initialize sentiment-based reply pools
REPLY_POOLS = load_gzip_json(REPLY_POOLS_PATH) or {
    "positive": {
        "responses": ["You seem happy! Tell me more about what's exciting you..."],
        "questions": ["What's making you feel so positive right now?"]
    },
    "neutral": {
        "responses": ["Let's explore this further..."],
        "questions": ["How would you like to continue?"]
    },
    "negative": {
        "responses": ["I sense some hesitation...", "Let's work through this together..."],
        "questions": ["What's been troubling you?", "How can I help improve this situation?"]
    }
}

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
    sentiment_category: str
    sentiment_score: float
    confidence: float
    replies: List[str]

def log_to_dataset(user_input: str, response_data: dict, operator: str):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "sentiment_category": response_data["sentiment_category"],
        "sentiment_score": response_data["sentiment_score"],
        "operator": operator,
        "replies": response_data["replies"],
        "embedding": nlp(user_input).vector.tolist()
    }
    
    with gzip.open(DATASET_PATH, "at", compresslevel=GZIP_COMPRESSION_LEVEL) as f:
        f.write(json.dumps(entry) + "\n")

def determine_sentiment_category(score: float) -> str:
    if score >= 0.3:
        return "positive"
    elif score <= -0.3:
        return "negative"
    return "neutral"

def get_response_pair(category: str) -> List[str]:
    responses = REPLY_POOLS.get(category, {}).get("responses", [])
    questions = REPLY_POOLS.get(category, {}).get("questions", [])
    
    if not responses or not questions:
        return [
            "Let's explore this further...",
            "How would you like to continue?"
        ]
    
    response = random.choice(responses)
    question = random.choice(questions)
    return [response, question]

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
    
    # Perform sentiment analysis
    blob = TextBlob(message)
    sentiment_score = blob.sentiment.polarity
    sentiment_category = determine_sentiment_category(sentiment_score)
    
    # Get appropriate response pair
    response_pair = get_response_pair(sentiment_category)
    
    # Calculate confidence based on sentiment strength
    confidence = min(abs(sentiment_score) * 2, 1.0)  # Convert to 0-1 range
    
    response_data = {
        "sentiment_category": sentiment_category,
        "sentiment_score": round(sentiment_score, 2),
        "confidence": round(confidence, 2),
        "replies": response_pair
    }
    
    # Log interaction
    log_to_dataset(message, response_data, operator)
    
    # Handle low confidence cases
    if confidence < 0.5:
        fallback_pair = [
            "I want to make sure I understand correctly...",
            "Could you rephrase that in different words?"
        ]
        response_data["replies"] = fallback_pair
    
    return response_data

@app.get("/dataset/analytics")
async def get_analytics():
    analytics = {
        "total_entries": 0,
        "sentiment_distribution": {},
        "average_scores": {},
        "common_responses": {}
    }
    
    if DATASET_PATH.exists():
        try:
            with gzip.open(DATASET_PATH, "rt") as f:
                entries = [json.loads(line) for line in f]
            
            analytics["total_entries"] = len(entries)
            
            # Sentiment distribution
            categories = [e["sentiment_category"] for e in entries]
            analytics["sentiment_distribution"] = Counter(categories)
            
            # Average scores
            sentiment_scores = {
                "positive": [],
                "neutral": [],
                "negative": []
            }
            for entry in entries:
                cat = entry["sentiment_category"]
                sentiment_scores[cat].append(entry["sentiment_score"])
            
            analytics["average_scores"] = {
                cat: round(sum(scores)/len(scores), 2) if scores else 0
                for cat, scores in sentiment_scores.items()
            }
            
            # Common responses
            all_responses = [tuple(e["replies"]) for e in entries]
            analytics["common_responses"] = Counter(all_responses).most_common(5)
        
        except Exception as e:
            print(f"Analytics generation failed: {e}")
    
    return analytics

@app.post("/augment")
async def trigger_augmentation():
    # Analyze dataset to improve responses
    if DATASET_PATH.exists():
        try:
            with gzip.open(DATASET_PATH, "rt") as f:
                entries = [json.loads(line) for line in f]
            
            # Learn from successful interactions
            for entry in entries:
                category = entry["sentiment_category"]
                replies = entry.get("replies", [])
                
                if len(replies) >= 2:
                    response = replies[0]
                    question = replies[1]
                    
                    # Add unique responses
                    if response not in REPLY_POOLS[category]["responses"]:
                        REPLY_POOLS[category]["responses"].append(response)
                    
                    # Add unique questions
                    if question not in REPLY_POOLS[category]["questions"]:
                        REPLY_POOLS[category]["questions"].append(question)
            
            save_gzip_json(REPLY_POOLS, REPLY_POOLS_PATH)
            return {"status": "Augmentation complete", "new_responses": REPLY_POOLS}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Augmentation failed: {str(e)}")
    
    return {"status": "No dataset available for augmentation"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
