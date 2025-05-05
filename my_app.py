from fastapi import FastAPI, HTTPException, Request, Depends as D
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as BM
import spacy, json, uuid, random, re, nltk, os
from datetime import datetime as dt
from pathlib import Path
from collections import defaultdict as dd, deque, Counter
from typing import Dict, List, Optional as O
from itertools import product as prod
from nltk.corpus import wordnet as wn

app = FastAPI(title="Sally API", description="Backend for Sally Assistant")
nlp = spacy.load("en_core_web_md") if spacy.util.is_package("en_core_web_md") else spacy.load("en_core_web_sm")
nltk.download('wordnet', quiet=True)

B_DIR = Path(__file__).parent.absolute()
D_PATH = B_DIR/"conv_data.jsonl"
U_PATH = B_DIR/"uncertain.jsonl"
R_PATH = B_DIR/"reply_pools.json"

for p in [D_PATH, U_PATH]:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch(exist_ok=True)

class TG:
    def __init__(self):
        self.miss = {'fuck':['fck','fuk','f*ck'],'sex':['sx','secks'],'cock':['cok','kock'],'pussy':['p*ssy'],
                     'anal':['anl','a**l'],'boobs':['b**bs','tits'],'cum':['c*m','coom'],'dick':['d*ck'],
                     'horny':['h0rny'],'naked':['n*ked'],'chat':['ch@t'],'video':['vid'],'picture':['pic'],'send':['snd']}
        self.phrases = ["show me","send me","wanna see","how much","you charge","how old","where are you",
                        "do you do","i want to","let's have","price for","are you real","can you send"]

    def gen_vars(self, t):
        v = {t}
        for w in t.lower().split():
            if w in self.miss: v.update([t.lower().replace(w, m) for m in self.miss[w]])
        for tok in nlp(t.lower()):
            if tok.pos_ == "VERB":
                v.update([f"want to {tok.lemma_}", f"how to {tok.lemma_}", f"let's {tok.lemma_}"])
        for tok in nlp(t.lower()):
            if tok.pos_ in ["NOUN","VERB","ADJ"]:
                for s in wn.synsets(tok.text)[:2]:
                    for l in s.lemmas()[:2]: v.add(l.name().replace('_',' '))
        for p in self.phrases:
            for w in t.lower().split():
                if len(w)>3: v.update([f"{p} {w}",f"{w} {p}"])
        return v

tg = TG()
DEFAULT = {
    "gen": {"t":["hi","hello","hey"],"r":["Hey there! How's your day?","Hello! What brings you here?"],
            "q":["What are you looking for?","How can I help?"]},
    "ver": {"t":["are you real","verify"],"r":["I'm real! Want proof?","No bots here!"],
            "q":["What would convince you?","Talk about something fun?"]},
    "loc": {"t":["where are you","location"],"r":["In the city center","Private location"],
            "q":["Where are you?","Want to meet?"]},
    "price": {"t":["how much","cost"],"r":["Depends on what you want","Let's discuss"],
              "q":["What experience?","Duration?"]},
    "pics": {"t":["pics","photos"],"r":["Special pics for serious","No free pics"],
             "q":["What pics?","Share first?"]},
    "srv": {"t":["services","offer"],"r":["Open-minded GFE","Fulfill fantasies"],
            "q":["What mood?","Fantasy?"]},
    "exp": {"t":["sex","fuck","dick"],"r":["Naughty mood!","You're direct"],
            "q":["Favorite position?","Do now?"]}
}

R_POOLS = DEFAULT
if R_PATH.exists():
    try: R_POOLS = json.loads(R_PATH.read_text())
    except: R_PATH.write_text(json.dumps(DEFAULT))

for c in R_POOLS.values():
    c["t"] = list({x for t in c["t"] for x in tg.gen_vars(t)})

C_PRIO = ["exp","price","pics","loc","ver","srv","gen"]
C_QUEUES = {}
for c in C_PRIO:
    rs = R_POOLS[c]["r"]
    qs = R_POOLS[c]["q"]
    if rs and qs:
        combos = list(prod(range(len(rs)), range(len(qs))))
        random.shuffle(combos)
        C_QUEUES[c] = deque(combos)

A_OPS = {"cone478","cone353","cone229","cone516","cone481","admin@co.com"}

def auth_op(e):
    if not e: return False
    e = e.lower().strip()
    if e in A_Ops: return True
    return any(e.endswith(d[1:]) for d in A_Ops if '@' in d)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Msg(BM):
    m: str
    ctx: O[Dict] = None

class Resp(BM):
    cat: str = "gen"
    conf: float = 0.0
    reps: List[str] = []

def log_msg(m, r, op, ctx=None):
    entry = {"id":str(uuid.uuid4()),"t":dt.utcnow().isoformat(),"m":m,"cat":r["cat"],"reps":r["reps"],"op":op,"ctx":ctx}
    try: entry["emb"] = nlp(m).vector.tolist()
    except: pass
    with open(D_PATH, "a") as f: f.write(json.dumps(entry)+"\n")

def store_uncertain(m, ctx=None):
    entry = {"id":str(uuid.uuid4()),"t":dt.utcnow().isoformat(),"m":m,"rev":False,"ctx":ctx}
    with open(U_PATH, "a") as f: f.write(json.dumps(entry)+"\n")

def match_cat(m):
    m = m.lower()
    for c in C_PRIO:
        for t in R_POOLS[c]["t"]:
            if re.search(rf'\b{re.escape(t)}\b', m, re.I): return (c, t, 1.0)
            if '*' in t and re.search(rf'\b{t.replace("*",r"\w*")}\b', m, re.I): return (c, t, 0.9)
    best = ("gen", None, 0.0)
    try:
        md = nlp(m)
        for c in C_PRIO:
            for t in R_POOLS[c]["t"]:
                td = nlp(t)
                s = md.similarity(td)
                if s > best[2]: best = (c, t, s)
    except: pass
    return best

@app.post("/1A9I6F1O5R1C8O3N1E5145ID", response_model=Resp)
async def handle_msg(req: Request, msg: Msg, op: str = D(verify_operator)):
    m = msg.m.strip()
    if not m: return Resp(reps=["Waiting...", "What's up?"])
    cat, _, conf = match_cat(m)
    r_data = {"cat":cat, "conf":round(conf,2), "reps":[]}
    if cat in C_QUEUES and C_QUEUES[cat]:
        r_idx, q_idx = C_QUEUES[cat][0]
        C_QUEUES[cat].rotate(-1)
        r = R_POOLS[cat]["r"][r_idx % len(R_POOLS[cat]["r"])]
        q = R_POOLS[cat]["q"][q_idx % len(R_POOLS[cat]["q"])]
        r_data["reps"] = [r, q]
    if not r_data["reps"]: r_data["reps"] = ["Tell me more...", "What else?"]
    log_msg(m, r_data, op, msg.ctx)
    if conf <0.6: 
        store_uncertain(m, msg.ctx)
        if len(r_data["reps"])>1: r_data["reps"][1] += " More details?"
    return Resp(**r_data)

@app.get("/health")
async def health(): return {"status":"ok", "v":"8.5"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
