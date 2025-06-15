import json
import random
import re
import os
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


# Comprehensive paraphrasing templates (100 variations)
PARAPHRASE_TEMPLATES = [
    # Word replacements - positive (10)
    lambda x: x.replace("good", "excellent"),
    lambda x: x.replace("good", "wonderful"),
    lambda x: x.replace("good", "fantastic"),
    lambda x: x.replace("great", "amazing"),
    lambda x: x.replace("great", "outstanding"),
    lambda x: x.replace("nice", "lovely"),
    lambda x: x.replace("nice", "pleasant"),
    lambda x: x.replace("happy", "delighted"),
    lambda x: x.replace("happy", "thrilled"),
    lambda x: x.replace("excited", "enthusiastic"),

    # Word replacements - negative (10)
    lambda x: x.replace("bad", "terrible"),
    lambda x: x.replace("bad", "awful"),
    lambda x: x.replace("sad", "disappointed"),
    lambda x: x.replace("sad", "heartbroken"),
    lambda x: x.replace("angry", "furious"),
    lambda x: x.replace("upset", "distressed"),
    lambda x: x.replace("worried", "concerned"),
    lambda x: x.replace("scared", "terrified"),
    lambda x: x.replace("confused", "perplexed"),
    lambda x: x.replace("tired", "exhausted"),

    # Intensifiers (10)
    lambda x: x.replace("very", "extremely"),
    lambda x: x.replace("very", "incredibly"),
    lambda x: x.replace("really", "absolutely"),
    lambda x: x.replace("quite", "rather"),
    lambda x: x.replace("pretty", "fairly"),
    lambda x: x.replace("so", "tremendously"),
    lambda x: x.replace("too", "excessively"),
    lambda x: x.replace("much", "significantly"),
    lambda x: x.replace("little", "somewhat"),
    lambda x: x.replace("big", "enormous"),

    # Verb transformations (10)
    lambda x: re.sub(r'\bis\b', 'appears to be', x),
    lambda x: re.sub(r'\bwas\b', 'seemed to be', x),
    lambda x: re.sub(r'\bcan\b', 'might be able to', x),
    lambda x: re.sub(r'\bcould\b', 'would be capable of', x),
    lambda x: re.sub(r'\bwill\b', 'is going to', x),
    lambda x: re.sub(r'\bshould\b', 'ought to', x),
    lambda x: re.sub(r'\bmust\b', 'has to', x),
    lambda x: re.sub(r'\bmight\b', 'could possibly', x),
    lambda x: re.sub(r'\bmay\b', 'might possibly', x),
    lambda x: re.sub(r'\bwant\b', 'desire', x),

    # Sentence starters (10)
    lambda x: f"You know, {x.lower()}" if not x.startswith(('You', 'I', 'We')) else x,
    lambda x: f"Actually, {x.lower()}" if not x.startswith(('Actually', 'I', 'You')) else x,
    lambda x: f"Honestly, {x.lower()}" if not x.startswith(('Honestly', 'I', 'You')) else x,
    lambda x: f"In fact, {x.lower()}" if not x.startswith(('In fact', 'I', 'You')) else x,
    lambda x: f"To be honest, {x.lower()}" if not x.startswith(('To be', 'I', 'You')) else x,
    lambda x: f"Frankly, {x.lower()}" if not x.startswith(('Frankly', 'I', 'You')) else x,
    lambda x: f"Basically, {x.lower()}" if not x.startswith(('Basically', 'I', 'You')) else x,
    lambda x: f"Generally, {x.lower()}" if not x.startswith(('Generally', 'I', 'You')) else x,
    lambda x: f"Typically, {x.lower()}" if not x.startswith(('Typically', 'I', 'You')) else x,
    lambda x: f"Usually, {x.lower()}" if not x.startswith(('Usually', 'I', 'You')) else x,

    # Opinion markers (10)
    lambda x: x.replace("I think", "In my opinion"),
    lambda x: x.replace("I think", "I believe"),
    lambda x: x.replace("I think", "It seems to me"),
    lambda x: x.replace("I think", "From my perspective"),
    lambda x: x.replace("I feel", "I sense"),
    lambda x: x.replace("I feel", "My impression is"),
    lambda x: x.replace("I believe", "I'm convinced"),
    lambda x: x.replace("I know", "I'm aware"),
    lambda x: x.replace("I understand", "I comprehend"),
    lambda x: x.replace("I realize", "I recognize"),

    # Frequency modifiers (10)
    lambda x: x.replace("always", "constantly"),
    lambda x: x.replace("always", "consistently"),
    lambda x: x.replace("never", "rarely"),
    lambda x: x.replace("never", "hardly ever"),
    lambda x: x.replace("often", "frequently"),
    lambda x: x.replace("sometimes", "occasionally"),
    lambda x: x.replace("usually", "typically"),
    lambda x: x.replace("seldom", "infrequently"),
    lambda x: x.replace("regularly", "routinely"),
    lambda x: x.replace("constantly", "continuously"),

    # Connectors and transitions (10)
    lambda x: x.replace("but", "however"),
    lambda x: x.replace("and", "plus"),
    lambda x: x.replace("also", "additionally"),
    lambda x: x.replace("because", "since"),
    lambda x: x.replace("so", "therefore"),
    lambda x: x.replace("then", "consequently"),
    lambda x: x.replace("though", "although"),
    lambda x: x.replace("while", "whereas"),
    lambda x: x.replace("if", "provided that"),
    lambda x: x.replace("when", "whenever"),

    # Emotional expressions (10)
    lambda x: x.replace("love", "adore"),
    lambda x: x.replace("hate", "despise"),
    lambda x: x.replace("like", "enjoy"),
    lambda x: x.replace("dislike", "find unpleasant"),
    lambda x: x.replace("enjoy", "take pleasure in"),
    lambda x: x.replace("prefer", "would rather"),
    lambda x: x.replace("want", "would like"),
    lambda x: x.replace("need", "require"),
    lambda x: x.replace("hope", "wish"),
    lambda x: x.replace("expect", "anticipate"),

    # Sentence restructuring (10)
    lambda x: re.sub(r"It's (.+)", r"The fact is that it's \1", x),
    lambda x: re.sub(r"That's (.+)", r"The reality is that it's \1", x),
    lambda x: re.sub(r"This is (.+)", r"What we have here is \1", x),
    lambda x: re.sub(r"There are (.+)", r"You'll find \1", x),
    lambda x: re.sub(r"There is (.+)", r"What exists is \1", x),
    lambda x: re.sub(r"You have (.+)", r"What you possess is \1", x),
    lambda x: re.sub(r"I have (.+)", r"What I possess is \1", x),
    lambda x: re.sub(r"We have (.+)", r"What we possess is \1", x),
    lambda x: re.sub(r"They have (.+)", r"What they possess is \1", x),
    lambda x: re.sub(r"People (.+)", r"Individuals \1", x),

    # Additional word variations (20 more to reach 100)
    lambda x: x.replace("hard", "challenging"),
    lambda x: x.replace("easy", "simple"),
    lambda x: x.replace("difficult", "tough"),
    lambda x: x.replace("problem", "issue"),
    lambda x: x.replace("solution", "answer"),
    lambda x: x.replace("important", "crucial"),
    lambda x: x.replace("interesting", "fascinating"),
    lambda x: x.replace("amazing", "incredible"),
    lambda x: x.replace("terrible", "dreadful"),
    lambda x: x.replace("beautiful", "gorgeous"),
    lambda x: x.replace("ugly", "hideous"),
    lambda x: x.replace("smart", "intelligent"),
    lambda x: x.replace("stupid", "foolish"),
    lambda x: x.replace("funny", "hilarious"),
    lambda x: x.replace("boring", "tedious"),
    lambda x: x.replace("weird", "strange"),
    lambda x: x.replace("normal", "typical"),
    lambda x: x.replace("special", "unique"),
    lambda x: x.replace("common", "ordinary"),
    lambda x: x.replace("perfect", "flawless"),

    # NEW ADDITIONS - 100 more templates below

    # More positive word replacements (10)
    lambda x: x.replace("awesome", "phenomenal"),
    lambda x: x.replace("cool", "remarkable"),
    lambda x: x.replace("sweet", "delightful"),
    lambda x: x.replace("fine", "splendid"),
    lambda x: x.replace("super", "extraordinary"),
    lambda x: x.replace("neat", "impressive"),
    lambda x: x.replace("solid", "excellent"),
    lambda x: x.replace("decent", "respectable"),
    lambda x: x.replace("okay", "acceptable"),
    lambda x: x.replace("alright", "satisfactory"),

    # More negative word replacements (10)
    lambda x: x.replace("horrible", "atrocious"),
    lambda x: x.replace("nasty", "revolting"),
    lambda x: x.replace("gross", "disgusting"),
    lambda x: x.replace("lame", "disappointing"),
    lambda x: x.replace("sucks", "is inadequate"),
    lambda x: x.replace("crappy", "substandard"),
    lambda x: x.replace("lousy", "inferior"),
    lambda x: x.replace("pathetic", "pitiful"),
    lambda x: x.replace("useless", "worthless"),
    lambda x: x.replace("pointless", "meaningless"),

    # Time expressions (10)
    lambda x: x.replace("now", "currently"),
    lambda x: x.replace("today", "at present"),
    lambda x: x.replace("yesterday", "the previous day"),
    lambda x: x.replace("tomorrow", "the following day"),
    lambda x: x.replace("soon", "in the near future"),
    lambda x: x.replace("later", "subsequently"),
    lambda x: x.replace("earlier", "previously"),
    lambda x: x.replace("recently", "not long ago"),
    lambda x: x.replace("eventually", "ultimately"),
    lambda x: x.replace("immediately", "right away"),

    # Professional/formal replacements (10)
    lambda x: x.replace("help", "assist"),
    lambda x: x.replace("show", "demonstrate"),
    lambda x: x.replace("tell", "inform"),
    lambda x: x.replace("ask", "inquire"),
    lambda x: x.replace("get", "obtain"),
    lambda x: x.replace("give", "provide"),
    lambda x: x.replace("make", "create"),
    lambda x: x.replace("do", "perform"),
    lambda x: x.replace("say", "state"),
    lambda x: x.replace("try", "attempt"),

    # Size and quantity modifiers (10)
    lambda x: x.replace("huge", "massive"),
    lambda x: x.replace("tiny", "minuscule"),
    lambda x: x.replace("small", "compact"),
    lambda x: x.replace("large", "substantial"),
    lambda x: x.replace("many", "numerous"),
    lambda x: x.replace("few", "a handful of"),
    lambda x: x.replace("lots", "plenty"),
    lambda x: x.replace("tons", "an abundance of"),
    lambda x: x.replace("loads", "a great deal of"),
    lambda x: x.replace("bunch", "a collection of"),

    # Movement and action verbs (10)
    lambda x: x.replace("go", "proceed"),
    lambda x: x.replace("come", "arrive"),
    lambda x: x.replace("run", "sprint"),
    lambda x: x.replace("walk", "stroll"),
    lambda x: x.replace("jump", "leap"),
    lambda x: x.replace("fall", "tumble"),
    lambda x: x.replace("climb", "ascend"),
    lambda x: x.replace("throw", "hurl"),
    lambda x: x.replace("catch", "capture"),
    lambda x: x.replace("break", "shatter"),

    # Mental state words (10)
    lambda x: x.replace("think", "contemplate"),
    lambda x: x.replace("remember", "recall"),
    lambda x: x.replace("forget", "overlook"),
    lambda x: x.replace("learn", "acquire knowledge about"),
    lambda x: x.replace("understand", "comprehend"),
    lambda x: x.replace("wonder", "ponder"),
    lambda x: x.replace("imagine", "envision"),
    lambda x: x.replace("dream", "fantasize"),
    lambda x: x.replace("focus", "concentrate"),
    lambda x: x.replace("ignore", "disregard"),

    # More sentence starters (10)
    lambda x: f"Clearly, {x.lower()}" if not x.startswith(('Clearly', 'I', 'You')) else x,
    lambda x: f"Obviously, {x.lower()}" if not x.startswith(('Obviously', 'I', 'You')) else x,
    lambda x: f"Naturally, {x.lower()}" if not x.startswith(('Naturally', 'I', 'You')) else x,
    lambda x: f"Certainly, {x.lower()}" if not x.startswith(('Certainly', 'I', 'You')) else x,
    lambda x: f"Indeed, {x.lower()}" if not x.startswith(('Indeed', 'I', 'You')) else x,
    lambda x: f"Undoubtedly, {x.lower()}" if not x.startswith(('Undoubtedly', 'I', 'You')) else x,
    lambda x: f"Surprisingly, {x.lower()}" if not x.startswith(('Surprisingly', 'I', 'You')) else x,
    lambda x: f"Interestingly, {x.lower()}" if not x.startswith(('Interestingly', 'I', 'You')) else x,
    lambda x: f"Fortunately, {x.lower()}" if not x.startswith(('Fortunately', 'I', 'You')) else x,
    lambda x: f"Unfortunately, {x.lower()}" if not x.startswith(('Unfortunately', 'I', 'You')) else x,

    # Question transformations (10)
    lambda x: re.sub(r"What is (.+)?", r"Can you tell me what \1 is?", x),
    lambda x: re.sub(r"How do (.+)?", r"What's the method for \1?", x),
    lambda x: re.sub(r"Why (.+)?", r"What's the reason that \1?", x),
    lambda x: re.sub(r"Where (.+)?", r"What location \1?", x),
    lambda x: re.sub(r"When (.+)?", r"At what time \1?", x),
    lambda x: re.sub(r"Who (.+)?", r"Which person \1?", x),
    lambda x: re.sub(r"Which (.+)?", r"What particular \1?", x),
    lambda x: re.sub(r"Are you (.+)?", r"Would you say you're \1?", x),
    lambda x: re.sub(r"Do you (.+)?", r"Is it true that you \1?", x),
    lambda x: re.sub(r"Can you (.+)?", r"Are you able to \1?", x),

    # Comparative expressions (10)
    lambda x: x.replace("better", "superior"),
    lambda x: x.replace("worse", "inferior"),
    lambda x: x.replace("best", "optimal"),
    lambda x: x.replace("worst", "most inadequate"),
    lambda x: x.replace("more", "additional"),
    lambda x: x.replace("less", "reduced"),
    lambda x: x.replace("most", "the majority of"),
    lambda x: x.replace("least", "minimal"),
    lambda x: x.replace("same", "identical"),
    lambda x: x.replace("different", "distinct"),

    # Casual to formal transformations (10)
    lambda x: x.replace("stuff", "items"),
    lambda x: x.replace("things", "matters"),
    lambda x: x.replace("guy", "individual"),
    lambda x: x.replace("girl", "young woman"),
    lambda x: x.replace("kid", "child"),
    lambda x: x.replace("folks", "people"),
    lambda x: x.replace("place", "location"),
    lambda x: x.replace("way", "method"),
    lambda x: x.replace("kind of", "somewhat"),
    lambda x: x.replace("sort of", "rather"),
]

# Comprehensive question generation patterns (100 variations)
QUESTION_PATTERNS = [
    # Basic inquiry patterns
    lambda x: f"What do you think about {x}?",
    lambda x: f"How do you feel about {x}?",
    lambda x: f"Can you tell me more about {x}?",
    lambda x: f"What's your experience with {x}?",
    lambda x: f"Why is {x} important to you?",
    lambda x: f"How would you describe {x}?",
    lambda x: f"What comes to mind when you think of {x}?",
    lambda x: f"How does {x} make you feel?",
    lambda x: f"What's interesting about {x}?",
    lambda x: f"What would you like to know about {x}?",

    # Deeper exploration
    lambda x: f"What aspects of {x} intrigue you most?",
    lambda x: f"How has {x} influenced your life?",
    lambda x: f"What challenges have you faced with {x}?",
    lambda x: f"What opportunities does {x} present?",
    lambda x: f"How do you see {x} evolving?",
    lambda x: f"What surprises you about {x}?",
    lambda x: f"What would you change about {x}?",
    lambda x: f"How do others perceive {x}?",
    lambda x: f"What lessons have you learned from {x}?",
    lambda x: f"What advice would you give about {x}?",
    lambda x: f"What’s the hardest truth you’ve learned about {x}?",
lambda x: f"Has {x} ever made you cry?",
lambda x: f"What scares you the most about {x}?",
lambda x: f"Who do you talk to when you're overwhelmed by {x}?",
lambda x: f"Have you ever felt lost because of {x}?",
lambda x: f"Is there a side of {x} you keep hidden?",
lambda x: f"What kind of healing has {x} brought you?",
lambda x: f"What are you still learning about {x}?",
lambda x: f"What memory involving {x} do you cherish most?",
lambda x: f"What do you wish someone had told you earlier about {x}?",

    # Comparative questions
    lambda x: f"How does {x} compare to your expectations?",
    lambda x: f"What makes {x} different from similar things?",
    lambda x: f"How has your view of {x} changed over time?",
    lambda x: f"What's the best thing about {x}?",
    lambda x: f"What's the most challenging aspect of {x}?",
    lambda x: f"How does {x} fit into your daily life?",
    lambda x: f"What role does {x} play in your happiness?",
    lambda x: f"How do you balance {x} with other priorities?",
    lambda x: f"What would life be like without {x}?",
    lambda x: f"How do you make decisions about {x}?",

    # Future-oriented questions
    lambda x: f"Where do you see {x} heading in the future?",
    lambda x: f"What are your goals regarding {x}?",
    lambda x: f"How do you plan to develop your relationship with {x}?",
    lambda x: f"What would you like to achieve with {x}?",
    lambda x: f"How might {x} change in the coming years?",
    lambda x: f"What hopes do you have for {x}?",
    lambda x: f"What concerns do you have about {x}?",
    lambda x: f"How can you improve your experience with {x}?",
    lambda x: f"What steps are you taking regarding {x}?",
    lambda x: f"What would success look like with {x}?",

    # Emotional and personal
    lambda x: f"What emotions does {x} bring up for you?",
    lambda x: f"How has {x} shaped who you are?",
    lambda x: f"What memories do you associate with {x}?",
    lambda x: f"How do you cope when {x} becomes difficult?",
    lambda x: f"What support do you need regarding {x}?",
    lambda x: f"How do you celebrate successes with {x}?",
    lambda x: f"What fears do you have about {x}?",
    lambda x: f"How do you find motivation for {x}?",
    lambda x: f"What brings you joy about {x}?",
    lambda x: f"How do you share {x} with others?",
    lambda x: f"What kind of relationship are you looking for when it comes to {x}?",
    lambda x: f"Have you ever been heartbroken over {x}?",
    lambda x: f"What does love look like to you in the context of {x}?",
    lambda x: f"How do you show affection through {x}?",
    lambda x: f"What’s your love language when it comes to {x}?",
    lambda x: f"What kind of partner do you see yourself being in {x}?",
    lambda x: f"Would you ever compromise on {x} for someone you love?",
    lambda x: f"What does commitment mean to you regarding {x}?",
    lambda x: f"How do you handle conflict when it's about {x}?",
    lambda x: f"Do you believe in forever when it comes to {x}?",


    # Practical and actionable
    lambda x: f"What practical steps can you take with {x}?",
    lambda x: f"How do you organize your approach to {x}?",
    lambda x: f"What resources do you use for {x}?",
    lambda x: f"How do you measure progress with {x}?",
    lambda x: f"What obstacles do you face with {x}?",
    lambda x: f"How do you overcome challenges in {x}?",
    lambda x: f"What tools help you with {x}?",
    lambda x: f"How do you stay consistent with {x}?",
    lambda x: f"What habits have you developed around {x}?",
    lambda x: f"How do you track your growth in {x}?",

    # Social and relational
    lambda x: f"How do you discuss {x} with friends?",
    lambda x: f"What do others think of your approach to {x}?",
    lambda x: f"How has {x} affected your relationships?",
    lambda x: f"What communities are you part of regarding {x}?",
    lambda x: f"How do you learn from others about {x}?",
    lambda x: f"What wisdom have you gained about {x}?",
    lambda x: f"How do you teach others about {x}?",
    lambda x: f"What support networks do you have for {x}?",
    lambda x: f"How do you collaborate with others on {x}?",
    lambda x: f"What cultural aspects of {x} interest you?",
    lambda x: f"Is {x} part of your long-term plans?",
lambda x: f"What role does ambition play in your relationship with {x}?",
lambda x: f"How does your lifestyle affect your involvement in {x}?",
lambda x: f"Would you ever change your job/life because of {x}?",
lambda x: f"What sacrifices have you made for {x}?",
lambda x: f"Does {x} bring you stability or chaos?",
lambda x: f"How do you balance fun and responsibility around {x}?",
lambda x: f"What motivates you to keep going with {x}?",
lambda x: f"What would success look like in {x} for you?",
lambda x: f"What role does discipline play in your pursuit of {x}?",


    # Reflective and philosophical
    lambda x: f"What does {x} mean to you personally?",
    lambda x: f"How does {x} connect to your values?",
    lambda x: f"What deeper truths have you discovered about {x}?",
    lambda x: f"How does {x} contribute to your purpose?",
    lambda x: f"What spiritual aspects of {x} resonate with you?",
    lambda x: f"How does {x} help you grow as a person?",
    lambda x: f"What paradoxes do you notice in {x}?",
    lambda x: f"How does {x} challenge your assumptions?",
    lambda x: f"What mysteries about {x} fascinate you?",
    lambda x: f"How does {x} inspire you?",

    # Creative and imaginative
    lambda x: f"If you could redesign {x}, what would you change?",
    lambda x: f"What creative possibilities do you see in {x}?",
    lambda x: f"How would you explain {x} to a child?",
    lambda x: f"What metaphor best describes {x}?",
    lambda x: f"If {x} were a person, what would they be like?",
    lambda x: f"What story would you tell about {x}?",
    lambda x: f"How would you paint a picture of {x}?",
    lambda x: f"What song would represent {x}?",
    lambda x: f"What adventure would {x} take you on?",
    lambda x: f"How would you innovate {x}?",
    lambda x: f"What principles guide your decisions about {x}?",
lambda x: f"Does {x} align with your personal values?",
lambda x: f"How would you handle it if your partner felt differently about {x}?",
lambda x: f"What does integrity mean to you in the area of {x}?",
lambda x: f"Would you change your mind about {x} if given enough reason?",
lambda x: f"How does {x} reflect who you really are?",
lambda x: f"Is {x} something you'd teach your kids?",
lambda x: f"Have you ever taken a stand because of {x}?",
lambda x: f"Do you see {x} as black and white or more nuanced?",
lambda x: f"What values do you think are missing in people’s approach to {x}?",


    # Contextual and situational
  lambda x: f"What’s the most romantic thing you've done involving {x}?",
lambda x: f"If I were into {x}, would that impress you?",
lambda x: f"What kind of dates involving {x} turn you on?",
lambda x: f"Would you do something spontaneous like {x} with me?",
lambda x: f"How would you seduce someone using {x}?",
lambda x: f"What’s something naughty or fun you associate with {x}?",
lambda x: f"What kind of adventure would you take me on involving {x}?",
lambda x: f"What’s your guilty pleasure related to {x}?",
lambda x: f"If {x} were a date idea, would you try it with me?",
lambda x: f"What’s the flirtiest thing someone ever said to you about {x}?",

]


# Common English stop words
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
}

class OptimizedNLP:
    """Handles natural language processing tasks with optimizations"""
    
    def __init__(self):
        self.paraphrase_templates = PARAPHRASE_TEMPLATES
        self.question_patterns = QUESTION_PATTERNS
        self.category_keywords = CATEGORY_KEYWORDS

    def paraphrase(self, text, iterations=3):
        """Apply multiple paraphrasing transformations"""
        if not text.strip():
            return text

        for _ in range(iterations):
            template = random.choice(self.paraphrase_templates)
            try:
                new_text = template(text)
                if new_text != text:  # Only keep changes that actually modify the text
                    text = new_text
            except:
                continue

        return text

    def generate_question(self, text):
        """Generate a follow-up question based on the text"""
        if not text.strip():
            return "What are your thoughts on this?"

        # Extract keywords or noun phrases
        keywords = self._extract_keywords(text)
        if not keywords:
            return random.choice([
                "What do you think about that?",
                "How does that make you feel?",
                "Can you tell me more?"
            ])

        keyword = random.choice(keywords)
        pattern = random.choice(self.question_patterns)

        try:
            return pattern(keyword)
        except:
            return f"What about {keyword}?"

    def _extract_keywords(self, text):
        """Extract potential keywords from text"""
        # Simple implementation - can be enhanced with proper NLP
        words = re.findall(r'\b\w+\b', text.lower())

        # Prioritize nouns and adjectives (simple heuristic)
        keywords = []
        for word in words:
            if len(word) > 3 and word not in STOP_WORDS:  # Basic filtering
                # Check if word belongs to any category
                for category, terms in self.category_keywords.items():
                    if word in terms:
                        keywords.append(word)

        # If no category matches, return the longer words
        if not keywords:
            keywords = [w for w in words if len(w) > 4][:3]

        return keywords if keywords else ["that"]


class DatasetManager:
    """Handles the keyword-based dataset structure"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = {}  # Will store the keyword-based structure
        self.all_responses = []  # Flat list of all responses for fallback
        self.vectorizer = None
        self.tfidf_matrix = None
        self.load()

    def load(self):
        """Load dataset synchronously"""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            else:
                self.data = self.get_fallback_data()
                print(f"Warning: Using fallback data. File not found: {self.file_path}")

            self.prepare_responses()
            print("Dataset loaded successfully")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.data = self.get_fallback_data()
            self.prepare_responses()
            print("Using fallback data due to loading error")

    def prepare_responses(self):
        """Prepare flat list of all responses and similarity matrix"""
        self.all_responses = []
        for keyword, entries in self.data.items():
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict) and 'response' in entry:
                        if entry['response']:
                            self.all_responses.append(entry['response'])
                    elif isinstance(entry, str):
                        self.all_responses.append(entry)

        # Prepare similarity matrix with sampling for large datasets
        if self.all_responses:
            sample_size = min(5000, len(self.all_responses))
            sample = random.sample(self.all_responses, sample_size) if len(
                self.all_responses) > sample_size else self.all_responses

            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
            self.tfidf_matrix = self.vectorizer.fit_transform(sample)

    def get_keyword_responses(self, keyword):
        """Get all responses for a specific keyword"""
        return self.data.get(keyword.lower(), [])

    def find_similar_responses(self, query, top_n=5):
        """Find similar responses using TF-IDF"""
        if not self.vectorizer or not query.strip():
            return None

        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = np.argpartition(similarities, -top_n)[-top_n:]
            return [self.all_responses[i] for i in top_indices if similarities[i] > 0.2]
        except Exception as e:
            print(f"Similarity search error: {e}")
            return None

    def get_random_response(self):
        """Get random response from dataset"""
        if self.all_responses:
            return random.choice(self.all_responses)
        return random.choice(self.get_fallback_data()['general'])

    def get_fallback_data(self):
        """Basic fallback dataset structure"""
        return {
            "general": [
                {"context": "", "response": "That's interesting. Tell me more about that."},
                {"context": "", "response": "How does that make you feel?"},
                {"context": "", "response": "I'd love to hear more about your thoughts on this."},
                {"context": "", "response": "What else comes to mind when you think about this?"},
                {"context": "", "response": "Can you elaborate on that point?"}
            ],
            "hello": [
                {"context": "", "response": "Hello there! How are you doing today?"},
                {"context": "", "response": "Hi! What's on your mind?"}
            ],
            "how are you": [
                {"context": "", "response": "I'm doing well, thanks for asking! How about you?"}
            ]
        }

# Initialize components once at startup
nlp = OptimizedNLP()
dataset = DatasetManager("cone03.json")

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint to handle chat requests"""
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({"response": "Please provide a message"}), 400
        
        # First check for keyword matches
        keyword_matches = []
        for keyword in dataset.data.keys():
            if keyword.lower() in user_input.lower():
                keyword_matches.extend(dataset.get_keyword_responses(keyword))

        # If we found keyword matches, select from those
        if keyword_matches:
            selected = random.choice(keyword_matches)
            base_response = selected['response'] if isinstance(selected, dict) else selected
        else:
            # Otherwise try semantic similarity
            similar_responses = dataset.find_similar_responses(user_input)
            if similar_responses:
                base_response = random.choice(similar_responses)
            else:
                # Fallback to random response
                base_response = dataset.get_random_response()

        # Enhance the response
        try:
            paraphrased = nlp.paraphrase(base_response)
            question = nlp.generate_question(paraphrased)
            full_response = f"{paraphrased} {question}"
        except Exception as e:
            print(f"Error enhancing response: {e}")
            full_response = base_response

        return jsonify({"response": full_response})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"response": "I encountered an error processing your request. Please try again."}), 500

@app.route('/')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "active",
        "service": "Chatbot API",
        "dataset_loaded": bool(dataset.all_responses)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
