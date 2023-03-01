# -*- coding: utf-8 -*-
"""
@author: vjha1
"""

import warnings; warnings.filterwarnings('ignore')
import time
import re
import pandas as pd
# import ssl
# import sys
import json
import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

start = time.time()
wordlist = ["DisHonest", "InAccurate", "InCompetent", "InSane", "IrRational", "IrRationally","IrRationality", "MisUnderstood", "NonAdherent", "NonCompliant", "NonCooperative", "OwnFault", "TooSensitive", "UnMotivated)", "UnPleasant", "UnProfessional", "UnRelated", "Adherence", "Admits", "Aggressive", "Agitated", "Angry", "Anxious", "Apparently", "Argumentative", "Assumes", "Bad", "Challenging", "Claims", "Clueless", "Combative", "Complain", "Confront", "Crazy", "damant", "Defensive", "Depressed", "Difficult", "Dismiss", "Disruptive", "Elderly", "Exaggerate", "Hysterical", "Hysterical", "Incorrect or Inaccurate", "Insists", "Lists", "Lying", "Notes", "Obese", "Overreacting", "Paranoid", "Poor e.g., poor choice of…)", "Psycho", "Refuse", "Resist or Resistant", "Ruthless", "States", "Upset", "Wrong"]
wordlist = [word.lower() for word in wordlist]


subset = pd.read_csv('Labeled/KA_btwn_2_5_Labeled.csv')


from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
def is_positive(text: str) -> bool:
    """True if text has positive compound sentiment, False otherwise."""
    return sia.polarity_scores(text)["compound"]

def extract_quoted_text(text):
    return re.findall('"([^"]*)"', text)


def simple_preprocess(text):
    text = text.replace('\n\n','PAGE-BREAK')
    text = text.replace('\n','')
    text = text.replace('PAGE-BREAK','\n')
    return text

import nltk
def extract_pronoun_sentences(text):

    sentences = nltk.sent_tokenize(text)
    pronoun_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        for word, pos in pos_tags:
            if pos == 'PRP' or pos == 'PRP$':
                pronoun_sentences.append(sentence.replace(',','').replace('"',"").replace("'",""))
                break
    return pronoun_sentences

def extract_negative_actions(pronoun_sentences):
    negative_actions = []
    for p_sent in pronoun_sentences:
        sentences = nltk.sent_tokenize(p_sent)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
            for word, pos in pos_tags:
                if pos.startswith('VB') and is_positive(word) < 0:
                    negative_actions.append(word)   
    return negative_actions


def extract_taboo_matches(pronoun_sentences):
    taboo_words = []
    for sentence in pronoun_sentences:
        for taboo_word in wordlist:
            if taboo_word.lower() in sentence.lower():
                taboo_words.append(taboo_word)
#                 print(taboo_word,'=======', sentence)
#                 print('\n\n')
    return taboo_words

def extract_crucial_statements(sentences):
    crucial_statements = []
    for sentence in sentences:
        if is_positive(sentence) < 0:
            crucial_statements.append(sentence)
            
    return crucial_statements


def extract_emphasized_words(text):
    emphasized_words = []
    words = nltk.word_tokenize(' '.join(re.findall(r'\b[A-Z]+\b', raw_text)).lower())
    pos_tags = nltk.pos_tag(words)
    for word, pos in pos_tags:
        if len(word) > 3 and is_positive(word) < 0:
            emphasized_words.append(word) 
    return emphasized_words

def convert_text_to_features(raw_text, label):
    features = {}
    
    
    features['label'] = label

    
    text = simple_preprocess(raw_text)
    
    
    pronoun_sentences = extract_pronoun_sentences(text)
    
    
    crucial_statements = extract_crucial_statements(pronoun_sentences)
    features['crucial_statements'] = crucial_statements
    features['crucial_statements_count'] = len(crucial_statements)
    
    
    taboo_words = extract_taboo_matches(pronoun_sentences)
    features['taboo_words'] = taboo_words
    features['taboo_word_count'] = len(taboo_words)
    
    
    negative_actions = extract_negative_actions(pronoun_sentences)
    features['negative_actions_by_patient'] = negative_actions
    features['negative_actions_by_patient_count'] = len(negative_actions)
        
    
    quoted_text = extract_quoted_text(raw_text)
    features['quoted_text_found'] = quoted_text
    
    
    emphasized_words = extract_emphasized_words(raw_text)
    features['emphasized_words'] = emphasized_words


        
    return features

import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
    
results = []
for i in range(0,524):
    record = subset.iloc[i]
    raw_text = record['text']
    label = record['label (0 for Bias or 1 for unbiased)']

    result = convert_text_to_features(raw_text, label)
    results.append(result)   

end = time.time()
tot = end - start
print("\n" + str(tot))

#%% 


import csv

sfile = open('Labeled/resultFile.csv','w')
writer = csv.DictWriter(sfile, fieldnames=['crucial_statements','crucial_statements_count','emphasized_words','label','negative_actions_by_patient','negative_actions_by_patient_count','quoted_text_found','taboo_word_count','taboo_words'])
writer.writeheader()
writer.writerows(results)
sfile.close()

