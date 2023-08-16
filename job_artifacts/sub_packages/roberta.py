import tensorflow as tf
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import os

def preprocess(transcription):
        new_text = []
        for t in transcription.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    
    
def roberta_base(new_text):
    
    text = new_text

    #delete models if there (each Job run should not have, but to be sure)
    os.system("rm -r ./cardiffnlp/twitter-roberta-base-offensive")
    os.system("rm -r ./cardiffnlp/twitter-roberta-base-hate")

    #define offensive and hate model
    MODEL_OFFENSIVE = f"cardiffnlp/twitter-roberta-base-offensive"
    MODEL_HATE = f"cardiffnlp/twitter-roberta-base-hate"
    
    #load tokenizers
    tokenizer_offensive = AutoTokenizer.from_pretrained(MODEL_OFFENSIVE)
    tokenizer_hate = AutoTokenizer.from_pretrained(MODEL_HATE)

    #label mapping
    labels_offensive=[]
    mapping_link_offensive = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/offensive/mapping.txt"
    with urllib.request.urlopen(mapping_link_offensive) as f:
        html_offensive = f.read().decode('utf-8').split("\n")
        csvreader_offensive = csv.reader(html_offensive, delimiter='\t')
    labels_offensive = [row[1] for row in csvreader_offensive if len(row) > 1]

    labels_hate=[]
    mapping_link_hate = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/hate/mapping.txt"
    with urllib.request.urlopen(mapping_link_hate) as f:
        html_hate = f.read().decode('utf-8').split("\n")
        csvreader_hate = csv.reader(html_hate, delimiter='\t')
    labels_hate = [row[1] for row in csvreader_hate if len(row) > 1]       
    
    # PT
    model_offensive = AutoModelForSequenceClassification.from_pretrained(MODEL_OFFENSIVE)
    model_offensive.save_pretrained(MODEL_OFFENSIVE)
    
    model_hate = AutoModelForSequenceClassification.from_pretrained(MODEL_HATE)
    model_hate.save_pretrained(MODEL_HATE)
    
    #tokenizer text
    encoded_input_offensive = tokenizer_offensive(text, return_tensors='pt')
    encoded_input_hate = tokenizer_hate(text, return_tensors='pt')
    
    output_offensive = model_offensive(**encoded_input_offensive)
    output_hate = model_hate(**encoded_input_hate)
        
    scores_offensive = output_offensive[0][0].detach().numpy()
    scores_hate = output_hate[0][0].detach().numpy()
    
    scores_offensive = softmax(scores_offensive)
    scores_hate = softmax(scores_hate)

    non_offensive = scores_offensive[0]
    offensive = scores_offensive[1]
    
    non_hate = scores_hate[0]
    hate = scores_hate[1]

    print("non_offensive score = " + str(non_offensive))
    print("offensive score = " + str(offensive))
    print("non_hate score = " + str(non_hate))
    print("hate score = " + str(hate))
    
    return non_offensive, offensive, non_hate, hate
