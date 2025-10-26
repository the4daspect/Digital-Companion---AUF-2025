!pip install bert-score
!pip install nltk
!pip install rouge-score

from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.meteor_score import meteor_score, single_meteor_score
from rouge_score import rouge_scorer
import nltk
import re
import pandas as pd

pattern = re.compile(r"[^A-Za-z0-9 '’]")

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

def calculate_bertscore(candidate, reference):
  scorer = BERTScorer(model_type='bert-base-uncased')
  P, R, F1 = scorer.score([candidate], [reference])
  return F1.mean()

def calculate_bleu(answer, standard1, standard2):
  candidate = re.sub(pattern, "", answer).split()
  reference1 = re.sub(pattern, "", standard1).split()
  reference2 = re.sub(pattern, "", standard2).split()
  bleu_score = sentence_bleu([reference1, reference2], candidate)
  return bleu_score

def calculate_rouge_L(candidate, reference):
  scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
  rouge_L_score = scorer.score(reference, candidate)
  return rouge_L_score['rougeL'].fmeasure

def calculate_meteor(answer, standard):
  reference = re.sub(pattern, "", standard).split()
  candidate = re.sub(pattern, "", answer).split()
    # reference = nltk.word_tokenize(standard)
    # candidate = nltk.word_tokenize(answer)
  meteor_score = single_meteor_score(reference, candidate)
  return meteor_score
