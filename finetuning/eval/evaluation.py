from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import re
import pandas as pd
from nltk.translate.nist_score import corpus_nist
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize

pattern = re.compile(r"[^A-Za-z0-9 '’]")

def calculate_bertscore(candidate, ground_truth):
    scorer = BERTScorer(model_type='bert-base-uncased')

    P, R, F1 = scorer.score([candidate], [ground_truth])

    return F1.mean()

# def calculate_bleu(candidate, ground_truth):
#     pattern = re.compile(r"[^A-Za-z0-9 '’]")

#     candidate = re.sub(pattern, "", candidate).split()
#     ground_truth = re.sub(pattern, "", ground_truth).split()

#     reference = [
#         ground_truth,
#         ground_truth,
#         ground_truth,
#         ground_truth
#     ]

#     return sentence_bleu(reference, candidate)

def calculate_bleu(candidate, standard1, standard2):
  candidate = re.sub(pattern, "", candidate).split()
  reference1 = re.sub(pattern, "", standard1).split()
  reference2 = re.sub(pattern, "", standard2).split()
  bleu_score = sentence_bleu([reference1, reference2], candidate)

  return bleu_score

def calculate_rouge_L(candidate, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_L_score = scorer.score(candidate, ground_truth)

    return rouge_L_score['rougeL'].fmeasure

# def calculate_meteor(candidate, ground_truth):
#     return meteor_score([word_tokenize(ground_truth)], word_tokenize(candidate), 1)

if __name__ == "__main__":
    deepseek_df = pd.read_csv("xuelong/deepseek_baseline1.csv")
    gemma_df = pd.read_csv("xuelong/gemma_baseline1.csv")
    qwen_df = pd.read_csv("xuelong/qwen_baseline1.csv")

    # # calculate the average number of tokens in each cell of the 'Response' column
    # avg_tokens = deepseek_df['Response'].apply(lambda x: len(str(x).split())).mean()
    # print("Average number of tokens in 'Response':", avg_tokens)

    for i, row in deepseek_df.iterrows():
        candidate = row['Answer']
        ground_truth = row['Response']

        # stupid but quick fix
        try:
            bertscore = calculate_bertscore(candidate, ground_truth)
        except:
            bertscore = 0.0
        try:
            bleu = calculate_bleu(candidate, ground_truth, ground_truth)
        except:
            bleu = 0.0
        try:    
            rouge_L = calculate_rouge_L(candidate, ground_truth)
        except:
            rouge_L = 0.0
        # try:
        #     meteor = calculate_meteor(candidate, ground_truth)
        # except:
        #     meteor = 0.0
        
        deepseek_df.at[i, 'BERTScore'] = bertscore
        deepseek_df.at[i, 'BLEU'] = bleu
        deepseek_df.at[i, 'ROUGE-L'] = rouge_L
        # deepseek_df.at[i, 'METEOR'] = meteor

    deepseek_df.to_csv("xuelong/deepseek_baseline1_scores.csv", index=False)

    print(deepseek_df['BERTScore'].mean(), deepseek_df['BLEU'].mean(), deepseek_df['ROUGE-L'].mean())

    print(deepseek_df['METEOR'].mean())

    for i, row in gemma_df.iterrows():
        candidate = row['Answer']
        ground_truth = row['Response']

        try:
            bertscore = calculate_bertscore(candidate, ground_truth)
        except:
            bertscore = 0.0
        try:
            bleu = calculate_bleu(candidate, ground_truth, ground_truth)
        except:
            bleu = 0.0
        try:    
            rouge_L = calculate_rouge_L(candidate, ground_truth)
        except:
            rouge_L = 0.0
        # try:
        #     meteor = calculate_meteor(candidate, ground_truth)
        # except:
        #     meteor = 0.0

        gemma_df.at[i, 'BERTScore'] = bertscore
        gemma_df.at[i, 'BLEU'] = bleu
        gemma_df.at[i, 'ROUGE-L'] = rouge_L
        # gemma_df.at[i, 'METEOR'] = meteor
    
    gemma_df.to_csv("xuelong/gemma_baseline1_scores.csv", index=False)

    print(gemma_df['BERTScore'].mean(), gemma_df['BLEU'].mean(), gemma_df['ROUGE-L'].mean())

    print(gemma_df['METEOR'].mean())

    for i, row in qwen_df.iterrows():
        candidate = row['Answer']
        ground_truth = row['Response']

        try:
            bertscore = calculate_bertscore(candidate, ground_truth)
        except:
            bertscore = 0.0
        try:
            bleu = calculate_bleu(candidate, ground_truth, ground_truth)
        except:
            bleu = 0.0
        try:    
            rouge_L = calculate_rouge_L(candidate, ground_truth)
        except:
            rouge_L = 0.0
        # try:
        #     meteor = calculate_meteor(candidate, ground_truth)
        # except:
        #     meteor = 0.0

        qwen_df.at[i, 'BERTScore'] = bertscore
        qwen_df.at[i, 'BLEU'] = bleu
        qwen_df.at[i, 'ROUGE-L'] = rouge_L
        # qwen_df.at[i, 'METEOR'] = meteor

    qwen_df.to_csv("xuelong/qwen_baseline1_scores.csv", index=False)

    print(qwen_df['BERTScore'].mean(), qwen_df['BLEU'].mean(), qwen_df['ROUGE-L'].mean())

    print(qwen_df['METEOR'].mean())
