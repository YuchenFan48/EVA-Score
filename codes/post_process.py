from copy import deepcopy  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def filter_triples(original_list):
    unique_list = []
    seen = set()
    for item in original_list:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list


def get_cosine_similarity(paraphrases, references):
    all_sentences = [s for sublist in paraphrases + references for s in sublist]
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_sentences)
    
    sentence_to_tfidf = {sentence: tfidf_matrix[i] for i, sentence in enumerate(all_sentences)}
    
    scores = []
    for paraphrase, reference in zip(paraphrases, references):
        score = []
        for s_para in paraphrase:
            pair_scores = []
            for s_ref in reference:
                if s_para == s_ref:
                    pair_scores.append(1.0)
                else:
                    cosine_sim = cosine_similarity(sentence_to_tfidf[s_para], sentence_to_tfidf[s_ref])
                    pair_scores.append(cosine_sim[0][0])
            score.append(pair_scores)
        scores.append(score)
    return scores

def filter_text(paraphrases, references):   
    filtered = []
    cosine_scores = get_cosine_similarity(paraphrases, references)
    for paragraph, reference, cosine_score in zip(paraphrases, references, cosine_scores):
        cur = []
        for i, c_score in zip(range(len(paragraph)), cosine_score):
            max_c_score = max(c_score)
            max_c_score_idx = c_score.index(max_c_score)
            if max_c_score < 0.65:
                cur.append(paragraph[i])
        filtered.append(cur)
    return filtered
                