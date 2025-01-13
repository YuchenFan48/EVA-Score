import argparse
import json
import logging
import os
import re
import time

import torch
from factscore.factscorer import FactScorer
from nli import make_pair, is_entailment, is_matched
from paraphrase import paraphrase
from post_process import filter_text, filter_triples
from transfer_openai import chat_completion_use_cache
from convert_to_docred import convert_to_docred
from mask import combine_cascade

# Configure logging levels
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

def get_atomic_facts(args, docred_path, topics, generations):
    """
    Extract atomic facts from the provided topics and generations.
    """
    logging.info("Converting to DocRED format")
    docred, total_entities, info = convert_to_docred(args, topics, generations)

    logging.info("Begin document-level relation extraction")
    decoded_docred = []

    # Use prompt file path from args
    prompt_template_path = os.path.join(args.prompt_dir, 'prompt_docre.txt')

    for item in info:
        cur_docred = []
        text = item["text"]
        entities = item["entities"]

        with open(prompt_template_path, 'r') as f:
            prompt = f.read()
        prompt += f"\n\nText:\n{text}\n\nEntities:\n{entities}"

        response = chat_completion_use_cache(args, prompt, model=args.docre_model, temperature=0.7)
        while response is None:
            response = chat_completion_use_cache(args, prompt, model=args.docre_model, temperature=0.7)

        matches = re.findall(r'([^)]+)', response)
        for match in matches:
            cur_docred.append(match.replace('(', ''))
        decoded_docred.append(cur_docred)
    logging.info("End document-level relation extraction")

    logging.info("Begin Document-level Filtering and Paraphrasing")
    new_decoded_docred = [filter_triples(relations) for relations in decoded_docred]
    texts = paraphrase(args, new_decoded_docred)
    atomic_facts = [text for text in texts]
    logging.info("End Document-level Filtering and Paraphrasing")

    logging.info("Begin atomic fact extraction")
    fs = FactScorer()
    out = fs.get_fact(topics, generations, verbose=True)

    filtered = filter_text(atomic_facts, out)
    entails = is_entailment(
        topics, generations,
        args.nli_model_path,
        out
    )
    dataset = make_pair(entails)
    logging.info("End atomic fact extraction")

    new_atomic_facts = combine_cascade(dataset, threshold=args.cascade_threshold)
    new_filtered_facts = [[fact] for fact in filtered]

    facts = []
    for sentence_fact, document_fact in zip(new_atomic_facts, new_filtered_facts):
        facts.append({
            "sentence_fact": sentence_fact,
            "document_fact": document_fact
        })

    assert len(facts) == len(topics), "Length of facts and topics should be the same"
    return facts, total_entities

def calculate_recall(args, docred_path, topics, generations, model_answers, sentence_decisions):
    """
    Calculate recall and precision metrics for sentence-level decisions.
    """
    logging.info("Preprocess for golden generation")
    atomic_facts, entities = get_atomic_facts(args, docred_path, topics, generations)
    
    sentence_reference = [s["sentence_fact"] for s in atomic_facts]
    document_reference = [s["document_fact"] for s in atomic_facts]

    all_reference = []
    for sentence_fa, document_fa in zip(sentence_reference, document_reference):
        for fact in document_fa:
            for f in fact:
                sentence_fa.append([f])
        all_reference.append(sentence_fa)

    sentence_all_reference_length = [
        len([s for sublist in sentence for s in sublist]) for sentence in all_reference
    ]
    sentence_all_decision_length = [
        len([s for sublist in decision for s in sublist]) for decision in sentence_decisions
    ]
    document_all_reference_length = [
        len([s for sublist in document for s in sublist]) for document in document_reference
    ]

    sentence_results, _ = is_matched(
        args.llm_path,
        sentence_decisions,
        all_reference
    )

    sentence_true_hyps = []
    sentence_recall_values = []
    sentence_precision_values = []
    sentence_hyp_ref_mapping = []

    for i, sentence_result in enumerate(sentence_results):
        mapping = []
        found_hyps = []
        precision = 0
        reference_indices = [0] * sentence_all_reference_length[i]

        for values in sentence_result:
            cur_hyps = []
            for v in values:
                for index, value in enumerate(v):
                    if value['entailment']:
                        reference_indices[index] = 1
                        mapping.append(f"{value['hyp']} -> {value['ref']}")
                        if value['hyp'] not in cur_hyps:
                            cur_hyps.append(value['hyp'])
            if cur_hyps:
                found_hyps.append(cur_hyps[-1])
            precision += len(cur_hyps)

        sentence_hyp_ref_mapping.append(mapping)
        sentence_true_hyps.append(found_hyps)

        new_precision = (precision / sentence_all_decision_length[i]
                         if sentence_all_decision_length[i] else 0)
        new_recall = (sum(reference_indices) / sentence_all_reference_length[i]
                      if sentence_all_reference_length[i] else 0)

        assert new_precision <= 1 and new_recall <= 1
        print(new_precision, new_recall)

        sentence_recall_values.append(new_recall)
        sentence_precision_values.append(new_precision)

    sentence_overall_recall = (sum(sentence_recall_values) / len(sentence_recall_values)
                               if sentence_recall_values else 0)
    sentence_overall_precision = (sum(sentence_precision_values) / len(sentence_precision_values)
                                  if sentence_precision_values else 0)

    return {
        "sentence_recall": sentence_overall_recall,
        "sentence_precision": sentence_overall_precision,
        "sentence_true_hyps": sentence_true_hyps,
        "sentence_reference": sentence_reference,
        "sentence_hyp_ref_mapping": sentence_hyp_ref_mapping,
        "sentence_part_recall": sentence_recall_values,
        "sentence_part_precision": sentence_precision_values
    }

def main():
    """Main function to execute processing and evaluation."""
    parser = argparse.ArgumentParser()

    # Input and output paths
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input file')
    parser.add_argument('--docred_dir', type=str, default='data/labeled', help='Path to the converted DocRED directory')
    parser.add_argument('--output_dir', type=str, default='data/labeled/annotated', help='Output directory')
    parser.add_argument('--result_dir', type=str, default='data/results', help='Path to the result directory')

    # Model and prompt configurations
    parser.add_argument('--ner_model', type=str, default=None, help='Path to the NER model')
    parser.add_argument('--docre_model', type=str, default=None, help='Path to the DocRE model')
    parser.add_argument('--nli_model_path', type=str, default=None, help='Path to the LLM Validation model')
    parser.add_argument('--paraphrase_model', type=str, default=None, help='Path to the paraphrase model')
    parser.add_argument('--llm_path', type=str, default=None, help='Path to the LLM model')
    parser.add_argument('--prompt_dir', type=str, default='../prompts', help='Path to the prompts')
    parser.add_argument('--cascade_threshold', type=float, default=0.65, help='Threshold for cascade combination')
    parser.add_argument('--task', type=str, default='docred', help='Task name')
    
    # OpenAI arguments
    parser.add_argument('--openai_api_key', type=str, default=None, help='OpenAI API key')
    parser.add_argument('--openai_api_base', type=str, default=None, help='OpenAI API base URL')

    args = parser.parse_args()

    # Configure logger
    logging.basicConfig(level=logging.INFO)

    # Set up paths
    args.docred_dir = os.path.join(args.docred_dir, args.task)
    triples_jsonfile = os.path.join(args.docred_dir, 'triples')
    args.output_dir = os.path.join(args.output_dir, args.task)

    file_name = os.path.splitext(os.path.basename(args.input_path))[0]
    args.result_dir = os.path.join(args.result_dir, args.task)
    result_path = os.path.join(args.result_dir, f"{file_name}.json")
    docred_path = os.path.join(args.docred_dir, f"{file_name}.json")

    # Create necessary directories
    os.makedirs(triples_jsonfile, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    # Initialize result file if it doesn't exist
    if not os.path.isfile(result_path):
        with open(result_path, 'w') as f:
            json.dump({}, f)

    topics, generations, contexts = [], [], []
    with open(args.input_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            topics.append(item["text"])
            generations.append(item["model_pred"])
            contexts.append(item["gold_answer"])

    topics = topics
    generations = generations
    contexts = contexts
    logging.info("Preprocess for candidate generation")
    atomic_facts, total_entities = get_atomic_facts(args, docred_path, topics, generations)
    info = []

    for atomic_fact, total_entity, context, topic, generation in zip(
            atomic_facts, total_entities, contexts, topics, generations):
        info.append({
            "sentence_atomic_fact": atomic_fact["sentence_fact"],
            "document_atomic_fact": atomic_fact["document_fact"],
            "total_entities": total_entity,
            "context": context,
            "text": topic,
            "model_pred": generation
        })

    input_base = os.path.splitext(args.input_path)[0]
    file_path = f"{input_base}_atomic.jsonl"
    with open(file_path, 'w') as f:
        for item in info:
            f.write(json.dumps(item) + '\n')

    new_sentence_facts = [s["sentence_fact"] for s in atomic_facts]
    new_filtered_facts = [s["document_fact"] for s in atomic_facts]
    final_atomic_facts = []

    for sentence_fact, document_fact in zip(new_sentence_facts, new_filtered_facts):
        for fact in document_fact:
            for f in fact:
                sentence_fact.append([f])
        final_atomic_facts.append(sentence_fact)

    assert len(new_sentence_facts) == len(new_filtered_facts)

    sentence_results = calculate_recall(
        args, docred_path, topics, generations,
        generations, final_atomic_facts
    )

    # Extract metrics from results
    sentence_recall = sentence_results["sentence_recall"]
    sentence_precision = sentence_results["sentence_precision"]
    sentence_true_hyps = sentence_results["sentence_true_hyps"]
    sentence_reference = sentence_results["sentence_reference"]
    sentence_hyp_ref_mapping = sentence_results["sentence_hyp_ref_mapping"]
    sentence_part_recall = sentence_results["sentence_part_recall"]
    sentence_part_precision = sentence_results["sentence_part_precision"]

    print("Sentence Recall: ", sentence_recall)
    print("Sentence Precision: ", sentence_precision)
    f1_score = 2 * sentence_recall * sentence_precision / (sentence_recall + sentence_precision + 1e-8)
    print("Sentence F1: ", f1_score)

    for data, sentence_ref, true_hyp, hyp_ref, part_re, part_pre in zip(
            info, sentence_reference, sentence_true_hyps,
            sentence_hyp_ref_mapping, sentence_part_recall, sentence_part_precision):
        data.update({
            "sentence_true_hyps": true_hyp,
            "sentence_reference": sentence_ref,
            "sentence_hyp_ref_mapping": hyp_ref,
            "sentence_part_recall": part_re,
            "sentence_part_precision": part_pre
        })

    with open(result_path, 'w') as f:
        for item in info:
            f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    main()