import json
import os
import time
import torch
import string
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

from vllm import LLM, SamplingParams

def is_entailment(args, topics, generations, model_name, annotations):
    """
    Check if the hypothesis entails the reference.
    
    Args:
        topics (List[str]): A list of topics.
        generations (List[str]): A list of generations.
        model_name (str): The model name.
        annotations (List[List[str]]): A list of lists of annotations.
        
    Returns:
        List[Dict]: A list of dictionaries, each containing the following keys:
        'topic', 'generation', 'annotations', 'results'.
    """
    llm = LLM(model=model_name, tensor_parallel_size=1)  

    records = []

    all_prompts = []
    pair_indices = [] 

    for i_topic, (topic, generation, ann_list) in enumerate(zip(topics, generations, annotations)):
        hyps = ann_list[:-1]
        refs = ann_list[1:]
        prompt_path = os.path.join(args.prompt_dir, "prompt_nli.txt")
        for i_pair, (h, r) in enumerate(zip(hyps, refs)):
            with open(prompt_path, 'r') as f:
                template = f.read()

            final_prompt = (
                template
                + "\n\nHypothesis: " + h
                + "\n\nReference: " + r
            )

            all_prompts.append(final_prompt)
            # 在 pair_indices 里记录 (i_topic, i_pair, h, r)，以便后面映射
            pair_indices.append((i_topic, i_pair, h, r))

    if not all_prompts:
        return []

    sampling_params = SamplingParams(
        max_tokens=16,   
        temperature=0.8,
        top_p=0.9,       
    )
    outputs = llm.generate(all_prompts, sampling_params=sampling_params)
    results_by_topic = [[] for _ in range(len(topics))]

    for output, (i_topic, i_pair, h, r) in zip(outputs, pair_indices):
        generated_text = output.outputs[0].text  
        if '[/INST]' in generated_text:
            generated_text = generated_text.split('[/INST]', 1)[-1]
        if '(' in generated_text:
            generated_text = generated_text.split('(', 1)[0]

        is_entailed = ("true" in generated_text.lower())

        results_by_topic[i_topic].append({
            "ref": r,
            "hyp": h,
            "entailment": is_entailed
        })

    for i_topic, (topic, generation, ann_list) in enumerate(zip(topics, generations, annotations)):
        record = {
            'topic': topic,
            'generation': generation,
            'annotations': ann_list,
            'results': results_by_topic[i_topic]
        }
        records.append(record)

    return records

def is_matched(args, model_name, hyp_list, ref_list):
    """
    Check if the hypothesis matches the reference.
    
    Args:
        model_name (str): The model name.
        hyp_list (List[List[str]]): A list of lists of hypotheses.
        ref_list (List[List[str]]): A list of lists of references.
        
    Returns:    
        List[List[List[Dict]]]: A list of lists of lists of dictionaries, each containing the following keys:
        'ref', 'hyp', 'entailment'. 
    """

    llm = LLM(model=model_name, tensor_parallel_size=1)
    embed_model = SentenceTransformer('stsb-roberta-large')
    prompt_path = os.path.join(args.prompt_dir, "prompt_chain.txt")
    final_results = []
    times = []

    assert len(hyp_list) == len(ref_list), "hyp_list 和 ref_list 在长度上要对应"

    for hyps, refs in tqdm(zip(hyp_list, ref_list), total=len(hyp_list)):
        begin_time = time.time()

        real_refs = []
        for r_chain in refs:
            if r_chain:
                real_refs.append(r_chain[-1])
            else:
                real_refs.append("")  

        ref_len = len(real_refs)
        if ref_len == 0:
            final_results.append([])
            times.append(time.time() - begin_time)
            continue
        
        final_results_item = []

        for hyp_chain in hyps:
            if not hyp_chain:
                final_results_item.append([])
                continue
            
            most_informative_hyp = hyp_chain[-1]
            similar_refs = find_top_refs(
                embed_model,
                most_informative_hyp,
                real_refs,
                top_k=min(5, ref_len)
            )

            chain_results = []
            previous_hyps_formatted = "" 

            for hyp_idx, hyp in enumerate(hyp_chain):
                batch_prompts = []
                partial_result_for_this_hyp = [None] * len(similar_refs)

                for i_ref, (ref_text, ref_index) in enumerate(similar_refs):
                    with open(prompt_path, 'r') as f:
                        template_chain = f.read()

                    final_prompt = template_chain
                    if previous_hyps_formatted.strip():
                        final_prompt += f"\n\nPrevious facts were: {previous_hyps_formatted}"

                    final_prompt += f"\n\nHypothesis: {hyp}\n\nReference: {ref_text}"

                    batch_prompts.append(final_prompt)

                if not batch_prompts:
                    chain_results.append([])
                    continue

                sampling_params = SamplingParams(
                    max_tokens=16,
                    temperature=0.8,
                    top_p=0.9,
                )
                batch_outputs = llm.generate(batch_prompts, sampling_params=sampling_params)

                for i_ref, out in enumerate(batch_outputs):
                    generated_text = out.outputs[0].text
                    if '[/INST]' in generated_text:
                        generated_text = generated_text.split('[/INST]', 1)[-1]
                    if '(' in generated_text:
                        generated_text = generated_text.split('(', 1)[0]

                    is_entailed = ("true" in generated_text.lower())
                    ref_text, ref_index = similar_refs[i_ref]

                    partial_result_for_this_hyp[i_ref] = {
                        "ref": ref_text,
                        "hyp": hyp,
                        "entailment": is_entailed
                    }
                    if is_entailed:
                        pass


                any_true = any(x["entailment"] for x in partial_result_for_this_hyp if x is not None)
                if any_true:
                    previous_hyps_formatted += f"{hyp} (correct); "
                else:
                    previous_hyps_formatted += f"{hyp} (incorrect); "

                cur_result = [None] * ref_len
                for i_ref, (ref_text, ref_idx) in enumerate(similar_refs):
                    cur_result[ref_idx] = partial_result_for_this_hyp[i_ref]

                for i_all_ref in range(ref_len):
                    if cur_result[i_all_ref] is None:
                        cur_result[i_all_ref] = {
                            "ref": real_refs[i_all_ref],
                            "hyp": hyp,
                            "entailment": False
                        }
                
                chain_results.append(cur_result)
            
            final_results_item.append(chain_results)

        final_results.append(final_results_item)

        end_time = time.time()
        times.append(end_time - begin_time)

    return final_results, times

    


# ========== find_top_refs ==========

def find_top_refs(embed_model, hyp, ref_list, top_k=3):
    """
    Make a list of the top k references that are most similar to the hypothesis.
    
    Args:
        embed_model (SentenceTransformer.SentenceTransformer): A SentenceTransformer model.
        hyp (str): The hypothesis text.
        ref_list (List[str]): A list of reference texts.
        top_k (int): The number of top references to return.
        
    Returns:
        List[List[str, int]]: A list of the top k references, each with its index in the original list.
    """
    hyp_embedding = embed_model.encode(hyp, convert_to_tensor=True)
    ref_embeddings = embed_model.encode(ref_list, convert_to_tensor=True)

    cos_scores = util.pytorch_cos_sim(hyp_embedding, ref_embeddings)[0]
    top_k_indices = torch.topk(cos_scores, k=top_k).indices.tolist()
    top_refs_with_indices = [[ref_list[idx], idx] for idx in top_k_indices]

    return top_refs_with_indices


# ========== make_pair ==========

def make_pair(dataset):
    """
    Pair the NLI results with the annotations.
    
    Args:
        dataset (List[Dict]): A list of dictionaries, each containing the following
        keys: 'topic', 'generation', 'annotations', 'results'.

    Returns:
        List[Dict]: A list of dictionaries, each containing the following keys:
        'topic', 'generation', 'annotations', 'results', 'pairs
    """
    info = []
    for line_data in dataset:
        nli = line_data.get('results', [])
        nli_bools = [item["entailment"] for item in nli]
        annotations = line_data.get('annotations', [])

        hyps = annotations[:-1] 
        refs = annotations[1:] 

        pairs = []
        for nli_score, hyp, ref in zip(nli_bools, hyps, refs):
            pairs.append({'hyp': hyp, 'ref': ref, 'nli': nli_score})
        line_data['pairs'] = pairs
        info.append(line_data)

    return info