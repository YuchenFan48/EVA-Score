import torch
from vllm import LLM, SamplingParams
import os

def paraphrase(args, triples):
    """
    Paraphrase the given triples using the Mistral-7B-Instruct-v0.3 model.
    
    Args:
        triples (List[List[str]]): A list of lists of triples, where each triple is a comma-separated string.
    
    Returns:
        List[List[str]]: A list of lists of paraphrased triples
    """
    prompt_path = os.path.join(args.prompt_dir, 'prompt_paraphrase.txt')
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()
    
    llm = LLM(model=args.paraphrase_model)
    
    all_prompts = []
    indices = []  

    for record_idx, triple_record in enumerate(triples):
        for triple_idx, triple in enumerate(triple_record):
            triple_list = triple.split(',')
            if len(triple_list) != 3:
                continue
            if (not any(c.isalpha() for c in triple_list[0]) or
                not any(c.isalpha() for c in triple_list[1]) or
                not any(c.isalpha() for c in triple_list[2])):
                continue
            
            triple_dict = {'h': triple_list[0], 'r': triple_list[1], 't': triple_list[2]}
            new_prompt = prompt_template + '\n' + str(triple_dict)
            all_prompts.append(new_prompt)
            indices.append((record_idx, triple_idx))
    
    sampling_params = SamplingParams(
        max_tokens=128,
        temperature=0.8,
        top_p=0.9
    )
    
    outputs = llm.generate(all_prompts, sampling_params=sampling_params)
    
    paraphrases = [[] for _ in range(len(triples))]
    
    for output, (record_idx, triple_idx) in zip(outputs, indices):
        generated_text = output.outputs[0].text
        
        if '[/INST]' in generated_text:
            generated_text = generated_text.split('[/INST]', 1)[1]
        if '(' in generated_text:
            generated_text = generated_text.split('(', 1)[0]
        
        result_line = generated_text.strip().split('\n')[0].strip()
        
        paraphrases[record_idx].append(result_line)
    
    return paraphrases