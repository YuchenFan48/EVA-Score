import json
import os
import re
from tqdm import tqdm
from transfer_openai import chat_completion_use_cache
import logging
    
def post_process(ners):
    consolidated_entities = {}
    for entities in ners:
        if entities is None:
            continue
        for entity in entities:
            if isinstance(entity, str) and len(entity) > 1000:
                continue
            if isinstance(entity, str) and "text" in entity and "type" in entity:
                entity = json.loads(entity)
            if 'text' not in entity or 'type' not in entity:
                continue
            text = entity['text']
            ent_type = entity['type']

            consolidated_entities[text] = ent_type
    
    return consolidated_entities, True

def ner(args, text):
    prompt_path = os.path.join(args.prompt_dir, 'prompt_ner.txt')
    with open(prompt_path, 'r') as f:
        prompt = f.read()

    prompt += text
    try:
        response_json = chat_completion_use_cache(
            args, 
            prompt=prompt,
            model=args.ner_model,
            temperature=1
        ).replace('```json\n', '').replace('```', '')
        response = json.loads(response_json)
        return response
    except json.JSONDecodeError as e:
        print(f'Error: {e}')
        return None
    except Exception as e:
        print(f'Error: {e}')
        return None
    
def convert_to_docred(args, topics, generations):
    info = []
    total_entities = []
    logging.info("Extracting NERs")
    for topic, generation in tqdm(zip(topics, generations)):
        ners = []
        text = generation.split('\n')
        text = [t for t in text if t]
        for item in text:
            response = ner(args, item)
            cnt = 0
            while response is None and cnt < 3:
                cnt += 1
                response = ner(args, item)
            ners.append(response)
        checked_ners, success = post_process(ners)
        data = {
            'topic': topic,
            'text': text,
            'entities': checked_ners
        }
        total_entities.append(checked_ners)
        info.append(data)
    assert len(info) == len(topics)
    docred = []
    for data in info:
        paragraphs = [" ".join(data['text'])]
        for combined_text in paragraphs:
            vertexSet = []
            topic = data['topic']
            entities = data['entities']
            if not entities:
                continue
            sentences = [s.strip() + '.' for s in combined_text.split('.') if s.strip() != '']
            pattern = r"(\w+|[^\w\s])"
            words = [re.findall(pattern, s) for s in sentences]
            for entity_name, entity_type in entities.items():
                cur_vertexSet = []
                entity_name_word = re.findall(pattern, entity_name)
                entity_length = len(entity_name_word)
                for sent_id, word_list in enumerate(words):
                    for i in range(len(word_list) - entity_length + 1):
                        if word_list[i:i + entity_length] == entity_name_word:
                            cur_vertexSet.append({
                                "pos": [i, i + entity_length],
                                "type": entity_type,  
                                "sent_id": sent_id,
                                "name": entity_name
                            })
                if cur_vertexSet:
                    vertexSet.append(cur_vertexSet)
            
            if vertexSet:
                docred.append({
                    "vertexSet": vertexSet,
                    "title": topic,
                    "sents": words
                })
    logging.info("NER extraction completed")
    return docred, total_entities, info
