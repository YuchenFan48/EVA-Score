from transfer_openai import chat_completion_use_cache
import os

def docre(args, text, entities):
    prompt_path = os.path.join(args.prompt_dir, 'prompt_docre.txt')
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    prompt += '\n\n' + text + '\n\n' + entities
    response = chat_completion_use_cache(args, prompt, model="gpt-4-0125-preview")
    