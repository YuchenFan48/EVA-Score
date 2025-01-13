import openai
from openai import OpenAI
import time
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("requests").setLevel(logging.WARNING)
def chat_completion_use_cache(args, prompt, model, temperature=1.0,patience=10, sleep_time=0, use_json=False):
    while patience > 0:
        patience -= 1
        client = OpenAI(
    api_key=args.openai_api_key,base_url=args.openai_api_base
    )
    messages = [
        {'role': 'user', 'content': prompt}
    ]
    try:
        response = client.chat.completions.create(model=model,
                            messages=messages,
                            temperature=temperature)
        prediction = response.choices[0].message.content.strip()
        if prediction != "" and prediction != None:
            return prediction
        else:
            time.sleep(sleep_time)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(sleep_time)

    return ""