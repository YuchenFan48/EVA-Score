#!/bin/bash

export OPENAI_API_KEY=""
export OPENAI_API_BASE=""

# Define paths and parameters
INPUT_PATH="../toy.jsonl"
DOCRED_DIR="data/labeled"
OUTPUT_DIR="data/labeled/annotated"
RESULT_DIR="data/results"

# Model and prompt configurations
NER_MODEL="gpt-4o-mini"            # Set to actual path or leave blank if not used
DOCRE_MODEL="gpt-4o"        # Set to actual path or leave blank if not used
NLI_MODEL_PATH="Mistral-7B-Instruct-v0.3"       # Set to actual path or leave blank if not used
PARAPHRASE_MODEL="Mistral-7B-Instruct-v0.3"  # Set to actual path or leave blank if not used
LLM_PATH="Mistral-7B-Instruct-v0.3"             # Set to actual path or leave blank if not used
PROMPT_DIR="../prompts"
CASCADE_THRESHOLD=0.65
TASK="docred"

# OpenAI API settings (if using OpenAI)
OPENAI_API_KEY_VAR=${OPENAI_API_KEY}
OPENAI_API_BASE_VAR=${OPENAI_API_BASE}

# Run the Python script with the arguments
python main.py \
  --input_path "$INPUT_PATH" \
  --docred_dir "$DOCRED_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --result_dir "$RESULT_DIR" \
  --ner_model "$NER_MODEL" \
  --docre_model "$DOCRE_MODEL" \
  --nli_model_path "$NLI_MODEL_PATH" \
  --paraphrase_model "$PARAPHRASE_MODEL" \
  --llm_path "$LLM_PATH" \
  --prompt_dir "$PROMPT_DIR" \
  --cascade_threshold "$CASCADE_THRESHOLD" \
  --task "$TASK" \
  --openai_api_key "$OPENAI_API_KEY_VAR" \
  --openai_api_base "$OPENAI_API_BASE_VAR"