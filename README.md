# ics435-assignment4

## Part1 - 

### Requirements
pip install torch transformers datasets matplotlib

### Usage


## Part2 - GPT-2 Joke Fine-Tuning
Fine-tunes GPT-2 to generate a complete joke from any 3 starting words.

### Requirements
pip install transformers torch accelerate

### Usage
-  Train the model
python gpt2_joke_finetune.py --train
- Or with more epochs for better quality:
python gpt2_joke_finetune.py --train --epochs 10
- Generate jokes
python gpt2_joke_finetune.py --generate --prompt "What did the"
python gpt2_joke_finetune.py --generate --prompt "Why did the chicken"