# OpenAI's CLIP text encoder model in the Chinese Room


---

## Requirements
Install dependencies with:

```bash
pip install -r requirements
```
## How to Run
1.Fine-tune the CLIP model

Run the training script to align CLIP embeddings with SBERT space and to produce a folder clip-finetuned-sbert containing the fine-tuned model and tokenizer:

```bash
python "fine tuned model.py"

```
2. main.py:

Run this file script to compare the three models: 1.SBERT 2.Fine-tuned CLIP model 3.pre-traind CLIP model

```bash
python main.py
```
## Results

After running main.py, the console will display the average BERTScore Precision, Recall, and F1 for each model, allowing direct comparison.
