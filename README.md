# OpenAI's CLIP text encoder model in the Chinese Room

---

## Explanation of the Code

The project consists of two main scripts:

1. **`fine_tuned_model.py`**  
   - Loads and preprocesses the dataset (`IMBD DATASET.csv`).  
   - Uses **SBERT (all-MiniLM-L6-v2)** to generate semantic embeddings of plots.  
   - Fine-tunes the **CLIP text encoder** so that its embeddings align with SBERT’s space.  
   - A projection layer maps CLIP’s 512-dimensional embeddings to SBERT’s 384-dimensional space.  
   - The trained model and tokenizer are saved into the folder `clip-finetuned-sbert/`.

2. **`main.py`**  
   - Loads the dataset and a ground-truth dictionary of similar game titles manually constructed.  
   - Evaluates three models:
     - SBERT   
     - Fine-tuned CLIP Text Encoder   
     - Pre-trained CLIP Text Encoder  
   - Embeds the plots, retrieves the top-5 most similar items, and evaluates results using **BERTScore (Precision, Recall, F1)**.  
   - Prints average performance metrics.
## Dataset Files

- **IMBD DATASET.csv**  
  Contains the raw dataset of video game information collected from IMDB.  
  Main columns:  
  - `name`: title of the video game  
  - `plot`: short text description of the game’s plot
  - Geners
  - (other columns like `url`, `year`, `certificate`, `rating`, `votes` are dropped during preprocessing)  
  This file is used both for **training** (fine-tuning CLIP with SBERT embeddings) and **evaluation**.

- **ground truth.csv**  
  Contains a subset of games (20 video game).  
  It is used in **evaluation** to compare the retrieved plots against human-curated ground truth (We maually labeld the ground truth).  
  Each entry includes a `name` and its `plot` and a text column. The related titles are mapped in the `ground_truth` dictionary inside `main.py`.


This structure separates **training** (fine-tuning CLIP) and **evaluation** (comparing models).

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
python "fine_tuned_model.py"

```
2.Evaluate The models:

Run this file script to compare the three models: SBERT, Fine-tuned CLIP model and the Pre-traind CLIP model

```bash
python main.py
```
## Results

After running main.py, the console will display the average BERTScore Precision, Recall, and F1 for each model, allowing direct comparison.
