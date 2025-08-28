
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from bert_score import score as bertscore_score
import torch
from sentence_transformers import util
import numpy as np
import torch.nn.functional as F



# Ground truth data for evaluation, which maps game titles to their related titles.
GROUND_TRUTH_SiZE = 20

ground_truth ={
    "Down in Bermuda": [
        "Starbo",
        "Escape Rosecliff Island",
        "Jimmy's Island Adventure",
        "Farpoint",
        "Lost Pilot"
    ],
    "Star Trek: Deep Space Nine - Dominion Wars": [
        "Star Trek: Armada",
        "Star Trek",
        "Star Trek: Birth of the Federation",
        "Star Wars Commander",
        "Star Trek: Generations"
    ],
    "Have a Nice Death": [
        "Peace, Death!",
        "Die in Style",
        "Death Call",
        "Let It Die",
        "The Typing of the Dead: Overkill"
    ],
    "Road of the Dead 2": [
        "Road of the Dead",
        "Last Stand 2",
        "Deadhunt",
        "How to Survive 2",
        "The Last Stand"
    ],
    "Ni no Kuni: Wrath of the White Witch": [
        "Final Fantasy IX",
        "Myths of the World: Love Beyond Collector's Edition",
        "Max: The Curse of Brotherhood",
        "Gekido",
        "Deception III: Dark Delusion"
    ],
    "Spider-Man": [
        "Spider-Man",
        "The Amazing Spider-Man 2",
        "The Amazing Spider-Man vs. The Kingpin",
        "The Amazing Spider-Man",
        "Spider-Man 2: Enter Electro"
    ],
    "Postal 2: Paradise Lost": [
        "Postal III",
        "Postal 2: Apocalypse Weekend",
        "Fallout: Miami",
        "Fallout 2: A Post-Nuclear Role-Playing Game",
        "Wasteland 3"
    ],
    "Papa's Donuteria": [
        "Papa's Bakeria",
        "Papa's Cupcakeria",
        "Papa's Burgeria",
        "Papa's Wingeria",
        "Papa's Pancakeria"
    ],
    "Attack on Titan": [
        "Attack on Titan 2",
        "Attack on Titan: Humanity in Chains",
        "Titan Quest",
        "Titan Quest: Immortal Throne",
        "Titan Souls"
    ],
    "Spectacular Sparky": [
        "Spark the Electric Jester",
        "PPL: The Animated Adventures",
        "Apex Legends",
        "Sparklite",
        "Fiendish Freddy's Big Top o' Fun"
    ],
    "Bookworm": [
        "Bookworm Adventures Volume 2",
        "Bookworm Adventures",
        "Alphabet Stew",
        "Great Word Adventure",
        "WordTrap Dungeon"
    ],
    "Marvel vs. Capcom 3: Fate of Two Worlds": [
        "Ultimate Marvel vs. Capcom 3",
        "Marvel: Ultimate Alliance",
        "Marvel vs. Capcom 2: New Age of Heroes",
        "Marvel vs. Capcom: Infinite",
        "Marvel vs. Capcom: Clash of Super Heroes"
    ],
    "Megamind: Mega Team Unite": [
        "Megamind",
        "Mega Man Battle Network 2",
        "Mega Man 9",
        "Mega Man Battle Network 4",
        "Mega Man 6"
    ],
    "Superman: The Game": [
        "Superman",
        "Superman",
        "The Death and Return of Superman",
        "Superman: The Man of Steel",
        "Superman Returns"
    ],
    "Crash Bandicoot: On the Run!": [
        "Crash Bandicoot: Mobile",
        "Crash Bandicoot: The Huge Adventure",
        "Crash Bandicoot",
        "Crash Bandicoot N. Sane Trilogy",
        "Crash Bandicoot: The Wrath of Cortex"
    ],
    "Bot Vice": [
        "EVE: Gunjack",
        "Strafe",
        "Transformers: Revenge of the Fallen - Autobots",
        "Future Noir",
        "Resident Evil: Resistance"
    ],
    "Mass Effect 2: Lair of the Shadow Broker": [
        "Hitman 2",
        "Dead to Rights: Reckoning",
        "Cyber Shadow",
        "Shadowrun: Hong Kong",
        "Sachi's Quest"
    ],
    "Goodbye Volcano High": [
        "Bonesaw",
        "Dinosaur",
        "DinoScape",
        "Shivers II: Harvest of Souls",
        "Dinocity"
    ],
    "Devil Kings": [
        "Samurai Warriors",
        "Sengoku basara",
        "Muramasa: The Demon Blade",
        "Way of the Samurai",
        "World of Demons"
    ],
    "Surgical Strike": [
        "Strike Force 2: Terrorist Hunt",
        "CT Special Forces: Fire for Effect",
        "Sniper Elite V2",
        "Terrorist Takedown: Covert Operations",
        "Counter-Strike: Global Offensive"
    ]
}




def preprocess_data(csv_path = "IMBD DATASET.csv"):
    """
    Reads a CSV file, processes the data by removing specific columns, duplicates, and unwanted rows,
    and converts boolean columns to integer type. The function returns the processed dataframe.
    :param csv_path: The file path to the CSV file to be processed.
    :type csv_path: str, optional
    :return: A pandas DataFrame object containing the processed data.
    :rtype: pandas.DataFrame
    """
    df = pd.read_csv(csv_path)
    df.drop(df.columns[0], axis=1, inplace=True)
    cols_to_remove = ["url", "year", "certificate", "rating", "votes"]
    df.drop(columns=cols_to_remove, errors="ignore" ,inplace=True)
    df.drop_duplicates(keep="first", inplace=True)
    df = df[~df["plot"].str.contains("Add a Plot", na=False)]
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df

def retrieve_similar(query_text, df, embeddings, model, top_k=5):
    """
    Retrieve the most similar texts to a given query text from a dataset based on embeddings.
    This function calculates cosine similarity between the embedding of the query text and the
    precomputed embeddings of a dataset. It returns the top-k most similar texts, excluding the
    query text itself if it is present in the dataset.
    :param query_text: The query string for which similar texts are to be retrieved.
    :param df: Dataframe containing the dataset with a column "text" from which similar texts are
        retrieved.
    :param embeddings: A tensor containing precomputed embeddings corresponding to the "text"
        column in the dataframe.
    :param model: A language model used for encoding the query text to compute its embedding.
    :param top_k: The number of top similar texts to retrieve. Defaults to 5.
    :return: A list of top-k retrieved texts most similar to the query text.
    """
    query_emb = model.encode(query_text, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k + 1)
    top_indices = top_results.indices.cpu().numpy()
    filtered_texts = []
    for idx in top_indices:
        candidate_text = df.iloc[idx]["text"]
        if candidate_text.strip() != query_text.strip():
            filtered_texts.append(candidate_text)
        if len(filtered_texts) == top_k:
            break

    return filtered_texts



def evaluate_sbert_model(query_text, references, df, embeddings, model):
    """
    Evaluates the performance of an SBERT model using a set of query references and associated
    metrics. The function retrieves text candidates similar to the input query, computes BERTScore
    metrics (precision, recall, and F1-score) for the retrieved candidates against the given references,
    and averages these metrics to provide an overall measure of performance for the model.

    :param query_text: The input query text used to retrieve similar candidates.
    :param references: A list of reference texts against which the retrieved candidates are evaluated.
    :param df: A DataFrame containing text data that serves as the source for candidate retrieval.
    :param embeddings: Precomputed embeddings associated with the text data in the DataFrame.
    :param model: The SBERT model used to compute candidate similarity and retrieve similar texts.
    :return: A tuple containing the averaged precision, recall, and F1-score as floats:
             (avg_prec, avg_rec, avg_f1).
    """

    candidates = retrieve_similar(query_text, df, embeddings, model)
    all_prec, all_rec, all_f1 = [], [], []
    for cand in candidates:
        P, R, F1 = bertscore_score([cand] * len(references), references, lang="en", verbose=False)
        all_prec.append(P.mean().item())
        all_rec.append(R.mean().item())
        all_f1.append(F1.mean().item())

    # Average over top_k
    avg_prec = sum(all_prec) / len(all_prec)
    avg_rec = sum(all_rec) / len(all_rec)
    avg_f1 = sum(all_f1) / len(all_f1)

    return avg_prec, avg_rec, avg_f1




def embed_texts(clip_model, tokenizer, texts, batch_size=32):
    """
    Embeds a list of text strings into a higher-dimensional vector space using a provided
    CLIP model and tokenizer. The embeddings are calculated in batches to optimize memory
    usage and processing time. The process involves tokenizing the input texts, passing
    them through the CLIP model, and normalizing the output embeddings.

    :param clip_model: The CLIP model used for generating embeddings. It expects tokenized input data.
    :type clip_model: Any
    :param tokenizer: The tokenizer used to preprocess the input text strings. It transforms text into
        a format consumable by the CLIP model.
    :type tokenizer: Any
    :param texts: A list of text strings to be embedded into vector space.
    :type texts: list[str]
    :param batch_size: The number of text samples processed in each batch. This is used to balance
        memory usage and performance. Default is 32.
    :type batch_size: int, optional
    :return: A tensor containing the embeddings for the input texts. Each row in the tensor corresponds
        to the embedding of a single text string.
    """
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i+batch_size]
            tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            output = clip_model(**tokens).last_hidden_state[:, 0, :]
            output = F.normalize(output, p=2, dim=1)
            embeddings.append(output.cpu())
    return torch.cat(embeddings, dim=0)

#
# # Embedding function
# def embed_texts_with_clip(texts, model, tokenizer, batch_size=32):
#     embeddings = []
#     with torch.no_grad():
#         for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
#             batch = texts[i:i+batch_size]
#             tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
#             outputs = model(**tokens).last_hidden_state[:, 0, :]
#             outputs = F.normalize(outputs, p=2, dim=1)
#             embeddings.append(outputs.cpu())
#     return torch.cat(embeddings, dim=0)


def index_titles_name(title, all_titles):
    """
    Indexes the position of a given title in a list of titles.
    This function searches for the index of a specified title within a given list
    of titles. It returns the zero-based index of the specified title. The function
    expects the title to exist within the list.
    :param title: A string representing the title to search for.
    :param all_titles: A list of strings where the given title will be searched.
    :return: The index of the specified title in the list of titles.
    """
    return all_titles.index(title)




def evaluate_clip_model(clip_embeddings, texts, titles):
    """
    Evaluates a CLIP model's embedding quality by comparing the similarities between
    retrieved and reference texts using BERTScore metrics. For each given query title,
    retrieved texts are ranked based on cosine similarity with the query embedding.
    Precision, recall, and F1 scores are computed for the top retrieved texts against
    ground-truth references, and average scores are calculated across all queries.

    :param clip_embeddings: Precomputed embeddings of texts as PyTorch tensors.
    :param texts: A list of strings representing all the texts corresponding to the embeddings.
    :param titles: A list of strings representing the titles of the corresponding texts.
    :return: A tuple containing three float values representing the mean F1 score, mean precision,
        and mean recall across all evaluated queries.
    :rtype: tuple[float, float, float]
    """
    all_f1_scores = []
    all_p_scores = []
    all_r_scores = []
    for query_title, gt_titles in tqdm(ground_truth.items(), desc="Evaluating"):
        query_idx = index_titles_name(query_title, titles)
        query_emb = clip_embeddings[query_idx].unsqueeze(0)
        sims = util.cos_sim(query_emb, clip_embeddings)[0]
        top_indices = torch.topk(sims, k=6).indices.tolist()
        retrieved_texts = [texts[i] for i in top_indices if titles[i] != query_title][:5]
        gt_indices = [index_titles_name(gt, titles) for gt in gt_titles if index_titles_name(gt, titles) is not None]
        reference_texts = [texts[i] for i in gt_indices]
        all_prec, all_rec, all_f1 = [], [], []
        for cand in retrieved_texts:
            P, R, F1 = bertscore_score([cand] * len(reference_texts), reference_texts, lang="en", verbose=False)
            all_prec.append(P.mean().item())
            all_rec.append(R.mean().item())
            all_f1.append(F1.mean().item())

        avg_prec = sum(all_prec) / len(all_prec)
        avg_rec = sum(all_rec) / len(all_rec)
        avg_f1 = sum(all_f1) / len(all_f1)
        all_f1_scores.append(avg_f1)
        all_p_scores.append(avg_prec)
        all_r_scores.append(avg_rec)

    return np.mean(all_f1_scores), np.mean(all_p_scores), np.mean(all_r_scores)


# def evaluate_fine_tuned_clip_model(clip_embeddings, texts, titles):
#     """
#     Evaluates the performance of a fine-tuned CLIP model based on BERTScore metrics.
#     This function computes the average F1-score, precision, and recall of the
#     retrieved texts for a given set of queries, using a fine-tuned CLIP model's
#     embeddings. BERTScore is used to compare the semantic similarity between
#     retrieved titles and the ground truth reference titles.
#     :param clip_embeddings: A tensor containing the embeddings generated by the
#         fine-tuned CLIP model for the given dataset.
#     :type clip_embeddings: torch.Tensor
#     :param texts: A list of text titles corresponding to the data points,
#         which are used for evaluation of the retrieval task.
#     :type texts: list of str
#     :param titles: A list of text titles representing the names associated with
#         the embeddings in the same order as ``clip_embeddings``.
#     :type titles: list of str
#     :return: A tuple containing three lists - the average F1-scores, precision
#         scores, and recall scores for all evaluated queries.
#     :rtype: tuple of (list of float, list of float, list of float)
#     """
#
#     all_f1_scores = []
#     all_p_scores = []
#     all_r_scores = []
#
#     for query_title, gt_titles in tqdm(ground_truth.items(), desc="Evaluating"):
#         query_idx = index_titles_name(query_title, titles)
#         query_emb = clip_embeddings[query_idx].unsqueeze(0)
#         sims = util.cos_sim(query_emb, clip_embeddings)[0]
#         top_indices = torch.topk(sims, k=6).indices.tolist()
#
#         # Get top-5 retrieved texts
#         retrieved_titles = [texts[i] for i in top_indices if texts[i] != query_title][:5]
#         # Get ground truth reference texts
#         gt_indices = [index_titles_name(gt, titles) for gt in gt_titles if index_titles_name(gt, titles) is not None]
#
#         reference_titles = [texts[i] for i in gt_indices]
#         # Compute BERTScore
#         all_prec, all_rec, all_f1 = [], [], []
#         for cand in retrieved_titles:
#             P, R, F1 = bertscore_score([cand] * len(reference_titles), reference_titles, lang="en", verbose=False)
#             all_prec.append(P.mean().item())
#             all_rec.append(R.mean().item())
#             all_f1.append(F1.mean().item())
#
#         avg_prec = sum(all_prec) / len(all_prec)
#         avg_rec = sum(all_rec) / len(all_rec)
#         avg_f1 = sum(all_f1) / len(all_f1)
#         all_f1_scores.append(avg_f1)
#         all_p_scores.append(avg_prec)
#         all_r_scores.append(avg_rec)
#     return np.mean(all_f1_scores), np.mean(all_p_scores), np.mean(all_r_scores)



if __name__ == '__main__':

    csv_path = "IMBD DATASET.csv"
    data = preprocess_data(csv_path)
    data['text'] = 'Title: ' + data['name'].fillna('') + '. Plot: ' + data['plot'].fillna('')
    texts = data['text'].tolist()
    titles = data["name"].tolist()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_tensor=True)

    df_ground_truth = pd.read_csv('ground truth.csv')
    df_ground_truth['text'] = 'Title: ' + df_ground_truth['name'].fillna('') + '. Plot: ' + df_ground_truth['plot'].fillna('')

    # Evaluate the SBERT model using the ground truth data

    scores = []
    for i in range(GROUND_TRUTH_SiZE):
        query_name = df_ground_truth.iloc[i]['name']
        query_text = df_ground_truth.iloc[i]['text']

        ref_titles = ground_truth[query_name]
        references = []
        for title in ref_titles:
            match = data[data["name"].str.lower() == title.lower()]
            if not match.empty:
                references.append(match.iloc[0]["text"])

        prec, rec, f1 = evaluate_sbert_model(query_text, references, data, embeddings, model)
        scores.append({
            "query": query_name,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

    # Report average score
    df_scores = pd.DataFrame(scores)
    print("Average BERTScore F1 for the SBERT model:", df_scores['f1'].mean())
    print("Average BERTScore Precision for the SBERT model:", df_scores['precision'].mean())
    print("Average BERTScore Recall for the SBERT model:", df_scores['recall'].mean())




    # Load the fine-tuned model and tokenizer
    model_path = "clip-finetuned-sbert"
    clip_model = CLIPTextModel.from_pretrained(model_path)
    tokenizer = CLIPTokenizer.from_pretrained(model_path)

    clip_model.eval()

    clip_embeddings = embed_texts(clip_model, tokenizer, texts)

    # Evaluate the fine-tuned CLIP model
    f1, P, R = evaluate_clip_model(clip_embeddings, texts, titles)

    # Final average
    print(f"\nAverage BERTScore F1 for the fine tuned CLIP-text model: {f1:.3f}")
    print(f"Average BERTScore Precision for the fine tuned CLIP-text model: {P:.3f}")
    print(f"Average BERTScore Recall for the fine tuned CLIP-text model: {R:.3f}")



    # Load original CLIP
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    # Embed all texts
    clip_embeddings = embed_texts(clip_model, tokenizer, texts)
    # Run evaluation
    f1, P, R = evaluate_clip_model(clip_embeddings, texts, titles)
    print(f"\nAverage BERTScore F1 for the original CLIP-text model: {f1:.4f}")
    print(f"Average BERTScore Precision for the original CLIP-text model: {P:.4f}")
    print(f"Average BERTScore Recall for the original CLIP-text model: {R:.4f}")


