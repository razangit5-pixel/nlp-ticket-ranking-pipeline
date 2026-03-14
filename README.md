# nlp-ticket-ranking-pipeline

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%2B%20LSI-1D9E75?style=flat)
![License](https://img.shields.io/badge/License-MIT-534AB7?style=flat)
![Dependencies](https://img.shields.io/badge/Dependencies-Zero-success?style=flat)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)
![Inspired by](https://img.shields.io/badge/Inspired%20by-Uber%20Ticketing%20System-000000?style=flat&logo=uber&logoColor=white)

> An NLP pipeline for ranking support tickets by semantic similarity. Built with pure Python, zero external dependencies.

---
## Pipeline Diagram

![Pipeline](https://github.com/razangit5-pixel/nlp-ticket-ranking-pipeline/blob/master/Screenshot%202026-03-14%20172141.png?raw=true)

## Pipeline Architecture

```
Data Sources  →  Pre-processing  →  Feature Engineering  →  Cosine Similarity  →  Ranking
(Ticket Info,    (Tokenization,      (TF-IDF + LSI)          (vector angle)        (Issue /
 Ticket Text,     Lowercasing,                                                       Solution)
 Trip Data)       Stop word removal,
                  Lemmatization)
```

---

## Steps

### Step 1 — Pre-processing
![Tokenization](https://img.shields.io/badge/-Tokenization-blue?style=flat-square)
![Lowercase](https://img.shields.io/badge/-Lowercasing-blue?style=flat-square)
![StopWords](https://img.shields.io/badge/-Stop%20Word%20Removal-blue?style=flat-square)
![Lemma](https://img.shields.io/badge/-Lemmatization-blue?style=flat-square)

Cleans raw ticket text into meaningful tokens.

```python
"App keeps crashing when booking"
→ ['app', 'keeps', 'crash', 'book', 'ride']
```

### Step 2 — TF-IDF
![TF-IDF](https://img.shields.io/badge/-TF--IDF-1D9E75?style=flat-square)

Weighs terms by how unique they are across all tickets. Rare but frequent terms score highest.

```
tfidf(t, d) = tf(t,d) × log((N+1) / (df(t)+1)) + 1
```

### Step 3 — LSI (Latent Semantic Indexing)
![LSI](https://img.shields.io/badge/-LSI-1D9E75?style=flat-square)

Maps tickets into 5 latent topic dimensions to capture semantic meaning beyond exact word matches.

| Topic | Keywords |
|-------|----------|
| 0 | crash, app, book, freeze |
| 1 | payment, charge, fare, refund |
| 2 | driver, pickup, location, arrive |
| 3 | account, login, password, email |
| 4 | trip, cancel, ride, request |

### Step 4 — Cosine Similarity
![Cosine](https://img.shields.io/badge/-Cosine%20Similarity-534AB7?style=flat-square)

Measures the angle between the query vector and each ticket vector.

```
cos(q, d) = (q · d) / (‖q‖ × ‖d‖)
```

### Step 5 — Pointwise Ranking
![Ranking](https://img.shields.io/badge/-Pointwise%20Ranking-534AB7?style=flat-square)

Combines TF-IDF, LSI, and trip metadata into a final score.

```
final_score = 0.40 × tfidf_sim + 0.40 × lsi_sim + 0.20 × trip_score
```

---

## Evaluation Results

![P@1](https://img.shields.io/badge/Precision%401-1.00-success?style=flat)
![P@3](https://img.shields.io/badge/Precision%403-1.00-success?style=flat)
![NDCG](https://img.shields.io/badge/NDCG%403-1.00-success?style=flat)
![MRR](https://img.shields.io/badge/MRR-1.00-success?style=flat)
![MAP](https://img.shields.io/badge/MAP-1.00-success?style=flat)

---

## Usage

```bash
# Clone the repo
git clone https://github.com/razangit5-pixel/nlp-ticket-ranking-pipeline

# Run the pipeline
python ticket_pipeline.py > pipeline_output.json
```

To change the query ticket, edit `QUERY_TICKET` in `ticket_pipeline.py`:

```python
QUERY_TICKET = {
    "id": "QUERY",
    "title": "Your ticket title here",
    "text": "Describe the issue in detail here.",
    "trip_data": {"trips": 60, "cancellations": 2, "rating": 4.7}
}
```

---

## Files

| File | Description |
|------|-------------|
| `ticket_pipeline.py` | Full pipeline — pure Python, no dependencies |
| `pipeline_output.json` | Output from the last pipeline run |
| `README.md` | This file |

---

## Requirements

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Stdlib](https://img.shields.io/badge/stdlib-re%20%7C%20math%20%7C%20json%20%7C%20collections-lightgrey?style=flat)

No external libraries needed. Pure Python standard library only.

