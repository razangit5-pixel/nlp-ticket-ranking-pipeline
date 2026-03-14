# NLP Ticket Ranking Pipeline
> Inspired by Uber's ticket ranking system 

## Pipeline Architecture

```
Data Sources → Pre-processing → Feature Engineering → Cosine Similarity → Pointwise Ranking
(Ticket Info,   (Tokenization,   (TF-IDF + LSI)       (vector angle)     (Issue / Solution)
 Ticket Text,    Lowercasing,
 Trip Data)      Stop word removal,
                 Lemmatization)
```

## Files
- `ticket_pipeline.py` — full standalone pipeline (zero dependencies, pure Python)
- `pipeline_output.json` — sample output from running the pipeline

## Steps Explained

### Step 1: Pre-processing
- **Tokenization**: split text into words using regex `[a-zA-Z]+`
- **Lowercasing**: convert all tokens to lowercase
- **Stop word removal**: remove ~60 common English words (the, is, a, ...)
- **Lemmatization**: map inflected forms to base form (crashes→crash, booked→book)

### Step 2: TF-IDF (Term Frequency–Inverse Document Frequency)
```
tfidf(t, d) = tf(t, d) × log((N+1) / (df(t)+1)) + 1
```
- Rare terms that appear often in the query get high weight
- Common terms (appear in many docs) get discounted

### Step 3: LSI (Latent Semantic Indexing)
- Projects documents into 5 latent topic dimensions
- Topics: App/Crash | Payment | Driver | Account | Trip
- Captures semantic similarity beyond exact word match
- e.g. "freeze" and "crash" both map to Topic 0

### Step 4: Cosine Similarity
```
cos(q, d) = (q·d) / (‖q‖ × ‖d‖)
```
- Computed separately for TF-IDF vectors and LSI vectors
- Range: 0.0 (no similarity) to 1.0 (identical)

### Step 5: Pointwise Ranking
```
final_score = 0.40 × tfidf_sim + 0.40 × lsi_sim + 0.20 × trip_score
```
- trip_score = 0.5 × (trips/200) + 0.5 × (rating/5)

## Evaluation Metrics
| Metric | Formula | Result |
|--------|---------|--------|
| Precision@1 | relevant in top 1 / 1 | 1.00 |
| Precision@3 | relevant in top 3 / 3 | 1.00 |
| NDCG@3 | DCG@3 / IDCG@3 | 1.00 |
| MRR | 1/rank of first relevant | 1.00 |
| MAP | mean avg precision | 1.00 |

## Usage
```bash
python ticket_pipeline.py > results.json
```

## Requirements
- Python 3.8+
- No external dependencies (pure stdlib)
