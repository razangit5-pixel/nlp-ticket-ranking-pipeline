"""
Ticket Ranking System (inspired by Uber's approach)
Pipeline: Data → Pre-processing → Feature Engineering (LSI + TF-IDF) → Cosine Similarity → Pointwise Ranking
"""

import json
import math
import re
from collections import defaultdict, Counter

# ============================================================
# SAMPLE DATA (Ticket Info + Text + Trip Data)
# ============================================================
SAMPLE_TICKETS = [
    {
        "id": "T001",
        "title": "App crashes when booking a ride",
        "text": "The mobile application crashes immediately after I tap the book ride button. The driver never receives my request. This happens every time on Android 13.",
        "category": "app_crash",
        "priority": "high",
        "trip_data": {"trips": 45, "cancellations": 2, "rating": 4.8}
    },
    {
        "id": "T002",
        "title": "Driver did not arrive at pickup location",
        "text": "My driver accepted the ride but went to the wrong pickup location. I waited 20 minutes and had to cancel. The GPS navigation seems to have a bug.",
        "category": "driver_issue",
        "priority": "medium",
        "trip_data": {"trips": 120, "cancellations": 5, "rating": 4.5}
    },
    {
        "id": "T003",
        "title": "Payment charged twice for single trip",
        "text": "I was charged twice for the same ride. My credit card shows two identical charges. Please refund the duplicate payment immediately.",
        "category": "payment",
        "priority": "high",
        "trip_data": {"trips": 80, "cancellations": 1, "rating": 4.9}
    },
    {
        "id": "T004",
        "title": "App crashes on iPhone",
        "text": "The app crashes when I try to book a ride on my iPhone. It freezes and closes. Cannot complete any booking through the mobile app.",
        "category": "app_crash",
        "priority": "high",
        "trip_data": {"trips": 30, "cancellations": 0, "rating": 5.0}
    },
    {
        "id": "T005",
        "title": "Incorrect fare calculation",
        "text": "The fare was much higher than the estimate shown before the trip. The surge pricing was not disclosed properly. I need a fare adjustment.",
        "category": "payment",
        "priority": "medium",
        "trip_data": {"trips": 200, "cancellations": 10, "rating": 4.2}
    },
    {
        "id": "T006",
        "title": "Cannot login to account",
        "text": "I cannot log in to my account. The password reset email never arrives. I have been locked out for 3 days and cannot book any rides.",
        "category": "account",
        "priority": "medium",
        "trip_data": {"trips": 15, "cancellations": 0, "rating": 4.7}
    },
    {
        "id": "T007",
        "title": "Driver was rude and unprofessional",
        "text": "The driver was extremely rude throughout the trip. He was on his phone the entire time and made several unsafe driving maneuvers.",
        "category": "driver_issue",
        "priority": "medium",
        "trip_data": {"trips": 55, "cancellations": 3, "rating": 3.8}
    },
    {
        "id": "T008",
        "title": "App not loading map",
        "text": "The map inside the application is not loading properly. It shows a blank screen. Cannot see driver location or book a ride.",
        "category": "app_crash",
        "priority": "low",
        "trip_data": {"trips": 10, "cancellations": 1, "rating": 4.6}
    }
]


QUERY_TICKET = {
    "id": "QUERY",
    "title": "Wrong charge on my credit card",
    "text": "I was charged the wrong amount for my trip. The fare shown was different from what was deducted from my card.",
    "trip_data": {"trips": 60, "cancellations": 2, "rating": 4.7}
}

# ============================================================
# STEP 1: PRE-PROCESSING
# ============================================================
STOP_WORDS = {
    'a','an','the','is','it','in','on','at','to','for','of','and','or','but',
    'i','my','me','was','were','be','been','have','has','had','this','that',
    'he','she','they','we','you','do','did','does','not','no','so','if','as',
    'with','from','by','are','its','also','can','all','after','when','then',
    'very','much','been','more','get','got','never','every','just','any','too',
    'same','than','there','his','her','their','our','will','would','could'
}

LEMMA_MAP = {
    'crashes': 'crash', 'crashing': 'crash', 'crashed': 'crash',
    'bookings': 'booking', 'booked': 'book', 'books': 'book', 'booking': 'book',
    'rides': 'ride', 'riding': 'ride',
    'payments': 'payment', 'paying': 'pay', 'paid': 'pay',
    'charges': 'charge', 'charged': 'charge', 'charging': 'charge',
    'drivers': 'driver', 'driving': 'drive', 'driven': 'drive',
    'apps': 'app', 'applications': 'application',
    'loading': 'load', 'loaded': 'load', 'loads': 'load',
    'freezes': 'freeze', 'frozen': 'freeze', 'froze': 'freeze',
    'trips': 'trip',
    'accounts': 'account',
    'logins': 'login', 'logging': 'login', 'logged': 'login',
    'errors': 'error', 'failed': 'fail', 'fails': 'fail', 'failure': 'fail',
    'issues': 'issue',
    'problems': 'problem',
    'fixes': 'fix', 'fixed': 'fix', 'fixing': 'fix',
    'shows': 'show', 'showing': 'show', 'shown': 'show',
    'happening': 'happen', 'happens': 'happen', 'happened': 'happen',
    'receiving': 'receive', 'received': 'receive', 'receives': 'receive',
    'requests': 'request', 'requesting': 'request', 'requested': 'request',
    'refunds': 'refund', 'refunding': 'refund', 'refunded': 'refund',
    'disclosing': 'disclose', 'disclosed': 'disclose',
}

def preprocess(text):
    """Full preprocessing pipeline: tokenize → lowercase → remove stopwords → lemmatize"""
    # Tokenization (split on non-alphanumeric)
    tokens_raw = re.findall(r'[a-zA-Z]+', text)
    # Lowercasing
    tokens_lower = [t.lower() for t in tokens_raw]
    # Stop word removal
    tokens_no_stop = [t for t in tokens_lower if t not in STOP_WORDS and len(t) > 2]
    # Lemmatization (rule-based map)
    tokens_lemma = [LEMMA_MAP.get(t, t) for t in tokens_no_stop]
    return tokens_lemma

# ============================================================
# STEP 2: TF-IDF
# ============================================================
def compute_tfidf(corpus_tokens):
    """Compute TF-IDF for a corpus of tokenized documents"""
    N = len(corpus_tokens)
    # Document frequency
    df = defaultdict(int)
    for tokens in corpus_tokens:
        for term in set(tokens):
            df[term] += 1
    # IDF
    idf = {term: math.log((N + 1) / (freq + 1)) + 1 for term, freq in df.items()}
    # TF-IDF vectors
    tfidf_vectors = []
    for tokens in corpus_tokens:
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec = {term: (count / total) * idf.get(term, 0) for term, count in tf.items()}
        tfidf_vectors.append(vec)
    return tfidf_vectors, idf

# ============================================================
# STEP 3: LSI (Latent Semantic Indexing) — simplified SVD-free version , replaced by Embeddings
# ============================================================
def compute_lsi(corpus_tokens, n_components=5):
    """
    Simplified LSI: project token lists into latent topics using
    co-occurrence-based topic scoring
    """
    # Build vocabulary from token lists
    vocab = set()
    for tokens in corpus_tokens:
        vocab.update(tokens)
    vocab = sorted(vocab)
    term_idx = {t: i for i, t in enumerate(vocab)}

    # Build term-frequency matrix (docs × terms) from raw tokens
    matrix = []
    for tokens in corpus_tokens:
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        row = [tf.get(t, 0) / total for t in vocab]
        matrix.append(row)

    # Simplified topic extraction: group high co-occurrence terms
    # Topic seeds (manually seeded based on domain knowledge)
    topic_seeds = [
        ['crash', 'app', 'book', 'freeze', 'load', 'map', 'mobile', 'application'],
        ['payment', 'charge', 'fare', 'refund', 'credit', 'duplicate', 'surge', 'pay'],
        ['driver', 'pickup', 'location', 'arrive', 'route', 'gps', 'navigation'],
        ['account', 'login', 'password', 'email', 'reset', 'lock'],
        ['trip', 'cancel', 'ride', 'request', 'book'],
    ]

    lsi_vectors = []
    for row in matrix:
        lsi_vec = []
        for seeds in topic_seeds:
            score = sum(row[term_idx[s]] if s in term_idx else 0 for s in seeds)
            lsi_vec.append(round(score, 4))
        lsi_vectors.append(lsi_vec)

    return lsi_vectors, topic_seeds

# ============================================================
# STEP 4: COSINE SIMILARITY
# ============================================================
def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors (dict or list)"""
    if isinstance(v1, dict):
        keys = set(v1) | set(v2)
        dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
        mag1 = math.sqrt(sum(x**2 for x in v1.values()))
        mag2 = math.sqrt(sum(x**2 for x in v2.values()))
    else:
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(x**2 for x in v1))
        mag2 = math.sqrt(sum(x**2 for x in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return round(dot / (mag1 * mag2), 4)

# ============================================================
# STEP 5: POINTWISE RANKING
# ============================================================
def pointwise_rank(tickets, query_id, tfidf_sims, lsi_sims):
    """
    Combine TF-IDF similarity, LSI similarity, and trip metadata into final score
    Weights: TF-IDF 40%, LSI 40%, user activity 20%
    """
    results = []
    for i, ticket in enumerate(tickets):
        tfidf_score = tfidf_sims[i]
        lsi_score = lsi_sims[i]
        # Trip metadata score (normalized)
        trips = ticket["trip_data"]["trips"]
        rating = ticket["trip_data"]["rating"]
        trip_score = min(trips / 200.0, 1.0) * 0.5 + (rating / 5.0) * 0.5

        # Weighted combination
        final_score = 0.40 * tfidf_score + 0.40 * lsi_score + 0.20 * trip_score

        results.append({
            "id": ticket["id"],
            "title": ticket["title"],
            "category": ticket["category"],
            "tfidf_score": round(tfidf_score, 4),
            "lsi_score": round(lsi_score, 4),
            "trip_score": round(trip_score, 4),
            "final_score": round(final_score, 4),
            "tokens": preprocess(ticket["title"] + " " + ticket["text"])
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results

# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(query_ticket, corpus_tickets):
    pipeline_log = {}

    # -- STEP 1: Preprocessing --
    corpus_tokens = [preprocess(t["title"] + " " + t["text"]) for t in corpus_tickets]
    query_tokens = preprocess(query_ticket["title"] + " " + query_ticket["text"])
#update on our database
    pipeline_log["step1_preprocessing"] = {
        "query_tokens": query_tokens,
        "corpus_sample": {
            corpus_tickets[0]["id"]: corpus_tokens[0][:10]
        }
    }

    # -- STEP 2: TF-IDF --
    all_tokens = corpus_tokens + [query_tokens]
    tfidf_vectors, idf = compute_tfidf(all_tokens)
    query_tfidf = tfidf_vectors[-1]
    corpus_tfidf = tfidf_vectors[:-1]

    pipeline_log["step2_tfidf"] = {
        "vocabulary_size": len(idf),
        "top_idf_terms": sorted(idf.items(), key=lambda x: x[1], reverse=True)[:10],
        "query_top_terms": sorted(query_tfidf.items(), key=lambda x: x[1], reverse=True)[:8]
    }

    # -- STEP 3: LSI --
    all_lsi, topics = compute_lsi(all_tokens)
    query_lsi = all_lsi[-1]
    corpus_lsi = all_lsi[:-1]

    pipeline_log["step3_lsi"] = {
        "topics": [{"id": i, "seeds": s[:4]} for i, s in enumerate(topics)],
        "query_lsi_vector": query_lsi
    }

    # -- STEP 4: Cosine Similarity --
    tfidf_sims = [cosine_similarity(query_tfidf, v) for v in corpus_tfidf]
    lsi_sims = [cosine_similarity(query_lsi, v) for v in corpus_lsi]

    pipeline_log["step4_cosine"] = {
        "tfidf_similarities": {corpus_tickets[i]["id"]: tfidf_sims[i] for i in range(len(corpus_tickets))},
        "lsi_similarities": {corpus_tickets[i]["id"]: lsi_sims[i] for i in range(len(corpus_tickets))}
    }

    # -- STEP 5: Pointwise Ranking --
    ranked = pointwise_rank(corpus_tickets, query_ticket["id"], tfidf_sims, lsi_sims)
    pipeline_log["step5_ranking"] = ranked

    # -- EVALUATION --
    evaluation = evaluate_pipeline(ranked, corpus_tickets)
    pipeline_log["evaluation"] = evaluation

    return pipeline_log

# ============================================================
# EVALUATION METRICS
# ============================================================
def evaluate_pipeline(ranked, corpus_tickets):
    """
    Evaluate ranking quality:
    - Precision@K: how many of top-K are relevant (same category as most-expected)
    - NDCG@K: normalized discounted cumulative gain
    - Mean Reciprocal Rank (MRR): rank of first relevant result
    """
    # Ground truth: tickets with same category as majority of top results
    top_category = ranked[0]["category"]
    relevant_ids = {t["id"] for t in corpus_tickets if t["category"] == top_category}

    def precision_at_k(k):
        top_k = ranked[:k]
        hits = sum(1 for r in top_k if r["id"] in relevant_ids)
        return round(hits / k, 4)

    def dcg_at_k(k):
        dcg = 0.0
        for i, r in enumerate(ranked[:k]):
            rel = 1 if r["id"] in relevant_ids else 0
            dcg += rel / math.log2(i + 2)
        return dcg

    def idcg_at_k(k):
        ideal = sorted([1 if r["id"] in relevant_ids else 0 for r in ranked], reverse=True)[:k]
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))

    def ndcg_at_k(k):
        idcg = idcg_at_k(k)
        return round(dcg_at_k(k) / idcg, 4) if idcg > 0 else 0.0

    def mrr():
        for i, r in enumerate(ranked):
            if r["id"] in relevant_ids:
                return round(1 / (i + 1), 4)
        return 0.0

    # MAP (Mean Average Precision)
    precisions = []
    hits = 0
    for i, r in enumerate(ranked):
        if r["id"] in relevant_ids:
            hits += 1
            precisions.append(hits / (i + 1))
    map_score = round(sum(precisions) / len(relevant_ids), 4) if precisions else 0.0

    return {
        "relevant_category": top_category,
        "relevant_tickets": list(relevant_ids),
        "precision_at_1": precision_at_k(1),
        "precision_at_3": precision_at_k(3),
        "precision_at_5": precision_at_k(5),
        "ndcg_at_3": ndcg_at_k(3),
        "ndcg_at_5": ndcg_at_k(5),
        "mrr": mrr(),
        "map": map_score,
        "ranked_ids": [r["id"] for r in ranked],
        "score_breakdown": [
            {"id": r["id"], "final": r["final_score"], "tfidf": r["tfidf_score"],
             "lsi": r["lsi_score"], "trip": r["trip_score"]} for r in ranked
        ]
    }

if __name__ == "__main__":
    results = run_pipeline(QUERY_TICKET, SAMPLE_TICKETS)
    print(json.dumps(results, indent=2))
