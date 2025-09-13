import chromadb
import os
from huggingface_hub import InferenceClient
import chromadb
import json
from typing import Dict, Any, Tuple, List


hf_client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_API_TOKEN"],
)

# persistent chroma DB
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("investors")


def embed_text(text: str):
    embedding = hf_client.feature_extraction(text, model="BAAI/bge-small-en-v1.5")
    return embedding


def recommend_investors(startup: Dict[str, Any], k: int = 3) -> List[Tuple[Any, float]]:
    """Query Chroma and filter results.

    Builds a robust query string and filters results by ticket size and stage.
    """
    # build query text defensively
    parts = [
        startup.get("problem_statement", ""),
        startup.get("solution_description", ""),
    ]
    industry_tags = startup.get("industry_tags", [])
    if isinstance(industry_tags, list):
        parts.append(" ".join(industry_tags))
    else:
        parts.append(str(industry_tags))

    parts.append(f"Stage: {startup.get('stage', '')}")
    parts.append(f"Funding ask: {startup.get('funding_ask_egp', '')}")
    query_text = " ".join([p for p in parts if p])
    query_vector = embed_text(query_text)
    results = collection.query(query_embeddings=[query_vector], n_results=k)

    filtered: List[Tuple[Any, float]] = []
    try:
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
    except Exception as e:
        print(f"Error extracting results from Chroma: {e}")
        return []

    if not ids or not metadatas or not distances:
        print("No results returned from Chroma query.")
        return []

    for id_, meta, dist in zip(ids, metadatas, distances):
        reasons: List[str] = []
        try:
            ticket_min = int(meta.get("ticket_min_egp", 0))
            ticket_max = int(meta.get("ticket_max_egp", 0))
        except Exception as e:
            print(f"Error parsing ticket size for investor {id_}: {e}")
            continue

        raw_stage = meta.get("stage_focus")
        if isinstance(raw_stage, list):
            stages = raw_stage
        elif isinstance(raw_stage, str):
            raw = raw_stage.strip()
            if raw.startswith("[") and raw.endswith("]"):
                try:
                    stages = json.loads(raw)
                except Exception as e:
                    print(f"Error parsing stage_focus for investor {id_}: {e}")
                    stages = [raw]
            else:
                stages = [raw]
        else:
            stages = [str(raw_stage)]

        if industry_tags:
            inv_tags = meta.get("industry_tags", [])
            if isinstance(inv_tags, str):
                inv_tags = [inv_tags]
            matched_tags = list(set(industry_tags) & set(inv_tags))
            if matched_tags:
                reasons.append(f"Industry tags match: {', '.join(matched_tags)}")

        funding_ask = startup.get("funding_ask_egp")
        stage = startup.get("stage")

        if funding_ask is None or stage is None:
            reasons.append("Stage or funding not provided, matched by similarity only.")
            filtered.append({
                "investor_id": id_,
                "score": 1 - dist,
                "reasons": reasons,
            })
            continue

        try:
            if (
                ticket_min <= funding_ask <= ticket_max
                and stage in stages
            ):
                reasons.append("Funding matches investor range.")
                reasons.append("Stage matches investor focus.")
                filtered.append({
                    "investor_id": id_,
                    "score": 1 - dist,
                    "reasons": reasons,
                })
        except Exception as e:
            print(f"Error filtering investor {id_}: {e}")
            continue

    if not filtered:
        print("No investors matched the criteria.")
    return filtered