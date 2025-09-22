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


def recommend_investors(startup: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
    """Query Chroma and filter results with flexible matching.

    Builds a robust query string and scores results based on multiple criteria.
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
    
    try:
        query_vector = embed_text(query_text)
        # Get more results to have options after filtering
        results = collection.query(query_embeddings=[query_vector], n_results=k*3)
    except Exception as e:
        return [{"error": f"Error querying embeddings: {str(e)}"}]

    candidates: List[Dict[str, Any]] = []
    
    try:
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
    except Exception as e:
        return [{"error": f"Error extracting results from Chroma: {str(e)}"}]

    if not ids or not metadatas or not distances:
        return [{"error": "No results returned from Chroma query"}]

    startup_funding_ask = startup.get("funding_ask_egp")
    startup_stage = startup.get("stage")

    for id_, meta, dist in zip(ids, metadatas, distances):
        reasons: List[str] = []
        score_multiplier = 1.0
        
        # Base similarity score
        similarity_score = 1 - dist
        
        # Skip investors with very low similarity
        if similarity_score < 0.1:
            continue
            
        try:
            # Parse investor data with error handling for individual investors
            try:
                ticket_min = int(meta.get("ticket_min_egp", 0))
                ticket_max = int(meta.get("ticket_max_egp", float('inf')))
            except (ValueError, TypeError):
                ticket_min, ticket_max = 0, float('inf')
                reasons.append("Investor ticket size not available")

            # Parse stage focus
            try:
                raw_stage = meta.get("stage_focus")
                if isinstance(raw_stage, list):
                    stages = raw_stage
                elif isinstance(raw_stage, str):
                    raw = raw_stage.strip()
                    if raw.startswith("[") and raw.endswith("]"):
                        stages = json.loads(raw)
                    else:
                        stages = [raw]
                else:
                    stages = [str(raw_stage)] if raw_stage else []
            except (json.JSONDecodeError, TypeError):
                stages = []
                reasons.append("Investor stage focus not available")

            # Industry tag matching
            if industry_tags:
                inv_tags = meta.get("industry_tags", [])
                if isinstance(inv_tags, str):
                    inv_tags = [inv_tags]
                elif not isinstance(inv_tags, list):
                    inv_tags = []
                    
                matched_tags = list(set(industry_tags) & set(inv_tags))
                if matched_tags:
                    reasons.append(f"Industry match: {', '.join(matched_tags)}")
                    score_multiplier *= 1.2  # Boost score for industry match

            # Funding amount matching
            funding_match = False
            if startup_funding_ask is not None:
                try:
                    if ticket_min <= startup_funding_ask <= ticket_max:
                        reasons.append(f"Funding fits range (${ticket_min:,} - ${ticket_max:,})")
                        funding_match = True
                        score_multiplier *= 1.3
                    else:
                        reasons.append(f"Funding outside range (${ticket_min:,} - ${ticket_max:,})")
                        # Penalize but don't exclude
                        score_multiplier *= 0.7
                except (ValueError, TypeError):
                    reasons.append("Could not validate funding range")
            else:
                reasons.append("Startup funding ask not provided")

            # Stage matching
            stage_match = False
            if startup_stage and stages:
                if startup_stage.lower() in [s.lower() for s in stages if s]:
                    reasons.append(f"Stage match: {startup_stage}")
                    stage_match = True
                    score_multiplier *= 1.25
                else:
                    reasons.append(f"Stage mismatch (investor focuses on: {', '.join(stages)})")
                    score_multiplier *= 0.8
            else:
                reasons.append("Stage information incomplete")

            # Calculate final score
            final_score = similarity_score * score_multiplier

            candidates.append({
                "investor_id": id_,
                "score": final_score,
                "similarity_score": similarity_score,
                "reasons": reasons,
                "funding_match": funding_match,
                "stage_match": stage_match,
                "metadata": meta  # Include for debugging
            })

        except Exception as e:
            # Log error but continue processing other investors
            print(f"Warning: Error processing investor {id_}: {str(e)}")
            continue

    if not candidates:
        return [{"error": "No investors found in database"}]

    # Sort by final score and return top k
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top k results, but ensure we have some results even if none are perfect matches
    top_candidates = candidates[:k]
    
    # Clean up metadata for final response (optional)
    for candidate in top_candidates:
        if "metadata" in candidate:
            del candidate["metadata"]
    
    return top_candidates


def debug_recommendation(startup: Dict[str, Any], k: int = 3):
    """Debug version that shows more details about why matching failed"""
    print(f"Startup details:")
    print(f"  Stage: {startup.get('stage')}")
    print(f"  Funding ask: {startup.get('funding_ask_egp')}")
    print(f"  Industry tags: {startup.get('industry_tags')}")
    
    results = recommend_investors(startup, k)
    
    if results and "error" not in results[0]:
        print(f"\nFound {len(results)} recommendations:")
        for i, rec in enumerate(results, 1):
            print(f"{i}. Investor {rec['investor_id']}")
            print(f"   Score: {rec['score']:.3f} (similarity: {rec['similarity_score']:.3f})")
            print(f"   Funding match: {rec.get('funding_match', False)}")
            print(f"   Stage match: {rec.get('stage_match', False)}")
            print(f"   Reasons: {'; '.join(rec['reasons'])}")
    else:
        print(f"\nError: {results}")
    
    return results