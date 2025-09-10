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


def prepare_investor_text(investor: Dict[str, Any]) -> str:
    """Build text to embed from investor data."""
    tags = investor.get('industry_tags', '')
    if isinstance(tags, list):
        tags = " ".join(tags)
    thesis = investor.get('thesis_text', '')
    return f"{thesis} {tags}".strip()


def add_investor_to_chroma(investor: Dict[str, Any]):
    """Add one investor to Chroma DB and persist."""
    text_to_embed = prepare_investor_text(investor)

    # normalize/serialize fields that may be lists
    stage_focus = investor.get('stage_focus')
    if isinstance(stage_focus, list):
        stage_focus_serialized = json.dumps(stage_focus)
    else:
        stage_focus_serialized = stage_focus

    industry_tags = investor.get('industry_tags')
    if isinstance(industry_tags, list):
        industry_tags_serialized = json.dumps(industry_tags)
    else:
        industry_tags_serialized = industry_tags

    metadata = {
        "investor_id": str(investor.get("investor_id", "")),
        "name": investor.get("name", ""),
        "stage_focus": stage_focus_serialized,
        "ticket_min_egp": int(investor.get("ticket_min_egp", 0)),
        "ticket_max_egp": int(investor.get("ticket_max_egp", 0)),
        "industry_tags": industry_tags_serialized,
        "thesis_text": investor.get("thesis_text", ""),
        "contact_email": investor.get("contact") or investor.get("email")
    }

    # Serialize any dict values in metadata to JSON strings
    for k, v in metadata.items():
        if isinstance(v, dict):
            metadata[k] = json.dumps(v)

    embedding = embed_text(text_to_embed)
    collection.add(
        documents=[text_to_embed],
        embeddings=[embedding],  
        metadatas=[metadata],
        ids=[str(investor.get("investor_id", ""))]
    )


