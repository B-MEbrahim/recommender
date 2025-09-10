# FounderX
FounderX is an AI-powered Entrepreneurs Assistant App that helps early-stage founders turn raw ideas into investor-ready pitches. It offers pitch analysis with scoring, auto-pitch generation, in addition to investor-startup matching.

## Recommender API

This API provides investor recommendations for startups. It is built with Flask, uses ChromaDB as a vector store, and HuggingFace embeddings `(BAAI/bge-small-en)` to represent text. The backend is deployed on Heroku.

## Documentation
### Base URL
```
https://investor-recommender-527f3e10282e.herokuapp.com/
```
### Endpoints
- Health Check: `GET/`

    Returns a simple message to confirm the API is running.

    **Response**
    ```
    Welcome to the Startup Investor Recommendation API!
    ```
- Recommend Investors: `POST /recommend`

    Takes a startup description and returns the top investor matches.

    **Request Body (JSON)**
    ```
    {
      "problem_statement": str,
      "solution_description": str,
      "industry_tags": [str, ...],
      "stage": str,  # e.g., "Seed", "Series A"
      "funding_ask_egp": int,
      "k": int  # number of results
    }
    ```
    **Response (JSON)**
    ```
    {
        "startup_id": "ST001",
        "k": 3,
        "recommendations": [
            {"investor_id": "inv1", "reasons": "[]", "score": 0.87},
            {"investor_id": "inv2", "reasons": "[]", "score": 0.81},
            {"investor_id": "inv3", "reasons": "[]", "score": 0.76}
                            ]
    }
    ```

- Add Investors: `POST /add_investors`

    Adds one or more new investors to the database.

    **Request Body (JSON)**
    ```
    {
    "investors": [
        {
        "investor_id": "inv101",
        "name": "New Horizon Ventures",
        "stage_focus": ["seed", "series_a"],
        "ticket_min_egp": 500000,
        "ticket_max_egp": 2000000,
        "industry_tags": ["fintech", "payments"],
        "thesis_text": "We invest in early-stage fintech innovators in emerging markets.",
        "email": "contact@newhorizon.vc"
        }
    ]
    }
    ```
    **Response (JSON)**
    ```
    {
    "status": "success",
    "message": "2 investors added successfully"
    }
    ```
## Model Details
- Embedding Model: `BAAI/bge-small-en`
 from HuggingFace.

    - Used via HuggingFace Inference API (no local model download needed).

    - API key is stored securely in Heroku Config Vars (HF_API_TOKEN).
- Vector Database: `ChromaDB`
    - Stores precomputed embeddings of investors.

    - Supports semantic search for incoming startup queries.
- Pipeline Logic:

    1.Receive startup description from API request.

    2.Generate embedding for the description via HuggingFace API.

    3.Query ChromaDB for top `k` similar investors.

    4.Return a ranked list of investors with similarity scores.

## Testing
**Curl Example**
```
curl -X POST https://investor-recommender-527f3e10282e.herokuapp.com/recommend \
     -H "Content-Type: application/json" \
     -d '{"startup": "FinTech payments platform"}'
```
**Python Example**
```
import requests

url = "https://investor-recommender-527f3e10282e.herokuapp.com/recommend"
data = {"startup": "AI in healthcare"}

response = requests.post(url, json=data)
print(response.json())
```
**Output**
```
{
    "startup_id": "ST001",
    "k": 3,
    "recommendations": [
    {"investor_id": "inv1", "reasons": "[]", "score": 0.87},
    {"investor_id": "inv2", "reasons": "[]", "score": 0.81},
    {"investor_id": "inv3", "reasons": "[]", "score": 0.76}
                        ]
    }
```
