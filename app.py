from flask import Flask, request, jsonify
import os
from recommend import recommend_investors


app = Flask(__name__)


@app.route("/recommend", methods=["POST"])
def recommend_route():
    """Recommend investors for a startup.

    Expected JSON body:
    {
      "problem_statement": str,
      "solution_description": str,
      "industry_tags": [str, ...],
      "stage": str,  # e.g., "Seed", "Series A"
      "funding_ask_usd": int,
      "k": int  # optional, number of results
    }
    """
    data = request.get_json(silent=True) or {}

    # Basic validation and defaults
    startup = {
        "problem_statement": data.get("problem_statement", ""),
        "solution_description": data.get("solution_description", ""),
        "industry_tags": data.get("industry_tags", []),
        "stage": data.get("stage"),
        "funding_ask_usd": data.get("funding_ask_usd"),
    }

    # Optional k
    k = data.get("k")
    if isinstance(k, int) and k > 0:
        k_val = k
    else:
        k_val = 3

    # Validate types where reasonable
    if startup["industry_tags"] is None:
        startup["industry_tags"] = []
    if not isinstance(startup["industry_tags"], list):
        startup["industry_tags"] = [str(startup["industry_tags"])]

    # Call recommender
    try:
        recommendations = recommend_investors(startup, k=k_val)
    except Exception as e:
        return jsonify({"error": "Recommendation failed", "detail": str(e)}), 500

    return jsonify({
        "startup_id": data.get("startup_id"),
        "k": k_val,
        "recommendations": recommendations,
    })


@app.route("/", methods=["GET"]) 
def home():
    return jsonify({"message": "Welcome to the Startup Investor Recommendation API!", "endpoints": ["GET /", "POST /recommend"]})


if __name__ == "__main__":
    # For local testing; Heroku will run via gunicorn using Procfile
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
