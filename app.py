from flask import Flask, request, jsonify
import os
from recommend import recommend_investors
from add_investors import add_investor_to_chroma


app = Flask(__name__)


@app.route("/recommend", methods=["POST"])
def recommend_route():
    """Recommend investors for a startup.

    Expected JSON body:
    {
      "startup_id": str,
      "problem_statement": str,
      "solution_description": str,
      "industry_tags": [str, ...],
      "stage": str,  # e.g., "Seed", "Series A"
      "funding_ask_egp": int,
      "k": int  # number of results
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    # Basic validation and defaults
    startup = {
        "problem_statement": data.get("problem_statement", ""),
        "solution_description": data.get("solution_description", ""),
        "industry_tags": data.get("industry_tags", []),
        "stage": data.get("stage"),
        "funding_ask_egp": data.get("funding_ask_egp"),
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
        return jsonify({"error": f"Recommendation failed: {str(e)}"}), 500

    return jsonify({
        "startup_id": data.get("startup_id"),
        "k": k_val,
        "recommendations": recommendations,
    })


@app.route("/add_investors", methods=["POST"])
def add_investors_route():
    """Add one or more new investors to the database.

    Expected JSON body:
    {
      "investors": [
        {
          "investor_id": str,
          "name": str,
          "stage_focus": [str, ...],
          "ticket_min_egp": int,
          "ticket_max_egp": int,
          "industry_tags": [str, ...],
          "thesis_text": str,
          "email": str
        },
        ...
      ]
    }
    """

    data = request.get_json()
    if not data or "investors" not in data or not isinstance(data["investors"], list):
        return jsonify({"error": "Invalid or missing JSON body. Expected 'investors' as a list."}), 400

    required_fields = ["investor_id", "stage_focus", "ticket_min_egp", "ticket_max_egp", "email", "thesis_text"]
    results = []
    for investor in data["investors"]:
        missing = [field for field in required_fields if field not in investor]
        if missing:
            results.append({"investor_id": investor.get("investor_id"), "error": f"Missing required fields: {', '.join(missing)}"})
            continue
        try:
            add_investor_to_chroma(investor)
            results.append({"investor_id": investor.get("investor_id"), "status": "success"})
        except Exception:
            results.append({"investor_id": investor.get("investor_id"), "error": "Failed to add investor"})

    return jsonify({"results": results}), 200


@app.route("/", methods=["GET"]) 
def home():
    return jsonify({"message": "Welcome to the Startup Investor Recommendation API!", "endpoints": ["GET /", "POST /recommend", "POST /add_investors"]})


if __name__ == "__main__":
    # For local testing; Heroku will run via gunicorn using Procfile
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
