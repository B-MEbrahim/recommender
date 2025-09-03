from flask import Flask, request, jsonify
from recommend import recommend_investors


app = Flask(__name__)

@app.route("/recommend", method=['POST'])
def recommend():
    data = request.json
    startup_desc = data.get("startup", "")
    if not startup_desc:
        return jsonify({"error": "No startup description provided"}), 400
    
    recommendations = recommend_investors(startup_desc)
    return jsonify({
        "startup": startup_desc,
        "recommendations": recommendations
    })

@app.route("/", methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Startup Investor Recommendation API!"})
