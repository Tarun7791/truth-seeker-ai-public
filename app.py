from flask import Flask, render_template, request, jsonify
import joblib
import logging
from twilio.twiml.messaging_response import MessagingResponse
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------
# Load trained model & vectorizer
# ------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ------------------------
# Reason generation logic
# ------------------------
def generate_reason(text, prediction, confidence):
    text = text.lower()

    if prediction == "FAKE":
        if any(word in text for word in ["alien", "miracle", "secret", "shocking", "exposed"]):
            return "Uses sensational or exaggerated language commonly found in fake news."
        elif confidence > 85:
            return "Highly similar to previously identified fake news patterns."
        else:
            return "No trusted or official sources mentioned."
    else:
        if any(word in text for word in ["imd", "supreme court", "government", "official", "ministry"]):
            return "Mentions official authorities and follows verified reporting style."
        elif confidence > 80:
            return "Strong similarity with trusted news articles."
        else:
            return "Language appears neutral and factual, but verification is advised."

# ------------------------
# Web UI
# ------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ------------------------
# Web prediction API
# ------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        original_text = data.get("text", "").strip()

        if not original_text:
            return jsonify({"error": "No text provided"}), 400

        # Minimum 15 words rule
        if len(original_text.split()) < 15:
            return jsonify({
                "error": "Please enter at least 15 words for analysis."
            }), 400

        english_text = original_text  # Hinglish supported directly

        # Vectorize & predict
        text_vec = vectorizer.transform([english_text])
        raw_pred = model.predict(text_vec)[0]
        proba = model.predict_proba(text_vec)[0]

        classes = model.classes_.tolist()
        confidence = round(proba[classes.index(raw_pred)] * 100, 2)

        prediction = "REAL" if raw_pred == "true" else "FAKE"
        reason = generate_reason(english_text, prediction, confidence)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "translated_text": english_text,
            "reason": reason,
            "source_language": "auto"
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ------------------------
# WhatsApp Bot (Twilio)
# ------------------------
@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    incoming_msg = request.form.get("Body", "").strip()

    resp = MessagingResponse()
    msg = resp.message()

    # Minimum 15 words rule
    if len(incoming_msg.split()) < 15:
        msg.body("⚠️ Please send at least 15 words to analyze fake news.")
        return str(resp)

    english_text = incoming_msg

    vector = vectorizer.transform([english_text])
    raw_pred = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]

    classes = model.classes_.tolist()
    confidence = round(proba[classes.index(raw_pred)] * 100, 2)

    prediction = "REAL" if raw_pred == "true" else "FAKE"

    if prediction == "FAKE":
        reason = "Message shows sensational or unverifiable patterns."
    else:
        reason = "Message matches common verified news patterns."

    msg.body(
        f"📰 Fake News Analysis\n\n"
        f"Result: {prediction}\n"
        f"Confidence: {confidence}%\n\n"
        f"Reason:\n{reason}"
    )

    return str(resp)

# ------------------------
# Run app
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
