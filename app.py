from flask import Flask, render_template, request, jsonify
from googletrans import Translator
import joblib
import logging
from twilio.twiml.messaging_response import MessagingResponse
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load trained model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

translator = Translator()

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

    else:  # REAL
        if any(word in text for word in ["imd", "supreme court", "government", "official", "ministry"]):
            return "Mentions official authorities and follows verified reporting style."
        elif confidence > 80:
            return "Strong similarity with trusted news articles."
        else:
            return "Language appears neutral and factual, but verification is advised."

# ------------------------
# Web UI route
# ------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ------------------------
# Web prediction route
# ------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        original_text = data.get("text", "").strip()

        if not original_text:
            return jsonify({"error": "No text provided"}), 400

        # Translate to English
        try:
            translated = translator.translate(original_text, dest="en")
            english_text = translated.text
            source_language = translated.src
        except Exception:
            english_text = original_text
            source_language = "unknown"

        # Vectorize and predict
        text_vec = vectorizer.transform([english_text])
        prediction_raw = model.predict(text_vec)[0]  # true / fake
        probabilities = model.predict_proba(text_vec)[0]

        classes = model.classes_.tolist()
        confidence = round(probabilities[classes.index(prediction_raw)] * 100, 2)

        prediction = "REAL" if prediction_raw == "true" else "FAKE"
        reason = generate_reason(english_text, prediction, confidence)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "translated_text": english_text,
            "reason": reason,
            "source_language": source_language
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# ------------------------
# WhatsApp Bot route
# ------------------------
@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    incoming_msg = request.form.get("Body", "").strip()

    resp = MessagingResponse()
    msg = resp.message()

    # Minimum 15 words rule
    if len(incoming_msg.split()) < 15:
        msg.body("⚠️ Please send at least 15 words for analysis.")
        return str(resp)

    # Translate to English
    try:
        translated = translator.translate(incoming_msg, dest="en")
        english_text = translated.text
    except Exception:
        english_text = incoming_msg

    # Vectorize & predict
    vector = vectorizer.transform([english_text])
    raw_pred = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]

    classes = model.classes_.tolist()
    confidence = round(proba[classes.index(raw_pred)] * 100, 2)

    prediction = "REAL" if raw_pred == "true" else "FAKE"

    # Simple WhatsApp-friendly reason
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
