import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

MODEL_DIR = "models/final_model"

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Model loaded on: {device}")

labels = [
    "Checking or savings account",
    "Consumer Loan",
    "Credit card or prepaid card",
    "Credit reporting, credit repair services, or other personal consumer reports",
    "Debt collection",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Payday loan, title loan, or personal loan",
    "Student loan",
    "Vehicle loan or lease",
]
id2label = {i: label for i, label in enumerate(labels)}

MIN_INPUT_WORDS = 5

def validate_input(text) -> str | None:
    if not isinstance(text, str):
        return "Input must be text."
    
    text = text.strip()
    
    if len(text) == 0:
        return "Input cannot be empty."
    
    word_count = len(text.split())
    if word_count < MIN_INPUT_WORDS:
        return f"Input too short to classify reliably (minimum {MIN_INPUT_WORDS} words)."
    
    return None

CONFIDENCE_THRESHOLD = 0.5

def predict(text: str) -> dict:
    error = validate_input(text)
    if error:
        return {"prediction": "invalid_input", "error": error}

    inputs = tokenizer(text, truncation=True, max_length=512, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    confidence, predicted_id = torch.max(probabilities, dim=0)
    confidence = confidence.item()
    predicted_label = id2label[predicted_id.item()]

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "prediction": "needs_human_review",
            "top_guess": predicted_label,
            "confidence": round(confidence, 3),
        }

    return {
        "prediction": predicted_label,
        "confidence": round(confidence, 3),
    }

if __name__ == "__main__":
    test_text = "My mortgage servicer applied my payment to the wrong month and now shows me as 60 days late."
    result = predict(test_text)
    print(result)

    test_text_2 = "This is about a problem with my account."
    result_2 = predict(test_text_2)
    print(result_2)

    print(predict(""))
    print(predict("no"))
    print(predict(12345))    