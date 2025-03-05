import os
import cv2
import pytesseract
import shutil
import time
import re
import requests
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Configure Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Increase file upload limit (50MB)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  

# Auto-detect Tesseract path
tesseract_path = shutil.which("tesseract")

if not tesseract_path:
    tesseract_path = os.getenv("TESSERACT_PATH", "/usr/bin/tesseract")

if not tesseract_path or not os.path.exists(tesseract_path):
    raise FileNotFoundError("Tesseract-OCR not found on the system!")

pytesseract.pytesseract.tesseract_cmd = tesseract_path
print(f"Tesseract found at: {tesseract_path}")

# Google Gemini AI API configuration
GEMINI_API_KEY = "AIzaSyCrGKPhY0JzopMwyMc1nXAp3U-Xj5zKvHU"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

def allowed_file(filename):
    """Check if uploaded file has a valid extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_frame(frame):
    """Convert frame to grayscale, apply Gaussian blur, and threshold for better OCR."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text(frame):
    """Extract text using Tesseract OCR and clean it."""
    text = pytesseract.image_to_string(frame, config='--psm 6').strip()
    text = " ".join(text.split()).lower()  # Normalize spaces, convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove numbers and symbols
    return text

def process_video(file_path):
    """Extract unique texts from video frames efficiently."""
    cap = cv2.VideoCapture(file_path)
    start_time = time.time()
    extracted_texts = set()  # Store unique texts
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(frame_count // 50, 1)  # Skip intelligently (~50 frames)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)
        extracted_text = extract_text(processed_frame)

        if extracted_text and len(extracted_text) > 3 and extracted_text not in extracted_texts:
            extracted_texts.add(extracted_text)

        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_skip)  # Skip frames

    cap.release()
    execution_time = time.time() - start_time
    return list(extracted_texts), execution_time

def analyze_text_with_gemini(text):
    """Analyze extracted text using Google Gemini AI."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": text}]}]
    }

    response = requests.post(
        GEMINI_ENDPOINT,
        headers=headers,
        json=payload,
        params={"key": GEMINI_API_KEY}
    )

    try:
        response_data = response.json()
        return response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No analysis available.")
    except Exception as e:
        return f"Error processing AI analysis: {str(e)}"

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    extracted_texts, execution_time = process_video(file_path)

    analysis = analyze_text_with_gemini("\n".join(extracted_texts))

    return jsonify({
        "message": "File uploaded and processed successfully!",
        "filename": filename,
        "extracted_texts": [f"<span style='background-color:yellow; font-weight:bold;'>{text}</span>" for text in extracted_texts],  # Bold & highlighted
        "execution_time": f"<strong>{execution_time:.2f} seconds</strong>",
        "analysis": analysis
    })

if __name__ == "__main__":
    app.run(debug=True)
