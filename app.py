"""
Flask web app for gender classification using KNN, Decision Tree, and Naive Bayes.
Uses HOG features — must match train_models.py preprocessing exactly.
"""
import os, pickle, json, io, base64
import numpy as np
from PIL import Image
from skimage.feature import hog
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# ── Load config FIRST so IMG_SIZE is available everywhere ────────────────────
_cfg_path = os.path.join(MODELS_DIR, 'config.json')
if os.path.exists(_cfg_path):
    with open(_cfg_path) as _f:
        _cfg = json.load(_f)
    IMG_SIZE = tuple(_cfg.get('img_size', [128, 128]))
else:
    IMG_SIZE = (128, 128)

# Load OpenCV face detector once at startup
_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
FACE_CASCADE  = cv2.CascadeClassifier(_cascade_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ── Preprocessing — mirrors train_models.py exactly ──────────────────────────
def crop_face(img_rgb_array):
    """Detect largest face, crop with padding, resize. Falls back to centre-crop."""
    gray    = cv2.cvtColor(img_rgb_array, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    faces = []
    for img_g, scale, neighbors, min_sz in [
        (gray_eq, 1.05, 3, (30, 30)),
        (gray_eq, 1.1,  2, (20, 20)),
        (gray,    1.05, 3, (30, 30)),
        (gray,    1.1,  2, (20, 20)),
    ]:
        faces = FACE_CASCADE.detectMultiScale(
            img_g, scaleFactor=scale, minNeighbors=neighbors, minSize=min_sz
        )
        if len(faces) > 0:
            break

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        pad = int(max(w, h) * 0.20)
        x1 = max(0, x - pad);  y1 = max(0, y - pad)
        x2 = min(img_rgb_array.shape[1], x + w + pad)
        y2 = min(img_rgb_array.shape[0], y + h + pad)
        face = img_rgb_array[y1:y2, x1:x2]
    else:
        # centre-crop fallback
        h, w = img_rgb_array.shape[:2]
        s  = min(h, w)
        y1 = (h - s) // 2;  x1 = (w - s) // 2
        face = img_rgb_array[y1:y1+s, x1:x1+s]

    return cv2.resize(face, IMG_SIZE, interpolation=cv2.INTER_AREA)

def extract_hog(img_rgb_array):
    """Crop face → equalise → HOG. Mirrors train_models.py exactly."""
    cropped  = crop_face(img_rgb_array)
    gray     = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    gray     = cv2.equalizeHist(gray)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )
    return features

def preprocess(file_bytes):
    img  = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    arr  = np.array(img)
    feat = extract_hog(arr)
    return feat.reshape(1, -1)

# ── Load models ───────────────────────────────────────────────────────────────
def load_models():
    models = {}
    for key, fname in [('KNN', 'knn.pkl'),
                       ('Decision Tree', 'decision_tree.pkl'),
                       ('Naive Bayes', 'naive_bayes.pkl')]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[key] = pickle.load(f)
    return models

def load_results():
    path = os.path.join(MODELS_DIR, 'results.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

MODELS  = load_models()
RESULTS = load_results()
CLASSES = ['Female', 'Male']

# ── Confusion matrix as base64 ────────────────────────────────────────────────
def cm_to_base64(cm_list, model_name):
    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{model_name} – Confusion Matrix')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── Compute per-class and macro metrics from a 2×2 confusion matrix ──────────
def compute_metrics(cm_list):
    """
    Given a 2×2 CM [[TN, FP], [FN, TP]] (classes: Female=0, Male=1),
    return per-class precision/recall/F1 and macro averages.
    """
    cm = np.array(cm_list)
    metrics = {}
    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        metrics[cls] = {
            'precision': round(precision * 100, 2),
            'recall':    round(recall    * 100, 2),
            'f1':        round(f1        * 100, 2),
        }
    # macro averages
    metrics['macro'] = {
        'precision': round(np.mean([metrics[c]['precision'] for c in CLASSES]), 2),
        'recall':    round(np.mean([metrics[c]['recall']    for c in CLASSES]), 2),
        'f1':        round(np.mean([metrics[c]['f1']        for c in CLASSES]), 2),
    }
    return metrics

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    trained = bool(MODELS)
    return render_template('index.html', trained=trained, results=RESULTS)

@app.route('/metrics')
def metrics():
    """Model performance metrics page."""
    if not RESULTS:
        return render_template('metrics.html', metrics_data={}, cms={}, trained=False)

    metrics_data = {}
    cms = {}
    for name, res in RESULTS.items():
        metrics_data[name] = {
            'accuracy': res.get('accuracy', 0),
            **compute_metrics(res['cm']),
        }
        cms[name] = cm_to_base64(res['cm'], name)

    return render_template('metrics.html', metrics_data=metrics_data, cms=cms, trained=True)

@app.route('/predict', methods=['POST'])
def predict():
    if not MODELS:
        return jsonify({'error': 'Models not trained yet. Run train_models.py first.'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type. Please upload a JPG or PNG image.'
        }), 400

    data = file.read()

    try:
        X = preprocess(data)
    except Exception as e:
        return jsonify({'error': f'Could not process image: {e}'}), 400

    predictions = {}
    for name, clf in MODELS.items():
        pred = clf.predict(X)[0]
        try:
            proba = clf.predict_proba(X)[0]
            conf  = round(float(max(proba)) * 100, 1)
        except Exception:
            conf = None
        predictions[name] = {
            'label':      CLASSES[pred],
            'confidence': conf,
            'accuracy':   RESULTS.get(name, {}).get('accuracy', 'N/A'),
        }

    best = max(predictions,
               key=lambda k: predictions[k]['accuracy']
               if isinstance(predictions[k]['accuracy'], float) else 0)

    cms = {}
    for name, res in RESULTS.items():
        cms[name] = cm_to_base64(res['cm'], name)

    return jsonify({'predictions': predictions, 'best_model': best, 'cms': cms})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
