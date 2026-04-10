from flask import Flask, render_template, request, jsonify, session
import cv2
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace

app = Flask(__name__)
app.secret_key = "secret_key_123"

IMG_PATH = "static/temp.jpg"
DB_FILE = "users.json"


# ---------------- DB ----------------
def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ---------------- CAMERA CAPTURE ----------------
@app.route("/capture", methods=["POST"])
def capture():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cam.read()
    cam.release()

    if not ret or frame is None:
        return jsonify({"status": "camera_error"})

    os.makedirs("static", exist_ok=True)
    cv2.imwrite(IMG_PATH, frame)

    return jsonify({"status": "ok"})


# ---------------- FACE EMBEDDING ----------------
def get_embedding(img_path):
    try:
        embedding = DeepFace.represent(
            img_path,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

        emb = np.array(embedding)

        # normalize
        norm = np.linalg.norm(emb)
        if norm != 0:
            emb = emb / norm

        return emb

    except Exception as e:
        print("Embedding error:", e)
        return None


# ---------------- FUSION (ASSIMILATION) ----------------
def fuse_embeddings(emb_list):
    emb_list = np.array(emb_list)

    fused = np.mean(emb_list, axis=0)

    norm = np.linalg.norm(fused)
    if norm != 0:
        fused = fused / norm

    return fused


# ---------------- SIGNUP (5 SAMPLE TRAINING) ----------------
@app.route("/signup", methods=["POST"])
def signup():
    username = request.json.get("username")

    if not username:
        return jsonify({"status": "username_missing"})

    db = load_db()

    if username in db:
        return jsonify({"status": "user_exists"})

    embeddings = []

    # capture 5 face samples
    for i in range(5):

        emb = get_embedding(IMG_PATH)

        if emb is None:
            return jsonify({"status": "no_face", "step": i})

        embeddings.append(emb)

    # fuse into single identity vector
    fused_emb = fuse_embeddings(embeddings)

    db[username] = fused_emb.tolist()
    save_db(db)

    return jsonify({
        "status": "user_saved",
        "redirect": "/login-page"
    })


# ---------------- LOGIN ----------------
@app.route("/login", methods=["POST"])
def login():

    emb = get_embedding(IMG_PATH)

    if emb is None:
        return jsonify({"status": "no_face"})

    db = load_db()

    if len(db) == 0:
        return jsonify({"status": "no_users"})

    best_user = None
    best_score = -1

    for user, stored_emb in db.items():

        stored_emb = np.array(stored_emb)

        score = cosine_similarity([emb], [stored_emb])[0][0]

        if score > best_score:
            best_score = score
            best_user = user

    if best_score >= 0.90:
        session["user"] = best_user

        return jsonify({
            "status": "success",
            "user": best_user,
            "score": float(best_score),
            "redirect": "/dashboard"
        })

    return jsonify({
        "status": "not_matched",
        "score": float(best_score)
    })


# ---------------- PAGES ----------------
@app.route("/")
def signup_page():
    return render_template("signup.html")


@app.route("/login-page")
def login_page():
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login-page")
    return render_template("dashboard.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "logged_out"})


# ---------------- RUN ----------------
if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
