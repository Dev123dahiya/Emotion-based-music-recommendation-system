from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "muse_v3.csv"
MODEL_PATH = BASE_DIR / "model.h5"
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"

EMOTION_DICT = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

EMOTION_TO_BUCKET = {
    "Neutral": "neutral",
    "Angry": "angry",
    "Fearful": "fear",
    "Happy": "happy",
    "Sad": "sad",
    "Disgusted": "sad",
    "Surprised": "happy",
}


@st.cache_data
def load_music_data() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(CSV_PATH)
    df["link"] = df["lastfm_url"]
    df["name"] = df["track"]
    df["emotional"] = df["number_of_emotion_tags"]
    df["pleasant"] = df["valence_tags"]
    df = df[["name", "emotional", "pleasant", "link", "artist"]]
    df = df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

    # Keep the original behavior: split ordered data into five mood buckets.
    n = len(df) // 5
    return {
        "sad": df[:n],
        "fear": df[n : 2 * n],
        "angry": df[2 * n : 3 * n],
        "neutral": df[3 * n : 4 * n],
        "happy": df[4 * n :],
    }


@st.cache_resource
def load_model() -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation="softmax"))
    model.load_weights(str(MODEL_PATH))
    return model


@st.cache_resource
def load_cascade() -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if cascade.empty():
        raise RuntimeError("Could not load haarcascade_frontalface_default.xml")
    return cascade


def top_unique_emotions(emotions: list[str]) -> list[str]:
    if not emotions:
        return []
    return [emotion for emotion, _ in Counter(emotions).most_common()]


def recommendation_split(size: int) -> list[int]:
    if size <= 1:
        return [30]
    if size == 2:
        return [30, 20]
    if size == 3:
        return [55, 20, 15]
    if size == 4:
        return [30, 29, 18, 9]
    return [10, 7, 6, 5, 2]


def recommend_songs(emotions: list[str], buckets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ranked = top_unique_emotions(emotions)[:5]
    if not ranked:
        ranked = ["Neutral"]

    picks = recommendation_split(len(ranked))
    out = pd.DataFrame(columns=["name", "artist", "link"])

    for emotion, n_rows in zip(ranked, picks):
        bucket_name = EMOTION_TO_BUCKET.get(emotion, "sad")
        source = buckets[bucket_name]
        n_rows = min(n_rows, len(source))
        sampled = source.sample(n=n_rows)
        out = pd.concat([out, sampled], ignore_index=True)

    return out


def decode_image(uploaded) -> np.ndarray | None:
    if uploaded is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded.getvalue()), dtype=np.uint8)
    if len(file_bytes) == 0:
        return None
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def detect_emotions(image_bgr: np.ndarray, model: Sequential, cascade: cv2.CascadeClassifier) -> tuple[list[str], np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    detected: list[str] = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y : y + h, x : x + w]
        roi = cv2.resize(roi_gray, (48, 48))
        roi = np.expand_dims(np.expand_dims(roi, -1), 0)
        pred = model.predict(roi, verbose=0)
        emotion = EMOTION_DICT[int(np.argmax(pred))]
        detected.append(emotion)

        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            image_bgr,
            emotion,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return detected, image_bgr


st.set_page_config(page_title="Emotion Based Music Recommendation", layout="wide")
st.title("Emotion Based Music Recommendation")
st.caption("Capture/upload a face image, detect emotion, and get song recommendations.")

buckets = load_music_data()
model = load_model()
cascade = load_cascade()

st.subheader("1) Provide an image")
camera_image = st.camera_input("Take a photo")
uploaded_image = st.file_uploader("Or upload image", type=["jpg", "jpeg", "png"])

st.subheader("2) Analyze")
manual_demo = st.multiselect(
    "Optional demo override (use if camera isn't available)",
    options=["Neutral", "Angry", "Fearful", "Happy", "Sad", "Disgusted", "Surprised"],
)

if st.button("Detect Emotion and Recommend Songs", type="primary"):
    emotions: list[str] = []
    annotated = None

    source = camera_image or uploaded_image
    image = decode_image(source) if source is not None else None
    if image is not None:
        emotions, annotated = detect_emotions(image, model, cascade)

    if manual_demo:
        emotions = manual_demo

    if annotated is not None:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected Faces", use_container_width=True)

    if not emotions:
        st.warning("No face detected. Try a clearer image or use demo override.")
    else:
        ranked = top_unique_emotions(emotions)
        st.success(f"Detected emotions: {', '.join(ranked)}")
        recommendations = recommend_songs(emotions, buckets).head(30)

        st.subheader("3) Recommended Songs")
        for idx, row in recommendations.iterrows():
            st.markdown(f"{idx + 1}. [{row['name']}]({row['link']})")
            st.caption(f"Artist: {row['artist']}")
