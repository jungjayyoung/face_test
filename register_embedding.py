import os
import glob
import cv2
import numpy as np
from insightface.app import FaceAnalysis

REGISTERED_IMAGE_DIR = "registered"
DB_DIR = "db"
EMBEDDING_PATH = "db/known_embedding.npy"

FACE_MODEL_NAME = "buffalo_sc"
FACE_PROVIDER = ["CPUExecutionProvider"]
FACE_DET_SIZE = (320, 320)

os.makedirs(DB_DIR, exist_ok=True)

app = FaceAnalysis(
    name=FACE_MODEL_NAME,
    providers=FACE_PROVIDER
)

app.prepare(
    ctx_id=0,
    det_size=FACE_DET_SIZE
)


def get_image_paths():
    paths = []

    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        paths.extend(glob.glob(os.path.join(REGISTERED_IMAGE_DIR, ext)))

    return paths


def extract_embedding(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"[ERROR] 이미지 읽기 실패: {image_path}")
        return None

    faces = app.get(img)

    if len(faces) == 0:
        print(f"[ERROR] 얼굴 검출 실패: {image_path}")
        return None

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    return face.embedding


def main():
    image_paths = get_image_paths()

    if len(image_paths) == 0:
        print(f"[ERROR] 등록 이미지가 없습니다: {REGISTERED_IMAGE_DIR}")
        return

    embeddings = []

    for path in image_paths:
        embedding = extract_embedding(path)

        if embedding is None:
            continue

        embeddings.append(embedding)
        print(f"[OK] 등록 이미지 처리 완료: {path}")

    if len(embeddings) == 0:
        print("[ERROR] 사용 가능한 embedding이 없습니다.")
        return

    known_embedding = np.mean(embeddings, axis=0)

    np.save(EMBEDDING_PATH, known_embedding)

    print("[OK] 등록 embedding 생성 완료")
    print(f"[PATH] {EMBEDDING_PATH}")


if __name__ == "__main__":
    main()