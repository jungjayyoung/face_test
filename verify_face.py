import os
import sys
import cv2
import numpy as np
from insightface.app import FaceAnalysis

EMBEDDING_PATH = "db/known_embedding.npy"
DEFAULT_TEST_IMAGE = "test/test1.png"

FACE_MODEL_NAME = "buffalo_sc"
FACE_PROVIDER = ["CPUExecutionProvider"]
FACE_DET_SIZE = (320, 320)

MATCH_THRESHOLD = 0.5

app = FaceAnalysis(
    name=FACE_MODEL_NAME,
    providers=FACE_PROVIDER
)

app.prepare(
    ctx_id=0,
    det_size=FACE_DET_SIZE
)


def cosine_similarity(a, b):
    return float(
        np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b))
    )


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
    test_image_path = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_TEST_IMAGE

    if not os.path.exists(EMBEDDING_PATH):
        print(f"[ERROR] 등록 embedding 파일이 없습니다: {EMBEDDING_PATH}")
        print("먼저 register_embedding.py를 실행하세요.")
        return

    if not os.path.exists(test_image_path):
        print(f"[ERROR] 테스트 이미지가 없습니다: {test_image_path}")
        return

    known_embedding = np.load(EMBEDDING_PATH)
    test_embedding = extract_embedding(test_image_path)

    if test_embedding is None:
        return

    similarity = cosine_similarity(known_embedding, test_embedding)
    is_match = similarity >= MATCH_THRESHOLD

    print("========== Face Verify Result ==========")
    print(f"Test Image : {test_image_path}")
    print(f"Similarity : {similarity:.4f}")
    print(f"Threshold  : {MATCH_THRESHOLD}")
    print(f"Match      : {is_match}")

    if is_match:
        print("[RESULT] 동일인으로 판단")
    else:
        print("[RESULT] 다른 사람으로 판단")


if __name__ == "__main__":
    main()