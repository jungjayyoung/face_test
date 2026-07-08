# Face Test

InsightFace를 이용하여 등록된 얼굴(Embedding)과 테스트 이미지를 비교하는 프로젝트입니다.

---

# 프로젝트 구조

```text
face_test/
│
├── db/
│   └── known_embedding.npy      # 생성된 등록 임베딩
│
├── registered/
│   ├── person1.png
│   └── person2.png
│
├── test/
│   └── test1.png
│
├── register_embedding.py
├── verify_face.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 개발 환경

- Python 3.11+
- InsightFace
- ONNX Runtime
- OpenCV

---

# 설치

가상환경 생성

```bash
python -m venv venv
```

Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

macOS / Linux

```bash
source venv/bin/activate
```

패키지 설치

```bash
pip install -r requirements.txt
```

---

# 면허증 용 등록 이미지 준비

등록할 면허증 이미지를

```
registered/
```

폴더에 넣습니다.

예시

```
registered/
    my_license.png

```

---

# 면허증 Embedding 생성

```bash
python register_embedding.py
```

성공하면

```
db/known_embedding.npy
```

파일이 생성됩니다.

예시 출력

```
[OK] 등록 이미지 처리 완료
[OK] 등록 embedding 생성 완료
```

---

# 얼굴 인증 테스트

비교할 이미지를

```
test/
```

폴더에 넣습니다.

예시

```
test/
    test1.png
```

실행

```bash
python verify_face.py
```

또는

```bash
python verify_face.py test/person2.png
```

---

# 결과 예시

성공

```
========== Face Verify Result ==========
Similarity : 0.9142
Threshold  : 0.5000
Match      : True

[RESULT] 동일인으로 판단
```

실패

```
========== Face Verify Result ==========
Similarity : 0.3185
Threshold  : 0.5000
Match      : False

[RESULT] 다른 사람으로 판단
```

---

# 얼굴 비교 방식

1. 등록 이미지에서 얼굴 Embedding 생성
2. 여러 장일 경우 평균 Embedding 생성
3. 테스트 이미지에서 얼굴 Embedding 생성
4. Cosine Similarity 계산
5. Threshold 이상이면 동일인으로 판단

---

# Threshold

현재 기준

```
0.5
```

Threshold는

```python
MATCH_THRESHOLD
```

값을 수정하여 변경할 수 있습니다.

---

# 주의사항

- 등록 이미지에는 얼굴이 선명하게 보여야 합니다.
- 한 이미지에는 가능한 한 한 명만 포함하는 것이 좋습니다.
- 조명이 충분한 사진을 사용하는 것이 좋습니다.
- 등록 이미지를 여러 장 사용할수록 정확도가 향상됩니다.
- `db/known_embedding.npy`는 자동 생성되므로 Git에 포함하지 않습니다.

---

# License

MIT License