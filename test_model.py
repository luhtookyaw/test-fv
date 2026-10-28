import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
    allowed_modules=["detection", "recognition"]
)

app.prepare(ctx_id=-1, det_size=(640, 640))


def get_embedding(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(image_path)

    faces = app.get(image)

    if len(faces) != 1:
        raise ValueError(
            f"{image_path}: expected 1 face, found {len(faces)}"
        )

    return faces[0].normed_embedding


embedding1 = get_embedding("images/saoirse_ronan_1.jpg")
embedding2 = get_embedding("images/saoirse_ronan_2.jpg")

# Already L2-normalized embeddings
similarity = float(np.dot(embedding1, embedding2))

print(f"Cosine similarity: {similarity:.4f}")