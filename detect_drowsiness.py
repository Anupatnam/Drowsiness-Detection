import cv2
import dlib
import numpy as np
from imutils import face_utils
import tensorflow as tf

print("✅ Script file is being run")  # Debug checkpoint

# === Load dlib face detector and landmark predictor ===
try:
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("✅ Dlib face detector and landmarks loaded")
except Exception as e:
    print(f"❌ Failed to load dlib components: {e}")
    exit()

# === Load CNN models ===
try:
    eye_models = [tf.keras.models.load_model(f'eye_cnn_{i}.keras') for i in range(1, 4)]
    mouth_models = [tf.keras.models.load_model(f'mouth_cnn_{i}.keras') for i in range(1, 4)]
    print("✅ All CNN models loaded")
except Exception as e:
    print(f"❌ Failed to load CNN models: {e}")
    exit()

# === Helper functions ===
def ensemble_predict(models, image):
    preds = [model.predict(image, verbose=0)[0] for model in models]
    avg_pred = np.mean(preds, axis=0)
    return np.argmax(avg_pred), avg_pred

def crop_and_process(region, gray):
    x, y, w, h = cv2.boundingRect(np.array([region]))
    roi = gray[y:y + h, x:x + w]
    roi = cv2.resize(roi, (50, 50))
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=-1)
    return roi.reshape(1, 50, 50, 1)

def detect_drowsiness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 0)

    if len(rects) == 0:
        print("⚠️ No face detected")
        return

    for rect in rects:
        shape = landmark_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[36:42]
        right_eye = shape[42:48]
        mouth = shape[48:68]

        left_eye_img = crop_and_process(left_eye, gray)
        right_eye_img = crop_and_process(right_eye, gray)
        mouth_img = crop_and_process(mouth, gray)

        left_eye_pred, _ = ensemble_predict(eye_models, left_eye_img)
        right_eye_pred, _ = ensemble_predict(eye_models, right_eye_img)
        mouth_pred, _ = ensemble_predict(mouth_models, mouth_img)

        eye_status = (left_eye_pred + right_eye_pred) / 2

        if eye_status == 1 or mouth_pred == 1:
            print("⚠️ Drowsiness Detected!")
        else:
            print("✅ Driver is Alert")

# === Main execution ===
if __name__ == "__main__":
    print("✅ Inside main block")

    # Load test image
    img_path = 'test_image.jpg'
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Failed to load image: {img_path}")
    else:
        print(f"✅ Image loaded: {img_path}")
        detect_drowsiness(img)
