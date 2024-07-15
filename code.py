import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16

# Initialize the model
model = VGG16(weights='imagenet', include_top=True)

def preprocess_frame(frame):
    """Preprocess the video frame for the deep learning model."""
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def decode_predictions(preds, top=5):
    """Decode the predictions of the model."""
    from tensorflow.keras.applications.imagenet_utils import decode_predictions
    return decode_predictions(preds, top=top)[0]

def check_alert_conditions(preds):
    """Check if predictions meet alert conditions."""
    for pred in preds:
        if pred[1] == 'person' and pred[2] > 0.5:
            send_alert()
            break

def send_alert():
    """Send an alert notification."""
    print("Alert! Patient detected with high confidence.")

def predict_on_video(source=0):
    """Capture video and make predictions in real-time."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_frame = preprocess_frame(frame)
        preds = model.predict(preprocessed_frame)
        decoded_preds = decode_predictions(preds, top=1)

        check_alert_conditions(decoded_preds)

        # Display predictions on the video feed
        label = f"{decoded_preds[0][1]}: {decoded_preds[0][2]*100:.2f}%"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('TeleICU Monitoring', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start monitoring
predict_on_video()