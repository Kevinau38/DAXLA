from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import pickle
import os

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
detection_results = []

# Load trained model
try:
    with open('micro_model_simple.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Trained model loaded successfully!")
    MODEL_AVAILABLE = True
except:
    print("No trained model found. Run simple_train.py first.")
    MODEL_AVAILABLE = False

class TrainedDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame_count = 0
        self.last_predictions = []
        self.stable_result = None
        self.last_face_coords = None  # Lưu tọa độ face
        
    def predict_micro_expression(self, face_roi):
        """Predict using trained model"""
        if not MODEL_AVAILABLE:
            return None, 0.5
            
        try:
            # Preprocess face
            face_resized = cv2.resize(face_roi, (48, 48))
            face_flattened = face_resized.flatten().astype('float32') / 255.0
            face_input = face_flattened.reshape(1, -1)
            
            # Predict
            prediction = model.predict(face_input)[0]
            probabilities = model.predict_proba(face_input)[0]
            confidence = np.max(probabilities)
            
            # Cải thiện logic cân bằng - dựa vào kích thước khuôn mặt
            face_area = face_roi.shape[0] * face_roi.shape[1]
            
            # Logic cân bằng đơn giản hơn - nhanh hơn
            if prediction == 1 and probabilities[1] < 0.75:  # Nếu lie confidence < 75%
                if probabilities[0] > 0.3:  # Và truth > 30%
                    prediction = 0
                    confidence = probabilities[0]
            
            return prediction, confidence
        except:
            return None, 0.5
        
    def get_frame(self):
        global detection_results
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            
            # Detect every 2 frames (siêu nhanh)
            if self.frame_count % 2 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Face detection nhanh như trước
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.15,    # Giữ nhanh
                    minNeighbors=3,      # Giữ nhanh
                    minSize=(40, 40)     # Giữ nhạy
                )
                
                if len(faces) > 0:
                    # Chỉ lấy mặt lớn nhất (loại bỏ false detection)
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    self.last_face_coords = (x, y, w, h)  # Lưu tọa độ
                    
                    # Extract face ROI
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Predict using trained model
                    predicted_class, confidence = self.predict_micro_expression(face_roi)
                    
                    if predicted_class is not None and confidence > 0.55:  # Giữ threshold thấp
                        # Phản hồi tức thì - cập nhật ngay
                        self.stable_result = {
                            'label': "Truth" if predicted_class == 0 else "Lie",
                            'confidence': confidence
                        }
                        
                        print(f"Detection: {self.stable_result['label']} - {self.stable_result['confidence']:.2f}")
            
            # Vẽ khung nếu có kết quả và tọa độ face
            if self.stable_result and self.last_face_coords:
                x, y, w, h = self.last_face_coords
                label = self.stable_result['label']
                confidence = self.stable_result['confidence']
                color = (0, 255, 0) if label == "Truth" else (0, 0, 255)
                
                cv2.rectangle(frame, (x, y-30), (x+w, y+h+10), color, 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", 
                           (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            detection_results = [self.stable_result] if self.stable_result else []
        
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg.tobytes(), detection_results if 'detection_results' in locals() else []

detector = TrainedDetector()

@app.route('/')
def index():
    return render_template('micro_index.html')

def gen():
    import time
    while True:
        frame, _ = detector.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        time.sleep(0.033)

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    return jsonify(detection_results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)