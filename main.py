import os
import cv2
from deepface import DeepFace

#Disable oneDNN optimizations to avoid TensorFlow messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def detect_emotion():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        try:
            # Change the detection backend
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True, detector_backend='mtcnn')
            
            # Only consider the first detected face
            if len(analysis) > 0:
                face = analysis[0]
                emotion = face['dominant_emotion']
                
                # Verify the structure of the face region.
                print(face['region'])  # To understand what the content of 'region' looks like
                
                # Modify the extraction of face coordinates
                x = face['region']['x']
                y = face['region']['y']
                w = face['region']['w']
                h = face['region']['h']
                
                if emotion in ['happy', 'smile']:
                    label = "Feliz"
                    color = (0, 255, 0)  # Green
                elif emotion == 'angry':
                    label = "Enojado"
                    color = (0, 0, 255)  # Red
                elif emotion == 'sad':
                    label = "Triste"
                    color = (255, 0, 0)  # Blue
                elif emotion == 'surprise':
                    label = "Sorprendido"
                    color = (255, 255, 0)  # Yellow
                else:
                    label = "Pensativo"
                    color = (0, 165, 255)  # Orange
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        except Exception as e:
            print(f"Error: {e}")
            cv2.putText(frame, "Error processing frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.putText(frame, "Press 'q' to exit", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Emotion Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Process terminated by user.")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotion()
