# Real-Time Emotion Detection with DeepFace  

This project explores the capabilities of DeepFace, a powerful Python library for deep learning-based facial recognition and analysis. The script utilizes DeepFace and OpenCV to detect emotions in real-time using a webcam.  

## 🚀 About the Project  
The goal of this project is to experiment with DeepFace's emotion detection capabilities. The current implementation detects a single face in the video stream and classifies its dominant emotion as happy, angry, sad, surprised, or neutral (thoughtful).  

This project is still in an exploratory phase, and future improvements could include:  
- Enhancing accuracy with different deep learning models.  
- Recognizing multiple faces simultaneously.  
- Optimizing performance for real-time applications.  

## 🛠️ Technologies Used  
- Python  
- OpenCV  
- DeepFace  
- TensorFlow  

## 📌 How It Works  
1. The script captures video from the webcam.  
2. It processes each frame using DeepFace’s `analyze` function to detect emotions.  
3. A bounding box is drawn around the detected face with a label indicating the predicted emotion.  
4. The user can exit by pressing 'q'.  

## 📦 Installation  
Before running the script, ensure you have the required dependencies installed in the file: requirements.txt  

1. Clone the repository:  
   ```bash
   [git clone https://github.com/AlejandroMeyer/Facial-Recognition]

----

Feel free to contribute or report any issues!
