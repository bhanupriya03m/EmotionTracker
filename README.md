# EmoInsight Engine

## Overview
EmoInsight Engine is a video emotion detection system built using Python and the DeepFace library. This project allows users to upload a video file, analyze the emotions present in the video, and save the results, including emotion occurrences and dominant emotions, to a CSV file.

## Features
- **Emotion Detection**: Detects and analyzes facial expressions in a video to identify emotions such as anger, neutral, happy, fear, and sad.
- **Dominant Emotion**: Identifies the dominant emotion throughout the video.
- **Timestamped Occurrences**: Provides timestamped occurrences for each detected emotion in the video.
- **CSV Data Storage**: Stores analyzed data, including user information, emotion occurrences, and dominant emotion, in a CSV file for future reference.

## Installation
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/EmoInsight-Engine.git
   cd EmoInsight-Engine
## Usage
1. Run the application:
   ```bash
   streamlit run main.py
2. Access the web interface at [http://localhost:8501](http://localhost:8501) in your browser.
3. Input user information, upload a video file, and click "Run Emotion Detection" to analyze the video.
4. View the analyzed results, including emotion occurrences, dominant emotion, and download the emotion-detected video.

## Project Structure
- **main.py**: Main script containing the Streamlit web application and emotion detection logic.
- **detected_video/**: Directory to store emotion-detected videos.
- **emotion_data.csv**: CSV file to store analyzed data.

## Dependencies
- OpenCV
- Streamlit
- DeepFace
- pandas

## Notes
- The project uses Haar Cascade for face detection and DeepFace for emotion analysis.
- Emotion detection results, including occurrences and dominant emotion, are saved in the `emotion_data.csv` file.

![Project Demo](https://github.com/bhanupriya03m/main_image_streamlit/blob/main/main-%C2%B7-Streamlit.png)
![Project Image](https://github.com/bhanupriya03m/main_image_streamlit)

