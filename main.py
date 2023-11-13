import cv2
import time
import streamlit as st
from deepface import DeepFace
import pandas as pd
import os
from datetime import timedelta

data_columns = ['Name', 'ID Number', 'DOB', 'Date', 'Sample Number', 'Uploaded File', 'Downloaded File', 'Dominant Emotion', 'Angry Duration', 'Neutral Duration', 'Happy Duration', 'Fear Duration', 'Sad Duration', 'Emotion Occurrences', 'Total Video Duration']
data_df = pd.DataFrame(columns=data_columns)

csv_filename = "emotion_data.csv"
if os.path.exists(csv_filename):
    data_df = pd.read_csv(csv_filename, dtype={'ID Number': str})

def save_data(name, id_number, dob, date, sample_number, uploaded_file_name, downloaded_file_name, dominant_emotion, emotion_durations, emotion_occurrences, total_video_duration):
    new_row = {'Name': name, 'ID Number': id_number, 'DOB': dob, 'Date': date, 'Sample Number': sample_number, 'Uploaded File': uploaded_file_name, 'Downloaded File': downloaded_file_name, 'Dominant Emotion': dominant_emotion}
    
    for emotion, duration in emotion_durations.items():
        new_row[f'{emotion.capitalize()} Duration'] = duration

    new_row['Emotion Occurrences'] = str(emotion_occurrences)
    new_row['Total Video Duration'] = total_video_duration
    
    data_df.loc[len(data_df)] = new_row
    data_df.to_csv(csv_filename, index=False)

def run_emotion_detection(video_path, id_number):
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    desired_width = 500
    desired_height = 500
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1 / fps
    emotion_occurrences = {emotion: [] for emotion in ['angry', 'neutral', 'happy', 'fear', 'sad']}
    last_occurrence_second = None
    frames = []

    while cap.isOpened():
        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = frame_num / fps
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (desired_width, desired_height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            face_img = frame[y:y + h, x:x + w]
            resized_face = cv2.resize(face_img, (48, 48), interpolation=cv2.INTER_AREA)
            emotion_preds = DeepFace.analyze(resized_face, actions=['emotion'], enforce_detection=False)
            dominant_emotion = emotion_preds[0]['dominant_emotion']

            if last_occurrence_second != int(current_time):
                emotion_occurrences[dominant_emotion].append(current_time)
                last_occurrence_second = int(current_time)

            # Overlay detected emotion on the frame
            emotion_label = f"Emotion: {dominant_emotion}"
            cv2.rectangle(frame, (10, 10), (300, 50), (0, 0, 0), -1)
            cv2.putText(frame, emotion_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(frame_delay)

    cap.release()
    detected_video_filename = f"{id_number}.mp4"
    out_path = os.path.join("detected_video", detected_video_filename)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (desired_width, desired_height))
    for frame in frames:
        out.write(frame)
    out.release()
    total_duration_seconds = len(frames) / fps
    emotion_durations = {emotion: timedelta(seconds=sum(occurrences[i+1] - occurrences[i] for i in range(len(occurrences)-1))) for emotion, occurrences in emotion_occurrences.items()}
    return out_path, emotion_occurrences, total_duration_seconds, emotion_durations

def main():
    st.title("Face Emotion Detection")
    name = st.text_input("Name")
    id_number = st.text_input("ID Number")
    dob = st.text_input("DOB")
    date = st.date_input("Date")
    sample_number = st.text_input("Phone Number")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"], key="uploaded_file")

    if uploaded_file is not None:
        if st.button("Run Emotion Detection"):
            # Check if ID number already exists
            if data_df['ID Number'].str.contains(id_number).any():
                st.warning("ID Number already exists. Please enter a different ID Number.")
            else:
                st.text("Running emotion detection...")
                with open("temp_video.mp4", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                output_video_path, emotion_occurrences, total_duration_seconds, emotion_durations = run_emotion_detection("temp_video.mp4", id_number)
                st.text("Emotion detection complete")
                st.text("Original Uploaded Video:")
                st.video(uploaded_file)
                st.text("Emotion Detected Video:")
                
                # Provide a download link for the emotion-detected video
                with open(output_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.download_button(label="Download Emotion-Detected Video", data=video_bytes, file_name="emotion_detected_video.mp4", mime="video/mp4")

                dominant_emotion = max(emotion_occurrences, key=lambda k: len(emotion_occurrences[k]))

                st.text("Emotion Occurrences:")
                for emotion, occurrences in emotion_occurrences.items():
                    st.info(f"**{emotion.capitalize()} Total Occurrences: {len(occurrences)}**")
                    st.write(f"**{emotion.capitalize()} Timestamps:**")
                    for i, occurrence in enumerate(occurrences):
                        st.write(f"Occurrence {i+1}: {timedelta(seconds=occurrence)}")

                st.info("Dominant Emotion: " + dominant_emotion)
                st.info(f"Total Video Duration: {timedelta(seconds=total_duration_seconds)}")

                save_data(name, id_number, dob, date, sample_number, uploaded_file.name, output_video_path, dominant_emotion, emotion_durations, emotion_occurrences, timedelta(seconds=total_duration_seconds))

if __name__ == "__main__":
    main()

