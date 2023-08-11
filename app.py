import streamlit as st
import joblib
import pandas as pd
import whisper
import os

print("Checking file existence...")
print("File exists:", os.path.exists("trained_detection_model_joblib.joblib"))

whisper_model = None

try:
    Model = joblib.load("trained_detection_model_joblib.joblib")
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)

try:
    whisper_model = whisper()
except Exception as e:
    st.error("Error loading Whisper model: " + str)


label_mapping = {0: "hate", 1: "offensive", 2: "neither"}

def transcribe(audio):

    text = whisper_model.transcribe(audio)

    text_list = []

    text_list.append(text)

    return text_list

def predict(list_to_check):
    pred_encoded = Model.predict(list_to_check)
    pred_labels = [label_mapping[pred] for pred in pred_encoded]

    result_df = pd.DataFrame({'text': list_to_check, 'predicted_label': pred_labels})

    result_df['class'] = result_df['predicted_label'].map(label_mapping)

    st.write("Predicted Results:")
    st.dataframe(result_df)

def main():
    st.title("Hate / Offensive Speech Detector From Audio")
    st.write("Upload an MP3 audio file and click the button to continue...:")

    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/mp3")

    if st.button("Continue"):
        if uploaded_file is not None:
            if whisper_model is not None:
                result = transcribe(uploaded_file)
                str.write(result)

                predict(result)

            else:
                st.error("Whisper model could not be loaded.")

    else:
        st.warning("Please upload an audio file before clicking the button.")

if __name__ == "__main__":
    main()