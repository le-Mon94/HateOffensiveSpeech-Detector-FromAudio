import joblib
import gradio as gr
import pandas as pd
from pydub import AudioSegment
import os
import math
import whisper
import numpy as np

import statistics
from statistics import mode

def split_text(text):
    return text.split()

def split_sentence(sentence):
    sentences = sentence.split('.')
    
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

PredictModel = joblib.load("trained_detection_model.joblib")
WhisperModel = whisper.load_model("medium")

def PredictFunction(audio_file):

  label_mapping = label_mapping = {0: 'hate', 1: 'offensive', 2: 'no Hate / Offensive'}

  text_result = WhisperModel.transcribe(audio_file)

  result_list = split_sentence(text_result["text"])

  print(text_result["text"])

  pred = PredictModel.predict(result_list)

  pred_labels = [label_mapping[label] for label in pred]

  result_df = pd.DataFrame({'text': result_list, 'predicted_label': pred_labels})

  result_mode = mode(pred_labels)

  return result_df, "This audio contains " + result_mode + " speeches"

demo = gr.Blocks()

mic_transcribe = gr.Interface(
    fn=PredictFunction,
    inputs=gr.Audio(source="microphone", type="filepath", label="Record an audio with your mic"),
    outputs=[gr.Dataframe(label="Prediction DataFrame"), gr.outputs.Textbox(label="Result")],
    title="Hate / Offensive Speech Detector From Audio",
    description="Detect Hate and Offensive speeches using Whisper AI to transcribe the audio and pretrained model from skleanr and LinearSVC to detect it"
)

file_transcribe = gr.Interface(
    fn=PredictFunction,
    inputs=gr.Audio(source="upload", type="filepath", label="Browse and upload your audio"),
    outputs=[gr.Dataframe(label="Prediction DataFrame"), gr.outputs.Textbox(label="Result")],
    title="Hate / Offensive Speech Detector From Audio",
    description="Detect Hate and Offensive speeches using Whisper AI to transcribe the audio and pretrained model from sklearn and LinearSVC to detect it"
)

with demo:
    gr.TabbedInterface(
        [mic_transcribe, file_transcribe],
        ["Transcribe Microphone", "Transcribe Audio File"],
    )

demo.launch(inline = False)