

import streamlit as st
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import openai
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2TokenizerFast, pipeline
import textwrap
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore")

# Get the password from Streamlit secrets
correct_password = st.secrets["password"]["value"]
password_placeholder = st.empty()
password = password_placeholder.text_input("Enter the password", type="password")
if password != correct_password:
    st.error("The password you entered is incorrect.")
    st.stop()

# Get the OpenAI key from Streamlit secrets
openai.api_key = st.secrets["openai"]["key"]

def split_audio(file_path, min_silence_len=500, silence_thresh=-40, chunk_length=30000):
    """
    Split an audio file into smaller chunks based on silence between audio segments.
    
    Parameters:
    - file_path: path to the audio file.
    - min_silence_len: minimum length of silence to consider for splitting (in ms).
    - silence_thresh: silence threshold (in dB). Anything quieter than this will be considered silence.
    - chunk_length: desired length of each chunk (in ms). Defaults to 30 seconds.
    
    Returns:
    - List of audio chunks as pydub.AudioSegment objects.
    """
    
    # Load audio file
    audio = AudioSegment.from_mp3(file_path)
    
    # Split audio into chunks based on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=100
    )
    
    # If chunks are longer than desired chunk_length, split them further
    split_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_length:
            num_mini_chunks = len(chunk) // chunk_length
            for i in range(num_mini_chunks):
                start_time = i * chunk_length
                end_time = start_time + chunk_length
                split_chunks.append(chunk[start_time:end_time])
        else:
            split_chunks.append(chunk)
            
    return split_chunks


st.title("Transcription and Summary App")

audio_file = st.file_uploader("Upload MP3 Audio File", type=["mp3"])

if audio_file is not None:
    try:
        # Write to a temp file
        with open("temp.mp3", "wb") as f:
            f.write(audio_file.getbuffer())

        # Splitting the audio into smaller chunks if file size exceeds 25MB
        audio_file_size = os.path.getsize("temp.mp3")
        if audio_file_size > 25 * 1024 * 1024:  # 25MB in bytes
            chunks = split_audio("temp.mp3")
            transcriptions = []
            for chunk in chunks:
                with open("temp_chunk.mp3", "wb") as f:
                    chunk.export(f, format="mp3")
                with open("temp_chunk.mp3", "rb") as audio:
                    transcription_chunk = openai.Audio.translate("whisper-1", audio)["text"]
                    transcriptions.append(transcription_chunk)
            transcription = " ".join(transcriptions)
        else:
            with open("temp.mp3", "rb") as audio:
                transcription = openai.Audio.translate("whisper-1", audio)["text"]

        st.write("Transcription: ", transcription)
        summarized_text = summarize_text(transcription)
        st.write("Summarized Text: ", summarized_text)
    except Exception as e:
        st.write("An error occurred: ", str(e))

