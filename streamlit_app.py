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
    st.write("Splitting audio into smaller chunks...")
    progress_bar = st.progress(0)
    
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
    for i, chunk in enumerate(chunks):
        if len(chunk) > chunk_length:
            num_mini_chunks = len(chunk) // chunk_length
            for j in range(num_mini_chunks):
                start_time = j * chunk_length
                end_time = start_time + chunk_length
                split_chunks.append(chunk[start_time:end_time])
        else:
            split_chunks.append(chunk)
        
        # Here's where you'd update the progress bar within the function
        progress_bar.progress(i / len(chunks))
            
    return split_chunks

st.title("Transcription and Summary App")

# Initialize the processing stage if it's not set
if "stage" not in st.session_state:
    st.session_state.stage = 0

# Stage 0: Wait for the user to upload a file
if st.session_state.stage == 0:
    audio_file = st.file_uploader("Upload MP3 Audio File", type=["mp3"])
    if audio_file is not None:
        st.session_state.stage = 1

# Stage 1: Transcribe the audio
if st.session_state.stage == 1:
    if audio_file is not None:
        try:
            # Write to a temp file
            with open("temp.mp3", "wb") as f:
                f.write(audio_file.getbuffer())
    
            # Splitting the audio into smaller chunks if file size exceeds 25MB
            audio_file_size = os.path.getsize("temp.mp3")
            if audio_file_size > 25 * 1024 * 1024:  # 25MB in bytes
                if st.button("start transcription now"):
                    progress_bar = st.progress(0)
                    chunks = split_audio("temp.mp3")
                    progress_bar = st.progress(0)
                    transcriptions = []
                    for i, chunk in enumerate(chunks):
                        progress_bar.progress(i / len(chunks))
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
            st.session_state.stage = 2

        except Exception as e:
            st.write("An error occurred: ", str(e))
    
# Stage 2: Summarize the transcription
if st.session_state.stage == 2:
    try:
        if st.button("summarize now"):
            st.write("Summarized Text: ", summarized_text)
            st.session_state.stage = 3  # or reset to 0 if you want the process to be repeatable
    except Exception as e:
        st.write("An error occurred: ", str(e))

# Functions for token count, truncation, summarization, etc.
def count_tokens(input_data, max_tokens=20000, input_type='text'):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if input_type == 'text':
        tokens = tokenizer.tokenize(input_data)
    elif input_type == 'tokens':
        tokens = input_data
    else:
        raise ValueError("Invalid input_type. Must be 'text' or 'tokens'")
    token_count = len(tokens)
    return token_count

def truncate_text_by_tokens(text, max_tokens=3000):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokens = tokenizer.tokenize(text)
    truncated_tokens = tokens[:max_tokens]
    trunc_token_len = count_tokens(truncated_tokens, input_type='tokens')
    truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
    return truncated_text

def summarize_chunk(classifier, chunk):
    summary = classifier(chunk)
    return summary[0]["summary_text"]

def summarize_text(text, model_name="t5-small", max_workers=8):
    classifier = pipeline("summarization", model=model_name)
    summarized_text = ""
    chunks = textwrap.wrap(text, width=500, break_long_words=False)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        summaries = executor.map(lambda chunk: summarize_chunk(classifier, chunk), chunks)
        summarized_text = " ".join(summaries)
    text_len_in_tokens = count_tokens(text)
    summary_token_len = count_tokens(summarized_text)
    if summary_token_len > 2500:
        summarized_text = truncate_text_by_tokens(summarized_text, max_tokens=2500)
    with open("transcript_summary.txt", "w") as file:
        file.write(summarized_text)
    return summarized_text.strip()

def gpt_summarize_transcript(transcript_text, token_len):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at summarizing long documents into concise and comprehensive summaries. Your summaries often capture the essence of the original text."},
            {"role": "user", "content": "I have a long transcript that I would like you to summarize for me. Please think carefully and do the best job you possibly can."},
            {"role": "system", "content": "Absolutely, I will provide a concise and comprehensive summary of the transcript."},
            {"role": "user", "content": "Excellent, here is the transcript: " + transcript_text}
        ],
        max_tokens=3800 - token_len,
        n=1,
        stop=None,
        temperature=0.5,
    )
    summary = response.choices[0].message["content"]
    return summary

# Additional functionalities and features can be added as per requirements.

