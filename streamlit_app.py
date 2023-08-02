
import streamlit as st
import openai
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2TokenizerFast, pipeline
import textwrap
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore")

openai.api_key = "your_api_key"

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
    print("Tokens in full transcript" + str(text_len_in_tokens))

    summary_token_len = count_tokens(summarized_text)
    print("Summary Token Length:"+ str(summary_token_len))

    return summarized_text.strip()

st.title("Transcription and Summary App")

audio_file = st.file_uploader("Upload MP3 Audio File", type=["mp3"])

if audio_file is not None:
    with open("temp.mp3", "wb") as f:
        f.write(audio_file.getbuffer())
    try:
        with open("temp.mp3", "rb") as audio:
            transcription = openai.Audio.translate("whisper-1", audio)["text"]
        st.write("Transcription: ", transcription)
        
        summarized_text = summarize_text(transcription)
        st.write("Summarized Text: ", summarized_text)

    except Exception as e:
        st.write("An error occurred: ", str(e))
