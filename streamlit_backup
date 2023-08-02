import streamlit as st
from audio_transcript_mp3_modified import get_transcript, count_tokens, summarize_text, gpt_summarize_transcript, generate_tweet_thread, generate_long_form_article

# Set page title
st.title("Transcription and Summary App")

# Upload audio file
audio_file = st.file_uploader("Upload MP3 Audio File", type=["mp3"])

if audio_file is not None:
    with open("temp.mp3", "wb") as f:
        f.write(audio_file.getbuffer())

    try:
        # Get the transcript
        transcription = get_transcript("temp.mp3")
        st.write("Transcription: ", transcription)

        # Get the token length of the transcript
        token_count = count_tokens(transcription)
        st.write("Token Count: ", token_count)

        # Summarize with either GPT3 or T5 depending on length of transcript:
        if token_count > 3000:
            summarized_text = summarize_text(transcription)
            new_token_count = count_tokens(summarized_text)
        else:
            summarized_text = gpt_summarize_transcript(transcription, token_count)
            new_token_count = count_tokens(summarized_text)

        st.write("Summarized Text: ", summarized_text)

        # Generate the tweet thread using the summary
        tweets = generate_tweet_thread(summarized_text)
        st.write("Tweets: ", tweets)

        # Generate the long-form article using the summary
        article = generate_long_form_article(summarized_text, new_token_count)
        st.write("Article: ", article)

    except Exception as e:
        st.write("An error occurred: ", str(e))
