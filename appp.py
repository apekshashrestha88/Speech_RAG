import os
import io
import wave
import json
import streamlit as st
import sounddevice as sd
import numpy as np
import tempfile
import ollama 
from vosk import Model, KaldiRecognizer
from TTS.api import TTS
from pypdf import PdfReader 
from concurrent.futures import ThreadPoolExecutor


#Real-time audio recording:
def record_audio(duration=10, samplerate=16000):
    """Records audio and keeps it in memory instead of writing to a file."""
    st.write(f"🎤 Recording for {duration} seconds...")
    
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait() 
    
    # Store audio in memory (instead of creating a file)
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    st.success("✅ Recording complete!")
    wav_buffer.seek(0) 
    return wav_buffer  # Returns in-memory audio


#Stt using Vosk:
MODEL_PATH = "vosk-model-en-us-0.22-lgraph" 
if not os.path.exists(MODEL_PATH):
    st.error("❌ Please download a Vosk model from https://alphacephei.com/vosk/models and extract it.")
else:
    model = Model(MODEL_PATH)

def speech_to_text(audio_buffer):
    """Converts speech to text using Vosk, handling partial responses."""
    wf = wave.open(audio_buffer, "rb")
    recognizer = KaldiRecognizer(model, wf.getframerate())
    
    results = []
    
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result_json = json.loads(recognizer.Result())
            results.append(result_json.get("text", ""))

    return " ".join(results).strip()

#pdf processing:
def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

#llama3.2 model:
def get_ai_response(prompt, context=""):
    """Generates an AI response strictly based on the uploaded PDF content."""
    if not context or context.strip() == "":
        return "⚠️ Please upload a valid PDF before asking questions."

    query = (
        f"You are a document-specific assistant. You must only answer based on the provided document text.\n\n"
        f"Document Context:\n{context}\n\n"
        f"User Question: {prompt}\n\n"
        f"If the answer is not in the document, reply with 'I cannot find this information in the uploaded document.'"
    )

    response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": query}],options={"gpu":True})
    return response['message']['content']


#Coqui TTS(with auto-play):
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
tts = TTS(TTS_MODEL).to("cuda")  # Load TTS on CPU

def text_to_speech(text):
    """Converts text to speech using Coqui TTS and returns the audio file path."""
    tts_file = "output_tts.wav"  # Use a fixed filename

    try:
        tts.tts_to_file(text=text, file_path=tts_file)  # Generate speech
        st.session_state.tts_audio_path = tts_file  # Store in session state
        return tts_file
    except Exception as e:
        st.error(f"❌ Error generating speech: {e}")
        return None  # Return None if speech fails

#Streamlit UI:
st.title("📄 Offline AI Chatbot with PDF, Voice & LLaMA 3.2")
st.markdown("🚀 **Upload PDFs, ask questions, use voice commands, and get AI-generated responses—all offline!**")

#Pdf uploading section:
# Sidebar PDF upload
st.sidebar.header("📂 Upload a PDF")
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_pdf:
    st.sidebar.success("✅ PDF Uploaded!")
    pdf_text = extract_text_from_pdf(uploaded_pdf)
    
    # Store extracted text in session state properly
    if pdf_text.strip():
        st.session_state.pdf_text = pdf_text
    else:
        st.session_state.pdf_text = None  # Ensure empty PDFs are handled

    # Show extracted text preview
    st.sidebar.text_area("📜 Extracted Text:", pdf_text[:500] + "...", height=150)

#Audio record:
st.sidebar.header("🎤 Voice Input")
record_btn = st.sidebar.button("🎙️ Record Voice")

if record_btn:
    audio_path = record_audio(duration=5)
    transcribed_text = speech_to_text(audio_path)
    st.sidebar.success(f"📝 Transcribed Text: {transcribed_text}")
    st.session_state.transcribed_text = transcribed_text  

#Text input:
user_query = st.text_area("🔍 Ask a question:", st.session_state.get("transcribed_text", ""))

#The response section:
if st.button("💬 Get AI Answer"):
    with st.spinner("Generating AI response..."):
        context = st.session_state.get("pdf_text", "")

        # Ensure the AI only answers if a PDF is uploaded
        if not context or context.strip() == "":
            st.error("⚠️ Please upload a PDF before asking questions.")
        else:
            with ThreadPoolExecutor() as executor:
                ai_future = executor.submit(get_ai_response, user_query, context)
                response = ai_future.result()

            st.success(f"🤖 AI Response:\n{response}")
            st.session_state.last_response = response

            # Text-to-Speech
            with ThreadPoolExecutor() as executor:
                tts_future = executor.submit(text_to_speech, response)
                tts_audio_path = tts_future.result()

            if tts_audio_path and os.path.exists(tts_audio_path):
                st.audio(tts_audio_path)
            else:
                st.error("❌ Failed to generate speech.")
