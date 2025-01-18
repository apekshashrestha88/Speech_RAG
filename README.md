Functionalities:
1. Real-time voice recording
2. STT using Vosk
3. Pdf processing and text extraction
4. AI qna using LLaMA3.2
5. TTS using Coqui
6. Multi-threading
7. Only context-based answering

The Vosk model used is:"vosk-model-en-us-0.22-lgraph"(only uses cpu)
Streamlit UI, completely offline 

Vosk converts audio into phonemes(speech sounds)  --> matches to words(using statistical model)
