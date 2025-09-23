# ATR Model - Audio Text Response System

A comprehensive AI-powered system that processes audio content and provides intelligent responses using Hugging Face Transformers.

## 🚀 Features

- **Audio Processing**: Upload and transcribe audio files (MP3, WAV, M4A, WEBM)
- **Intelligent Q&A**: Ask questions about your audio content using Hugging Face QA models
- **Text-to-Speech**: Convert responses back to audio using Piper TTS
- **Real-time Interaction**: Voice-based question and answer system in a Chat-style UI
- **Model Caching**: Efficient model loading and reuse for better performance
 - **Per-answer actions**: Play TTS, download WAV/TXT for the combined answer in Q&A

## 🏗️ Architecture

### Core Components

1. **Whisper Integration**: Speech-to-text transcription using OpenAI Whisper
2. **Hugging Face QA**: Question-answering using deepset/roberta-base-squad2
3. **Summarization**: Single comprehensive summary using facebook/bart-large-cnn
4. **Piper TTS**: Text-to-speech conversion for audio responses
5. **FastAPI Backend**: RESTful API for all operations
6. **Web Frontend**: Simple HTML/JS interface for user interaction

### Models Used

- **Whisper**: `whisper-1` (OpenAI) - Audio transcription
- **QA Model**: `deepset/roberta-base-squad2` - Question answering
- **Summarizer**: `facebook/bart-large-cnn` - Transcript summarization
- **TTS Model**: `en_US-kathleen-low.onnx` (Piper) - Text-to-speech

## 📋 Requirements

### System Requirements
- Python 3.8+
- Windows 10/11 (tested on Windows)
- 4GB+ RAM recommended
- Audio input/output capabilities

### Python Dependencies
```
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
transformers==4.56.1
torch==2.2.0
whisper==1.1.10
numpy==1.24.3
pathlib
sentence-transformers==2.2.2
supabase==2.6.0
psycopg-binary==3.2.1
```

## 🛠️ Installation

### Quick start (after cloning)

1. Clone the repo and enter the folder:
   ```bash
   git clone https://github.com/PareshKondeti/ATR_Phase2_Improved.git
   cd ATR_Phase2_Improved/"ATR Model"
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # WSL/macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables (Supabase; optional training behavior):
   - Windows PowerShell (current session):
     ```powershell
     $env:SUPABASE_URL="https://<your-project-ref>.supabase.co"
     $env:SUPABASE_KEY="<your-service-role-or-anon-key>"
     $env:AUTO_SUMMARY_ON_TRAIN=0  # 1 to enable auto-summary during /train-real
     ```
   - Or create a `.env` file (if you prefer) and load it before running.

5. (Optional) Piper TTS assets:
   - Place `piper.exe`, voice model `.onnx` and `espeak-ng-data/` under `piper/` if you need TTS.
   - The repo’s `.gitignore` excludes large binaries; keep them local.

6. Start the backend:
   ```bash
   .venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

7. In a second terminal, start the frontend (static):
   ```bash
   python -m http.server 5500
   ```

8. Open the app:
   - UI: `http://127.0.0.1:5500`
   - API: `http://127.0.0.1:8000`

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PareshKondeti/ATR_Phase2_Improved.git
   cd ATR_Phase2_Improved/"ATR Model"
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Piper TTS**:
   - Download Piper executable and model files
   - Place in `piper/` directory
   - Ensure `piper.exe` and `en_US-kathleen-low.onnx` are present

## 🚀 Usage

### Starting the System (Phase 2)

1. **Start Backend Server**:
   ```bash
   .venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Start Frontend Server**:
   ```bash
   python -m http.server 5500
   ```

3. **Access the Interface**:
   - Open browser to `http://localhost:5500` or open `index.html` directly
   - Backend API available at `http://127.0.0.1:8000`

### Workflow

1. **Upload Audio**: Select and upload audio files (MP3, WAV, M4A, WEBM)
2. **Train Model**: Process audio with Whisper and setup QA system (roberta-base-squad2)
3. **Ask Questions**: Use voice or text to ask questions about the content
4. **Summarize**: Generate a single comprehensive summary (BART) and cache in Supabase
5. **Get Responses**: Receive audio responses via TTS

## 📁 Project Structure

```
ATR-Model/
├── backend/
│   ├── main.py          # FastAPI application
│   ├── train.py         # Model training and inference
│   ├── summarizer.py    # Summarization (BART map-reduce)
│   ├── retrieval.py     # Semantic search over chunks
│   ├── chunking.py      # Chunking into indexed segments
│   ├── embeddings.py    # Sentence-transformers embeddings
│   ├── generative_qa.py # Contextual answer generation
│   ├── supabase_utils.py# Supabase persistence (transcripts/chunks/summaries)
│   └── state.py         # Training state management
├── training_data/
│   ├── whisper_data/    # Whisper transcripts
│   └── tts_data/        # TTS configuration
├── uploads/             # Uploaded audio files
├── piper/               # Piper TTS files
├── index.html           # Upload & Training page
├── interact.html        # Interaction (Q&A) and Summary page
├── styles.css           # Frontend styling
├── script.js            # Frontend JavaScript
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 🔧 API Endpoints (Phase 2)

### Upload/Train
- **POST** `/upload` - Upload audio files
- **POST** `/train-real` - Train models with uploaded content (no auto-summary unless env enabled)

### Interaction
- **POST** `/interact` - Ask questions (text or audio)
- **GET** `/progress` - Check training progress
- **POST** `/multiqa` - Ask multiple questions via RAG over chunks
- **POST** `/summarize-latest` - Generate or return cached comprehensive summary
- **GET** `/audio-files` - List transcripts (deduped)

### Management
- **POST** `/fix-training-state` - Manual training state fix

## 🎯 Key Features

### Model Caching
- Models are loaded once at startup
- No reloading during training or inference
- Improved performance and reduced memory usage

### Automatic Data Management
- Clears previous training data on new uploads
- Ensures fresh training for each new file
- Prevents data conflicts
 - Idempotent Supabase writes (no duplicate summaries/chunks)

### Clean Answer Generation
- Direct QA model responses
- No hardcoded sentence templates
- Natural, authentic answers
 - Retrieval-augmented generation over stored chunks/embeddings

### Error Handling
- Comprehensive error handling throughout
- Graceful fallbacks for failed operations
- Clear error messages for debugging

### CORS and Frontend Origins
- Backend allows requests from `http://127.0.0.1:5500` and `http://localhost:5500`.
- Credentials are disabled; wildcard headers and methods allowed.

### Notes on Training
- Training scans `training_data/whisper_data` for audio: `.wav`, `.mp3`, `.m4a`, `.webm`.
- If uploads go to `uploads/`, they are copied into `training_data/whisper_data` during training.
- For long filenames with spaces, ffmpeg can fail. Prefer shorter filenames or convert to `.wav`.

## 🔍 Troubleshooting

### Common Issues

1. **Upload Conflicts**:
   - System automatically handles file conflicts
   - Old files are removed when uploading new ones

2. **Model Loading Issues**:
   - Ensure all dependencies are installed
   - Check internet connection for model downloads
   - Verify sufficient RAM available

3. **TTS Issues**:
   - Ensure Piper files are in correct location
   - Check file permissions for Piper executable

### Debug Information

- Check server logs for detailed error messages
- Use `/progress` endpoint to verify system status
- Monitor console output for model loading status

## 🚀 Performance Tips

1. **Memory Management**: Close other applications to free RAM
2. **Audio Quality**: Use clear audio files for better transcription
3. **Question Clarity**: Ask specific questions for better answers
4. **Model Caching**: Restart server only when necessary

## 📈 Future Enhancements

- Support for more audio formats
- Additional language models
- Improved TTS voices
- Batch processing capabilities
- Cloud deployment options

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check troubleshooting section
2. Review server logs
3. Create GitHub issue with detailed information

---

**ATR Model** - Making audio content intelligent and interactive! 🎵🤖