# ATR Model - Complete Building Workflow (Phase 2)

## 🎯 **Project Overview**
This document explains how to build the ATR (Audio Text Response) Model system from scratch, including the order of file creation, libraries used, and detailed explanations for beginners.

## 📋 **System Architecture**
```
ATR Model System
├── Backend (Python/FastAPI)
│   ├── main.py (FastAPI server)
│   ├── train.py (AI model training)
│   ├── summarizer.py (BART map-reduce summarization)
│   ├── retrieval.py (semantic search over chunks)
│   ├── chunking.py (text → indexed chunks)
│   ├── embeddings.py (sentence-transformers embeddings)
│   ├── generative_qa.py (answer generation with context)
│   ├── supabase_utils.py (DB I/O helpers)
│   └── state.py (state management)
├── Frontend (HTML/CSS/JavaScript)
│   ├── index.html (user interface)
│   ├── styles.css (styling)
│   └── script.js (client logic)
├── AI Models
│   ├── Whisper (Speech-to-Text)
│   ├── deepset/roberta-base-squad2 (Question Answering)
│   ├── facebook/bart-large-cnn (Summarization)
│   └── Piper TTS (Text-to-Speech)
└── Data Storage
    ├── uploads/ (audio files)
    ├── training_data/ (transcripts + configs)
    ├── piper/ (TTS runtime + voices)
    └── Supabase (Postgres): transcripts, transcript_chunks, summaries
```

## 🚀 **Step-by-Step Building Process**

### Quick start (after cloning)

1. Clone and enter the project directory:
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

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set environment variables for Supabase and behavior flags:
   ```powershell
   $env:SUPABASE_URL="https://<your-project-ref>.supabase.co"
   $env:SUPABASE_KEY="<your-service-role-or-anon-key>"
   $env:AUTO_SUMMARY_ON_TRAIN=0  # 1 to enable summary during training
   ```

5. (Optional) Place Piper TTS runtime and voices under `piper/`:
   - `piper.exe`, `onnxruntime.dll`, voice `.onnx`, `espeak-ng-data/`
   - These are ignored by git; keep them local.

6. Run services:
   - Backend (terminal 1):
     ```bash
     .venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
     ```
   - Frontend (terminal 2):
     ```bash
     python -m http.server 5500
     ```

7. Open UI and test:
   - UI: `http://127.0.0.1:5500`
   - API: `http://127.0.0.1:8000`

### **Phase 1: Project Setup**

#### **1.1 Create Project Structure**
```bash
mkdir ATR-Model
cd ATR-Model
mkdir backend
mkdir uploads
mkdir training_data
mkdir training_data/whisper_data
mkdir training_data/tts_data
mkdir piper
```

#### **1.2 Create Virtual Environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

#### **1.3 Create requirements.txt**
```txt
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

#### **1.4 Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Phase 2: Backend Development (Phase 2 upgrades highlighted)**

#### **2.1 Create state.py (First Python File)**
**Purpose**: Manage training state and progress
**Why First**: Other files depend on this

```python
from dataclasses import dataclass

@dataclass
class TrainingState:
    last_uploaded: str = ""
    is_training: bool = False
    trained: bool = False
    progress: int = 0
    artifact: str = ""
    current_stage: str = ""
```

**Libraries Used**: `dataclasses` (built-in Python)
**Explanation**: Simple data structure to track system state

#### **2.2 Create train.py (Core AI Logic)**
**Purpose**: Handle AI model training and inference
**Why Second**: Core functionality that main.py will use

```python
import os
import json
import asyncio
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import whisper
import torch

# Hugging Face Transformers
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class ATRTrainer:
    def __init__(self, training_data_dir: str = "training_data"):
        # Initialize all components
        self.whisper_model = None
        self.qa_model = None
        self.tts_model = None
        self.training_state = None
        self.content_text = ""
        
    async def train_all_models(self):
        # Main training orchestration
        pass
        
    async def train_whisper(self):
        # Speech-to-text processing
        pass
        
    async def setup_qa_model(self):
        # Question-answering setup
        pass
        
    async def generate_response(self, question: str):
        # Generate answers to questions
        pass
```

**Libraries Used**:
- `whisper`: OpenAI's speech-to-text model
- `transformers`: Hugging Face pipelines (QA: deepset/roberta-base-squad2; Sum: BART in summarizer.py)
- `sentence-transformers`: Embeddings for retrieval
 - `torch`: PyTorch for AI computations
- `asyncio`: Asynchronous programming
- `pathlib`: File path handling

**Key Functions**:
- `train_whisper()`: Converts audio to text
- `setup_qa_model()`: Loads question-answering model
- `generate_response()`: Answers questions from content

#### **2.3 Create main.py (FastAPI Server)**
**Purpose**: Web API server and main application
**Why Third**: Uses both state.py and train.py

```python
import os
import io
import base64
import asyncio
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .state import TrainingState

# Initialize FastAPI app
app = FastAPI(title="ATR Model API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
training_state = TrainingState()
trainer = None

@app.on_event("startup")
async def startup_event():
    # Load AI models at startup
    pass

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Handle file uploads
    pass

@app.post("/train-real")
async def train_real():
    # Start training process
    pass

@app.post("/interact")
async def interact(text: str = Form(None), audio: UploadFile = File(None)):
    # Handle user questions
    pass

@app.get("/progress")
async def progress():
    # Return training progress
    pass
```

**Libraries Used**:
- `fastapi`: Web framework for APIs
- `uvicorn`: ASGI server
- `python-multipart`: File upload handling
- `CORS`: Cross-origin resource sharing

**Key Endpoints** (Phase 2):
- `POST /upload`: Upload audio files (validates types; resets old training artifacts for new file)
- `POST /train-real`: Start training (Whisper → transcripts; QA model setup). No auto-summary unless `AUTO_SUMMARY_ON_TRAIN=1`.
- `POST /interact`: Ask questions (text or audio; Whisper transcribe if audio provided; optional TTS reply)
- `GET /progress`: Check progress/state
- `POST /summarize-latest`: Returns cached summary if present; otherwise generates one comprehensive BART summary from transcript text. Idempotent upsert.
- `POST /multiqa`: Multi-question RAG using Supabase chunks + embeddings
- `GET /audio-files`: Lists transcripts (deduped by filename)

### **Phase 3: Frontend Development**

#### **3.1 Create index.html (User Interface)**
**Purpose**: Web interface for users
**Why After Backend**: Needs API endpoints to work

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ATR Model</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <!-- Upload Section -->
    <div class="tabpanel" data-panel="upload">
        <input type="file" id="audioFile" accept=".mp3,.wav,.m4a">
        <button id="uploadBtn">Upload</button>
        <div id="fileInfo">No file chosen</div>
    </div>
    
    <!-- Training Section -->
    <div class="tabpanel" data-panel="training">
        <button id="startRealTrainBtn">Train</button>
        <div id="trainStatus">Ready to train</div>
        <div class="progress-bar">
            <div id="trainProgressBar"></div>
        </div>
        <div id="trainProgressText">0%</div>
    </div>
    
    <!-- Interaction Section -->
    <div class="tabpanel" data-panel="interact">
        <input type="text" id="textPrompt" placeholder="Ask a question...">
        <button id="recStartBtn">Record</button>
        <button id="recStopBtn">Stop</button>
        <button id="submitInteractBtn">Ask Question</button>
        <div id="responseText"></div>
        <audio id="responseAudio" controls></audio>
    </div>
    
    <script src="script.js"></script>
</body>
</html>
```

**HTML Elements** (Phase 2 diffs):
- File input for audio uploads
- Progress bar for training
- Text input and voice recording
- Audio player for responses
- Summary section without length selector (single comprehensive summary only)

#### **3.2 Create styles.css (Styling)**
**Purpose**: Visual styling and layout
**Why After HTML**: Styles the HTML elements

```css
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f5f5f5;
}

.tabpanel {
    display: none;
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.tabpanel--active {
    display: block;
}

.progress-bar {
    width: 100%;
    height: 20px;
    background-color: #e0e0e0;
    border-radius: 10px;
    overflow: hidden;
}

#trainProgressBar {
    height: 100%;
    background-color: #4CAF50;
    width: 0%;
    transition: width 0.3s ease;
}

button {
    background-color: #007bff;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    margin: 5px;
}

button:hover {
    background-color: #0056b3;
}

button:disabled {
    background-color: #ccc;
    cursor: not-allowed;
}
```

**CSS Features**:
- Responsive design
- Progress bar animation
- Button hover effects
- Clean, modern layout

#### **3.3 Create script.js (Client Logic)**
**Purpose**: Handle user interactions and API calls
**Why Last**: Connects frontend to backend

```javascript
(function () {
    // DOM elements
    const fileInput = document.getElementById('audioFile');
    const uploadBtn = document.getElementById('uploadBtn');
    const trainBtn = document.getElementById('startRealTrainBtn');
    const progressBar = document.getElementById('trainProgressBar');
    const progressText = document.getElementById('trainProgressText');
    
    // Upload functionality
    uploadBtn.addEventListener('click', async function () {
        const file = fileInput.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('http://127.0.0.1:8000/upload', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                uploadBtn.textContent = 'Uploaded ✓';
                uploadBtn.style.backgroundColor = '#28a745';
            }
        } catch (error) {
            console.error('Upload failed:', error);
        }
    });
    
    // Training functionality
    trainBtn.addEventListener('click', async function () {
        try {
            const response = await fetch('http://127.0.0.1:8000/train-real', {
                method: 'POST'
            });
            
            if (response.ok) {
                pollTraining();
            }
        } catch (error) {
            console.error('Training failed:', error);
        }
    });
    
    // Progress polling
    function pollTraining() {
        const interval = setInterval(async () => {
            try {
                const response = await fetch('http://127.0.0.1:8000/progress');
                const data = await response.json();
                
                progressBar.style.width = data.progress + '%';
                progressText.textContent = data.progress + '%';
                
                if (!data.is_training) {
                    clearInterval(interval);
                }
            } catch (error) {
                console.error('Progress check failed:', error);
            }
        }, 300);
    }
})();
```

**JavaScript Features**:
- File upload handling
- Progress monitoring
- API communication
- Error handling

### **Phase 4: AI Models Integration**

#### **4.1 Whisper Integration (Speech-to-Text)**
```python
# In train.py
import whisper

def setup_whisper(self):
    self.whisper_model = whisper.load_model("small")
    
async def transcribe_audio(self, audio_path):
    result = self.whisper_model.transcribe(audio_path)
    return result["text"]
```

**Purpose**: Convert audio to text
**Model**: `whisper-small` (OpenAI)
**Input**: Audio files (MP3, WAV, M4A, WEBM)
**Output**: Transcribed text

#### **4.2 QA Integration (deepset/roberta-base-squad2)**
```python
# In train.py
from transformers import pipeline

def setup_qa_model(self):
    self.qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
    
async def generate_response(self, question: str):
    result = self.qa_model(
        question=question,
        context=self.content_text
    )
    return result["answer"]
```

**Purpose**: Answer questions from transcribed content
**Model**: `deepset/roberta-base-squad2`
#### **4.3 Summarization (facebook/bart-large-cnn)**
```python
# In summarizer.py (Phase 2)
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def _get_summarizer():
    # Loads BART with local cache under models/bart-large-cnn
    ...

def generate_summary(text: str) -> Optional[str]:
    # Map-reduce for long inputs; always returns one comprehensive paragraph
    # Idempotent usage from API: cached summaries returned unless force=true
    ...
```

**Purpose**: Produce a single comprehensive summary of the transcript
**Model**: `facebook/bart-large-cnn`
**Notes**: Avoids echoing prompts; retries re-summarize the draft (no instruction leakage)

**Input**: Question + Context text
**Output**: Answer

#### **4.3 Piper TTS Integration (Text-to-Speech)**
```python
# In main.py
import subprocess

def generate_audio_response(text: str):
    # Use Piper TTS to convert text to speech
    cmd = [
        "piper/piper.exe",
        "-m", "piper/en_US-kathleen-low.onnx",
        "-f", "output.wav"
    ]
    subprocess.run(cmd, input=text.encode())
    return "output.wav"
```

**Purpose**: Convert text responses to audio
**Model**: `en_US-kathleen-low.onnx`
**Input**: Text response
**Output**: Audio file

### **Phase 5: Data Flow Integration**

#### **5.1 Complete Workflow**
```
1. User uploads audio file
   ↓
2. File saved to uploads/
   ↓
3. Whisper transcribes audio to text
   ↓
4. Text saved to training_data/whisper_data/
   ↓
5. QA model (roberta-base-squad2) loads/refreshes content
   ↓
6. User asks question
   ↓
7. RAG over Supabase chunks + embeddings; QA answers with top context
   ↓
8. Piper TTS converts answer to audio
   ↓
9. Audio response sent to user
```

#### **5.2 File Dependencies**
```
state.py (no dependencies)
    ↓
train.py (depends on state.py)
    ↓
main.py (depends on state.py + train.py)
    ↓
index.html (depends on main.py API)
    ↓
styles.css (styles index.html)
    ↓
script.js (connects to main.py API)
```

## 🛠️ **Development Order**

### **Step 1: Backend Foundation**
1. Create `state.py` - Data structures
2. Create `train.py` - AI logic
3. Create `main.py` - API server
4. Test with simple endpoints

### **Step 2: Frontend Interface**
1. Create `index.html` - Basic structure
2. Create `styles.css` - Visual design
3. Create `script.js` - User interactions
4. Test frontend-backend connection

### **Step 3: AI Integration**
1. Integrate Whisper for speech-to-text
2. Integrate DistilBERT for question answering
3. Integrate Piper TTS for text-to-speech
4. Test complete workflow

### **Step 4: Testing & Refinement**
1. Test file uploads
2. Test training process
3. Test question answering
4. Fix bugs and improve UX

## 📚 **Key Learning Points**

### **For Beginners:**

#### **Python Concepts**:
- **Classes**: `ATRTrainer` class encapsulates AI functionality
- **Async/Await**: For handling multiple operations simultaneously
- **File Handling**: Reading/writing files and managing directories
- **Error Handling**: Try/except blocks for robust code

#### **Web Development**:
- **APIs**: RESTful endpoints for communication
- **CORS**: Cross-origin resource sharing for frontend-backend
- **File Uploads**: Handling multipart form data
- **Real-time Updates**: Polling for progress updates

#### **AI Integration**:
- **Model Loading**: Loading pre-trained models
- **Inference**: Using models to make predictions
- **Pipeline**: Connecting multiple AI models together
- **Context Management**: Managing data between models

### **Common Pitfalls to Avoid**:
1. **Import Order**: Import dependencies before using them
2. **Async Functions**: Use `await` with async functions
3. **File Paths**: Use absolute paths for reliability
4. **Error Handling**: Always handle potential errors
5. **Model Loading**: Load models once, reuse many times

## 🎯 **Final System Features**

### **What the System Does**:
1. **Upload Audio**: Users can upload MP3/WAV/M4A files
2. **Transcribe Speech**: Whisper converts audio to text
3. **Train AI**: QA model prepared; transcripts, chunks, embeddings stored
4. **Answer Questions**: Multi-question RAG with QA over top-k chunks
5. **Summaries**: Single comprehensive BART summary (cached in Supabase)
6. **Audio Responses**: Piper TTS converts answers to speech

### **Technical Stack**:
- **Backend**: Python + FastAPI + Uvicorn
- **AI Models**: Whisper + Roberta QA + BART Summarizer + Piper TTS
- **Frontend**: HTML + CSS + JavaScript
- **Data**: Supabase (transcripts, chunks, summaries) + local audio/transcripts

### ⚙️ Phase 2 Operational Notes
- Backend default port: 8000. Frontend uses a centralized `API_BASE`.
- Summarization during training is disabled by default; enable with `AUTO_SUMMARY_ON_TRAIN=1`.
- Supabase is used for persistence. Provide `SUPABASE_URL` and `SUPABASE_KEY` in environment.
- Idempotent writes:
  - Transcripts upserted by filename
  - Chunks inserted only when missing; stale chunks cleaned before insert
  - Summaries upserted (update if exists, else insert)
- Frontend robustness:
  - Uploads use `fetchWithTimeout` and retry logic
  - Summary length selector removed; always a single comprehensive summary

### CORS
- Allowed origins: `http://127.0.0.1:5500`, `http://localhost:5500`
- Credentials disabled; wildcard headers and methods allowed

### UI Split
- `index.html`: Upload & Training (no sidebar)
- `interact.html`: Interaction (Q&A) and Summary (sidebar + chat UI)

### Training Audio Formats
- Training scans `training_data/whisper_data` for `.wav`, `.mp3`, `.m4a`, `.webm`
- Prefer shorter filenames; ffmpeg may error on very long names with spaces

This workflow reflects Phase 2 improvements: robust persistence, faster repeated runs via caching/idempotency, a better summarization pipeline, and a unified frontend configuration.
