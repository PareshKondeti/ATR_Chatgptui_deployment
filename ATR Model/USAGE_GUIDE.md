# ATR Model - Usage Guide

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Windows 10/11 (tested on Windows)
- 4GB+ RAM recommended
- Audio input/output capabilities

### Installation
1. Clone the repository
2. Create virtual environment: `python -m venv .venv`
3. Activate virtual environment: `.venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Download Piper TTS files to `piper/` directory
6. (Optional) Configure Supabase: set environment variables `SUPABASE_URL` and `SUPABASE_KEY`

## 🎯 Basic Usage

### 1. Starting the System (Phase 2)

#### Start Backend Server
```bash
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8010 --reload
```

#### Start Frontend Server
```bash
python -m http.server 5500
```

#### Access the Interface
- Open browser to `http://localhost:5500` and load `index.html` (or open `index.html` directly)
- Backend API available at `http://127.0.0.1:8010`

### 2. Upload and Train

#### Step 1: Upload Audio File
- Click "Select audio file" button
- Choose MP3, WAV, or M4A file
- Click "Upload" button
- Wait for upload confirmation

#### Step 2: Train the Model
- Click "Train" button
- Monitor progress in the console
- Wait for "Training completed successfully!" message
  - Note: Summarization during training is disabled by default. Generate summaries later via the Summary section or enable env `AUTO_SUMMARY_ON_TRAIN=1` before starting the server.

#### Step 3: Ask Questions
- Type your question in the text box, OR
- Click microphone to record voice question
- Click "Ask Question" button
- Listen to the audio response

## 🔧 Advanced Usage

### API Endpoints (Phase 2)

#### Upload File
```bash
curl -X POST "http://127.0.0.1:8010/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio_file.mp3"
```

#### Train Model
```bash
curl -X POST "http://127.0.0.1:8010/train-real"
```

#### Ask Question (Text)
```bash
curl -X POST "http://127.0.0.1:8010/interact" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "text=What is Vijayawada famous for?"
```

#### Ask Question (Audio)
```bash
curl -X POST "http://127.0.0.1:8010/interact" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@your_question.wav"
```

#### Check Progress
```bash
curl "http://127.0.0.1:8010/progress"
```

#### Generate/Fetch Summary (comprehensive paragraph)
```bash
curl -X POST "http://127.0.0.1:8010/summarize-latest" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "transcript_id=<optional-uuid>"
```
Returns cached summary if available; generates otherwise. Idempotent upsert.

#### Multi-question RAG
```bash
curl -X POST "http://127.0.0.1:8010/multiqa" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "prompt=What is the capital? What is the currency?&transcript_id=<optional-uuid>"
```

#### List Audio Files (transcripts)
```bash
curl "http://127.0.0.1:8010/audio-files"
```

### Programmatic Usage

#### Python Example
```python
import requests

# Upload file
with open('audio.mp3', 'rb') as f:
    response = requests.post('http://127.0.0.1:8010/upload', files={'file': f})
    print(response.json())

# Train model
response = requests.post('http://127.0.0.1:8010/train-real')
print(response.json())

# Ask question
response = requests.post('http://127.0.0.1:8010/interact', 
                        data={'text': 'What is Vijayawada famous for?'})
print(response.json())
```

## 📊 Monitoring and Debugging

### Check System Status
```bash
curl "http://127.0.0.1:8010/progress"
```

Expected response:
```json
{
  "is_training": false,
  "progress": 0,
  "trained": true,
  "artifact": "qa_system_ready"
}
```

### Server Logs
Monitor the backend server console for:
- Model loading status
- Training progress
- Error messages
- API request logs

### Common Status Messages
- `"Whisper model pre-loaded successfully!"` - Whisper ready
- `"QA model loaded successfully"` - QA model ready
- `"Training completed successfully!"` - Training done
- `"Training: skipping auto-summary (AUTO_SUMMARY_ON_TRAIN disabled)"` - No auto-summary during training

## 🎯 Best Practices

### Audio Quality
- Use clear, high-quality audio files
- Avoid background noise
- Speak clearly and at normal pace
- Use supported formats (MP3, WAV, M4A)

### Question Asking
- Ask specific, clear questions
- Use complete sentences
- Avoid ambiguous questions
- Be patient with processing time

### Performance
- Close other applications to free RAM
- Use shorter audio files for faster processing
- Restart server only when necessary
- Monitor memory usage

## 🔍 Troubleshooting

### Common Issues

#### Upload Fails
- Check file format (MP3, WAV, M4A only)
- Ensure file is not corrupted
- Check available disk space
- Verify file permissions

#### Training Fails
- Check server logs for error messages
- Ensure sufficient RAM available
- Verify internet connection for model downloads
- Check audio file quality

#### No Response to Questions
- Verify training completed successfully
- Check if content was transcribed properly
- Try different question phrasings
- Check server logs for errors

#### TTS Issues
- Verify Piper files are in correct location
- Check file permissions for Piper executable
- Ensure audio output is working
- Check server logs for TTS errors

### Debug Steps

1. **Check Server Status**
   ```bash
   curl "http://127.0.0.1:8000/progress"
   ```

2. **Check Server Logs**
   - Look for error messages in console
   - Check for model loading issues
   - Verify API endpoint responses

3. **Test Individual Components**
   - Test file upload separately
   - Test training separately
   - Test question asking separately

4. **Restart System**
   - Stop both servers
   - Clear any cached data
   - Restart servers
   - Try again

## 📈 Performance Tips

### Memory Management
- Close other applications
- Use shorter audio files
- Monitor RAM usage
- Restart server if memory issues

### Processing Speed
- Use clear audio files
- Ask specific questions
- Avoid very long audio files
- Use SSD storage if available

### Accuracy Improvement
- Use high-quality audio
- Speak clearly
- Ask specific questions
- Use complete sentences

## 🚀 Advanced Features

### Batch Processing
- Upload multiple files sequentially
- Each new upload clears training_data (Phase 2 upload flow)
- Train after each upload
- Ask questions per transcript using the dropdown in Summary & Q&A

### Custom Models
- QA model is set in `train.py` (roberta-base-squad2)
- Summarizer model is set in `summarizer.py` (facebook/bart-large-cnn)
- Adjust summarization settings in `summarizer.py` if needed
- TTS voice is controlled by Piper files in `piper/`

### Integration
- Use API endpoints in your applications
- Integrate with other systems
- Build custom frontends
- Extend functionality

## 📚 Examples

### Example 1: Educational Content
1. Upload lecture audio
2. Train model
3. Ask: "What are the main topics covered?"
4. Ask: "Explain the key concepts"
5. Ask: "What examples were given?"

### Example 2: Meeting Notes
1. Upload meeting recording
2. Train model
3. Ask: "What decisions were made?"
4. Ask: "Who attended the meeting?"
5. Ask: "What are the action items?"

### Example 3: Interview Analysis
1. Upload interview audio
2. Train model
3. Ask: "What questions were asked?"
4. Ask: "What were the main responses?"
5. Ask: "What insights were shared?"

---

This usage guide reflects Phase 2: new port (8010), robust upload with timeouts, idempotent Supabase writes, cached summaries, single comprehensive summaries, and RAG-based multi-question answering.
