import os
import io
import base64
import asyncio
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .state import TrainingState
from .supabase_utils import (
    get_supabase,
    insert_transcript,
    upsert_transcript,
    insert_summary,
    insert_chunks,
    delete_chunks_for_transcript,
    fetch_latest_transcript_id,
    fetch_chunks_for_transcript,
    fetch_all_chunks,
    list_transcripts,
    fetch_summary_for_transcript,
    fetch_transcript_text,
)
from .chunking import build_indexed_chunks
from .embeddings import embed_texts
from .summarizer import generate_summary
from .retrieval import semantic_search
from .generative_qa import answer_with_context, paraphrase_succinct

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)

# Optional .env loader (no external deps). Loads ROOT_DIR/.env if present.
def _load_dotenv_if_present():
    env_path = os.path.join(ROOT_DIR, ".env")
    try:
        if os.path.isfile(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    key = key.strip()
                    # Strip optional surrounding quotes
                    value = value.strip().strip('"').strip("'")
                    # Do not overwrite existing env from the process
                    if key and (os.getenv(key) is None):
                        os.environ[key] = value
    except Exception:
        # Silent fail; env variables remain as-is
        pass

_load_dotenv_if_present()
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")
RESPONSE_DIR = os.path.join(ROOT_DIR, "responses")
PIPER_DIR = os.path.join(ROOT_DIR, "piper")
PIPER_EXE = os.path.join(PIPER_DIR, "piper.exe")
# Update the model filename below if different
PIPER_MODEL = os.path.join(PIPER_DIR, "en_US-kathleen-low.onnx")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESPONSE_DIR, exist_ok=True)

app = FastAPI(title="ATR Backend", version="0.1.0")

# Global DEBUG logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("atr")
logger.setLevel(logging.DEBUG)
try:
    logging.getLogger("atr.summarization").setLevel(logging.DEBUG)
except Exception:
    pass

# CORS configuration
_raw_allowed = (os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500") or "").strip()
_list_allowed = [o.strip() for o in _raw_allowed.split(",") if o.strip()]

# Special-case wildcard: FastAPI expects ["*"] to allow any origin
if len(_list_allowed) == 1 and _list_allowed[0] == "*":
    _allow_origins = ["*"]
else:
    _allow_origins = _list_allowed

# Optional credentials toggle via env (default off for simplicity)
_allow_credentials = (os.getenv("ALLOW_CREDENTIALS", "0") or "0").strip().lower() in ("1", "true", "yes", "on")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

training_state = TrainingState()
# Silence favicon 404s from browsers
@app.get("/favicon.ico")
async def _favicon():
    return JSONResponse(status_code=204, content=None)


# Startup event to pre-load models
@app.on_event("startup")
async def startup_event():
    print("Starting ATR Model server...")
    # Allow memory-constrained environments to skip heavy startup loading
    skip_startup = (os.getenv("SKIP_STARTUP_LOAD", "0") or "0").strip().lower() in ("1", "true", "yes", "on")
    if skip_startup:
        print("SKIP_STARTUP_LOAD=1 detected: skipping Whisper/QA preload for low-memory deploys")
        print("Server startup complete!")
        return

    await _preload_whisper_model()
    
    # Try to load QA models if they exist
    try:
        from .train import trainer, TRANSFORMERS_AVAILABLE
        if TRANSFORMERS_AVAILABLE:
            print("Loading QA model at startup...")
            await trainer.setup_qa_model()
            print("QA models loaded successfully!")
        
        # Set the pre-loaded Whisper model in trainer
        if _whisper_model is not None:
            trainer.set_whisper_model(_whisper_model)
            print("Whisper model set in trainer")
    except Exception as e:
        print(f"Failed to load QA models: {e}")
    
    print("Server startup complete!")

# Lazy-loaded models (initialized on first use)
_whisper_model = None
_tts_model = None
_tts_vocoder = None

# Pre-load Whisper model at startup
async def _preload_whisper_model():
    global _whisper_model
    try:
        print("Pre-loading Whisper model at startup...")
        import whisper
        _whisper_model = whisper.load_model("small")
        print("Whisper model pre-loaded successfully!")
    except Exception as e:
        print(f"Failed to pre-load Whisper model: {e}")
        _whisper_model = None


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        filename = file.filename
        if not filename:
            return JSONResponse(status_code=400, content={"error": "Missing filename"})

        allowed = {".mp3", ".wav", ".m4a"}
        _, ext = os.path.splitext(filename.lower())
        if ext not in allowed:
            return JSONResponse(status_code=400, content={"error": f"Unsupported file type: {ext}"})

        dest_path = os.path.join(UPLOAD_DIR, filename)
        
        # FIRST: Check if file has already been trained BEFORE doing ANYTHING else
        whisper_dir = os.path.join(ROOT_DIR, "training_data", "whisper_data")
        is_already_trained = False
        
        if os.path.exists(whisper_dir):
            for file in os.listdir(whisper_dir):
                if file.endswith('.txt'):
                    # Check if this transcript corresponds to our audio file
                    transcript_base = os.path.splitext(file)[0]
                    audio_base = os.path.splitext(filename)[0]
                    if transcript_base == audio_base:
                        is_already_trained = True
                        print(f"File '{filename}' has already been trained (found transcript: {file}). Skipping upload.")
                        break
        
        if is_already_trained:
            return {"status": "already_trained", "filename": filename, "message": f"Audio file '{filename}' has already been trained. You can ask questions about it directly."}
        
        # SECOND: Check if file already exists and handle it
        if os.path.exists(dest_path):
            # File exists but not trained, allow overwrite
            try:
                os.remove(dest_path)
                print(f"Removed existing file: {filename}")
            except Exception as e:
                print(f"Could not remove existing file: {e}")
                return JSONResponse(status_code=409, content={"error": f"Audio file '{filename}' already exists and could not be removed. Please try a different filename."})
        
        # THIRD: Only clear previous training data when uploading a genuinely new file
        print("Clearing previous training data for new file...")
        try:
            import shutil
            training_data_dir = os.path.join(os.path.dirname(__file__), "..", "training_data")
            
            # Clear whisper training data
            whisper_dir = os.path.join(training_data_dir, "whisper_data")
            if os.path.exists(whisper_dir):
                for file in os.listdir(whisper_dir):
                    file_path = os.path.join(whisper_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                print("Cleared whisper training data")
            
            # Clear TTS training data
            tts_dir = os.path.join(training_data_dir, "tts_data")
            if os.path.exists(tts_dir):
                for file in os.listdir(tts_dir):
                    file_path = os.path.join(tts_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                print("Cleared TTS training data")
            
            # Reset training state
            training_state.trained = False
            training_state.is_training = False
            training_state.progress = 0
            training_state.current_stage = ""
            print("Training state reset")
            
        except Exception as e:
            print(f"Error clearing training data: {e}")
        
        # Save new file
        try:
            # Ensure we have a valid file object
            if not hasattr(file, 'read') or not callable(getattr(file, 'read', None)):
                print(f"Invalid file object received. Type: {type(file)}")
                return JSONResponse(status_code=500, content={"error": "Invalid file object received"})
            
            with open(dest_path, "wb") as out:
                content = await file.read()
                if not content:
                    return JSONResponse(status_code=500, content={"error": "No content read from file"})
                out.write(content)
                print(f"Successfully saved {len(content)} bytes to {dest_path}")
        except Exception as e:
            print(f"Error saving file: {e}")
            print(f"File object type: {type(file)}")
            return JSONResponse(status_code=500, content={"error": f"Failed to save file: {str(e)}"})

        # Update training state
        training_state.last_uploaded = dest_path
        training_state.trained = False  # Reset training status when new file uploaded
        
        return {"status": "success", "filename": filename, "path": dest_path, "message": f"Audio file '{filename}' uploaded successfully. Ready for training."}
    
    except Exception as e:
        print(f"Upload error: {e}")
        return JSONResponse(status_code=500, content={"error": f"Upload failed: {str(e)}"})


@app.post("/train")
async def train():
    if training_state.is_training:
        return {"status": "already_training", "progress": training_state.progress}

    async def _simulate_training():
        training_state.is_training = True
        training_state.progress = 0
        training_state.trained = False
        # Simulate stages: prepare (0-20), train (20-90), finalize (90-100)
        for i in range(1, 101):
            await asyncio.sleep(0.06)
            training_state.progress = i
        # Mark an artifact path to show completion
        training_state.artifact = os.path.join(RESPONSE_DIR, 'checkpoint.mock')
        try:
            with open(training_state.artifact, 'w', encoding='utf-8') as f:
                f.write('mock checkpoint\n')
        except Exception:
            pass
        training_state.is_training = False
        training_state.trained = True

    asyncio.create_task(_simulate_training())
    return {"status": "started", "note": ("no_input" if not training_state.last_uploaded else "ok")}

@app.post("/train-real")
async def train_real():
    """
    Start real training process with actual AI model training
    """
    if training_state.is_training:
        return {"status": "already_training", "progress": training_state.progress}

    async def _real_training():
        training_state.is_training = True
        training_state.progress = 0
        training_state.trained = False
        
        try:
            # Import the trainer
            from .train import trainer
            
            # Set training state reference
            trainer.set_training_state(training_state)
            
            # Copy uploaded files to training data directory
            await _prepare_training_data()
            
            # Start real training
            success = await trainer.train_all_models()
            
            if success:
                training_state.trained = True
                training_state.artifact = "trained_models"
                print("Real training completed successfully!")
                print(f"Training state set to: trained={training_state.trained}")

                # After transcription, push to Supabase: transcript, chunks, embeddings, summary (idempotent)
                try:
                    sb = get_supabase()
                    if sb is not None:
                        # Load combined content
                        content_text = await trainer._load_all_content()
                        if content_text:
                            # Insert transcript row (latest filename if available)
                            latest_file = None
                            if training_state.last_uploaded:
                                latest_file = os.path.basename(training_state.last_uploaded)
                            transcript_id = upsert_transcript(sb, latest_file or "combined", content_text)
                            if transcript_id:
                                # Chunks: only (re)compute if none exist; otherwise keep existing to avoid duplicates
                                existing_chunks = fetch_chunks_for_transcript(sb, transcript_id)
                                if not existing_chunks:
                                    chunks = build_indexed_chunks(content_text)
                                    if chunks:
                                        embeds = embed_texts([c["text"] for c in chunks], kind="passage")
                                        for c, v in zip(chunks, embeds):
                                            c["embedding"] = v
                                        # Ensure no stale chunks before insert
                                        delete_chunks_for_transcript(sb, transcript_id)
                                        insert_chunks(sb, transcript_id, chunks)

                                # Summary: generate during training only if explicitly enabled
                                # Set AUTO_SUMMARY_ON_TRAIN=1 to enable; default is off to save time
                                auto_sum_flag = (os.getenv("AUTO_SUMMARY_ON_TRAIN", "0") or "").strip().lower() in ("1", "true", "yes")
                                if auto_sum_flag:
                                    cached_summary = fetch_summary_for_transcript(sb, transcript_id)
                                    if not cached_summary:
                                        summary = generate_summary(content_text)
                                        if summary:
                                            insert_summary(sb, transcript_id, summary)
                                else:
                                    logger.info("Training: skipping auto-summary (AUTO_SUMMARY_ON_TRAIN disabled)")
                    else:
                        print("Supabase not configured; skipping persistence")
                except Exception as e:
                    print(f"Post-train persistence failed: {e}")
            else:
                training_state.trained = False
                print("Real training failed!")
                
        except Exception as e:
            print(f"Real training error: {e}")
            training_state.trained = False
        finally:
            training_state.is_training = False

    asyncio.create_task(_real_training())
    return {"status": "real_training_started"}

async def _prepare_training_data():
    """
    Prepare training data by copying only the newly uploaded file to training directories
    """
    try:
        from .train import trainer
        import shutil
        
        # Ensure training directory exists
        trainer.whisper_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the most recently uploaded file (the new one)
        if training_state.last_uploaded:
            latest_file_path = training_state.last_uploaded
            latest_file_name = os.path.basename(latest_file_path)
            
            # Copy only the new file to training data
            dst_path = trainer.whisper_dir / latest_file_name
            
            # Only copy if not already exists in training data
            if not dst_path.exists():
                shutil.copy2(latest_file_path, dst_path)
                print(f"Copied new file: {latest_file_name} to training data")
            else:
                print(f"File {latest_file_name} already exists in training data")
        else:
            print("No recently uploaded file found")
                
    except Exception as e:
        print(f"Error preparing training data: {e}")

@app.get("/progress")
async def progress():
    return {
        "is_training": training_state.is_training,
        "progress": training_state.progress,
        "trained": training_state.trained,
        "artifact": getattr(training_state, 'artifact', ''),
        "current_stage": getattr(training_state, 'current_stage', ''),
    }

@app.post("/fix-training-state")
async def fix_training_state():
    """Manually fix training state if QA system is working"""
    try:
        from .train import trainer, TRANSFORMERS_AVAILABLE
        
        # Check if QA system is actually working
        if TRANSFORMERS_AVAILABLE and trainer.qa_model and trainer.content_text:
            training_state.trained = True
            training_state.artifact = "qa_system_ready"
            print("Training state manually set to True - QA system is working")
            return {"status": "success", "message": "Training state fixed", "trained": True}
        else:
            return {"status": "error", "message": "QA system not ready", "trained": False}
    except Exception as e:
        return {"status": "error", "message": str(e), "trained": False}


@app.post("/interact")
async def interact(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
):
    # If audio provided, transcribe locally with Whisper if available
    transcript = text or None
    if audio is not None and audio.filename:
        audio_bytes = await audio.read()
        # Store received audio locally
        save_path = os.path.join(UPLOAD_DIR, f"input_{audio.filename}")
        with open(save_path, "wb") as f:
            f.write(audio_bytes)
        if transcript is None:
            transcript = await _transcribe_whisper_safe(save_path)
            if transcript is None:
                transcript = f"[Audio received: {audio.filename}]"

    if transcript is None:
        return JSONResponse(status_code=400, content={"error": "Provide text or audio"})

    # Use trained model if available, otherwise fall back to rule-based
    response_text = await _generate_response_with_trained_model(transcript)

    # Try TTS; if it fails, fall back to silence
    wav_bytes = await _tts_synthesize_safe(response_text)
    if wav_bytes is None:
        return JSONResponse(status_code=500, content={"error": "TTS synthesis failed (Coqui)"})
    b64_audio = base64.b64encode(wav_bytes).decode('utf-8')

    return {"text": response_text, "audio_b64_wav": b64_audio}


@app.post("/summarize-latest")
async def summarize_latest(transcript_id: Optional[str] = Form(None), force: Optional[str] = Form(None), length: Optional[str] = Form(None)):
    try:
        sb = get_supabase()
        if sb is None:
            return JSONResponse(status_code=400, content={"error": "Supabase not configured"})
        tid = transcript_id or fetch_latest_transcript_id(sb)
        if not tid:
            return JSONResponse(status_code=404, content={"error": "No transcript found"})
        # If summary already exists in Supabase, return cached unless force regenerate requested
        cached = fetch_summary_for_transcript(sb, tid)
        force_flag = (force or "").lower() in ("1", "true", "yes")
        logger.info("/summarize-latest: transcript_id=%s force=%s cached_available=%s", tid, force_flag, bool(cached))
        if cached and not force_flag:
            return {"transcript_id": tid, "summary": cached, "cached": True}

        # Summarize from the canonical transcript text instead of recomputing from chunks
        text = fetch_transcript_text(sb, tid) or ""
        if not text.strip():
            # Fallback to chunks if transcript text missing
            chunks = fetch_chunks_for_transcript(sb, tid)
            logger.info("/summarize-latest: fetched %d chunks for transcript_id=%s", len(chunks), tid)
            text = " \n".join([c.get("text", "") for c in chunks])
        summary = generate_summary(text)
        if not summary:
            logger.warning("/summarize-latest: summarization returned None for transcript_id=%s", tid)
            return JSONResponse(status_code=500, content={"error": "Summarization failed"})
        # Upsert summary (update existing or insert new)
        insert_summary(sb, tid, summary)
        return {"transcript_id": tid, "summary": summary, "cached": False, "regenerated": bool(force_flag)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


def _split_questions(prompt: str) -> List[str]:
    import re
    parts = re.split(r"[\n\r]+|\?|;|\.|!", prompt)
    qs = [p.strip() for p in parts if p and len(p.strip()) > 0]
    return qs


@app.post("/multiqa")
async def multiqa(
    prompt: str = Form(""),
    audio: Optional[UploadFile] = File(None),
    transcript_id: Optional[str] = Form(None),
):
    logger.info("/multiqa: received request transcript_id=%s has_audio=%s", transcript_id, bool(audio and audio.filename))
    # If audio provided and prompt empty, transcribe it as the prompt
    if (not prompt or not prompt.strip()) and audio is not None and audio.filename:
        try:
            audio_bytes = await audio.read()
            save_path = os.path.join(UPLOAD_DIR, f"multiqa_{audio.filename}")
            with open(save_path, "wb") as f:
                f.write(audio_bytes)
            prompt = await _transcribe_whisper_safe(save_path) or ""
            logger.info("/multiqa: transcribed audio to prompt length_chars=%d", len(prompt))
        except Exception:
            logger.exception("/multiqa: audio transcription failed")
            prompt = ""
    if not prompt.strip():
        return JSONResponse(status_code=400, content={"error": "Empty prompt"})
    try:
        sb = get_supabase()
        questions = _split_questions(prompt)
        logger.info("/multiqa: questions_count=%d", len(questions))

        # Primary path: Supabase RAG
        if sb is not None:
            # Restrict to a specific transcript if provided; else use all
            if transcript_id:
                chunks = fetch_chunks_for_transcript(sb, transcript_id)
                logger.info("/multiqa: fetched %d chunks for transcript_id=%s", len(chunks), transcript_id)
            else:
                chunks = fetch_all_chunks(sb)
                logger.info("/multiqa: fetched %d chunks across all transcripts", len(chunks))
            if chunks:
                # Per-question generation using retrieved context
                results = []
                for idx, q in enumerate(questions, start=1):
                    ranked = semantic_search(q, [dict(c) for c in chunks])
                    logger.debug("/multiqa: q%02d top5_scores=%s", idx, [round(r.get("score", 0.0), 3) for r in ranked[:5]])
                    top_ctx = [r.get("text", "") for r in ranked[:5]]
                    answer = answer_with_context(q, top_ctx)
                    answer = paraphrase_succinct(answer)
                    results.append({"q": q, "a": answer})
                logger.info("/multiqa: answered %d questions", len(results))
                return {"items": results}

        # Fallback path: use locally loaded content if Supabase not ready
        from .train import trainer, TRANSFORMERS_AVAILABLE
        local_text = trainer.content_text if TRANSFORMERS_AVAILABLE else ""
        if not local_text:
            # Try reading training data directly
            local_text = await _get_transcribed_content()
        if not local_text:
            return JSONResponse(status_code=400, content={"error": "No content available. Train or configure Supabase first."})

        # Chunk locally and answer
        tmp_chunks = build_indexed_chunks(local_text)
        for c in tmp_chunks:
            c["embedding"] = None  # will be generated in semantic_search if needed
        # Per-question fallback on local chunks
        results = []
        for idx, q in enumerate(questions, start=1):
            ranked = semantic_search(q, [dict(c) for c in tmp_chunks])
            top_ctx = [r.get("text", "") for r in ranked[:5]]
            answer = answer_with_context(q, top_ctx)
            answer = paraphrase_succinct(answer)
            results.append({"q": q, "a": answer})
        return {"items": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Optional GET handler for diagnostics if some clients hit GET /multiqa
@app.get("/multiqa")
async def multiqa_get():
    return JSONResponse(status_code=405, content={"error": "Use POST for /multiqa"})


@app.post("/tts")
async def tts(text: str = Form("")):
    if not text.strip():
        return JSONResponse(status_code=400, content={"error": "Empty text"})
    try:
        wav_bytes = await _tts_synthesize_safe(text)
        if not wav_bytes:
            return JSONResponse(status_code=500, content={"error": "TTS failed"})
        b64_audio = base64.b64encode(wav_bytes).decode("utf-8")
        return {"audio_b64_wav": b64_audio}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/audio-files")
async def audio_files():
    # Fast path: return cached list if recently fetched to avoid repeated refresh flicker
    try:
        sb = get_supabase()
        if sb is None:
            return {"items": []}
        items = list_transcripts(sb)
        return {"items": items}
    except Exception as e:
        logger.warning("/audio-files failed: %s", e)
        return {"items": []}


def _generate_silence_wav(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    import wave
    import struct
    num_samples = int(duration_seconds * sample_rate)
    with io.BytesIO() as buf:
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            silence_frame = struct.pack('<h', 0)
            for _ in range(num_samples):
                wf.writeframesraw(silence_frame)
        return buf.getvalue()


async def _generate_response_with_trained_model(transcript: str) -> str:
    """
    Generate response using QA system for intelligent content understanding
    """
    try:
        # Check if training is complete
        print(f"Training state - trained: {training_state.trained}, is_training: {training_state.is_training}")
        
        # Manual fix: If QA system is working, use it regardless of training state
        from .train import trainer, TRANSFORMERS_AVAILABLE
        print(f"TRANSFORMERS_AVAILABLE: {TRANSFORMERS_AVAILABLE}")
        print(f"trainer.qa_model: {trainer.qa_model is not None}")
        print(f"trainer.content_text: {bool(trainer.content_text)}")
        
        if TRANSFORMERS_AVAILABLE and trainer.qa_model and trainer.content_text:
            print("QA system is available, using it directly")
            response = await trainer.predict_response(transcript)
            print(f"QA system returned: {response}")
            if response and response != "I'm sorry, I had trouble understanding that.":
                return response
        else:
            print("QA system not ready - models not loaded")
        
        if training_state.trained:
            from .train import trainer
            
            print(f"Calling QA system for: {transcript}")
            # Use QA system for intelligent responses
            response = await trainer.predict_response(transcript)
            print(f"QA system returned: {response}")
            if response and response != "I'm sorry, I had trouble understanding that.":
                return response
        else:
            print("Training not complete, using fallback")
    except Exception as e:
        print(f"QA response generation failed: {e}")
    
    # Fall back to rule-based response
    print("Using fallback response generation")
    return _generate_response_text(transcript)


async def _get_transcribed_content() -> str:
    """
    Get the actual transcribed content from the training data
    """
    try:
        from .train import trainer
        whisper_dir = trainer.whisper_dir
        
        # Read all transcript files
        content_parts = []
        for txt_file in whisper_dir.glob("*.txt"):
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    content_parts.append(content)
        
        return " ".join(content_parts) if content_parts else ""
    except Exception as e:
        print(f"Error reading transcribed content: {e}")
        return ""


async def _answer_from_content(question: str, content: str) -> str:
    """
    Answer questions based on the actual transcribed content - completely dynamic
    """
    try:
        question_lower = question.lower()
        content_lower = content.lower()
        
        # Extract key terms from the question
        question_words = set(question_lower.split())
        
        # Split content into sentences for better matching
        import re
        sentences = re.split(r'[.!?]+', content)
        
        # Look for sentences that contain question keywords
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if not sentence_lower:
                continue
                
            # Check if sentence contains any words from the question
            sentence_words = set(sentence_lower.split())
            if question_words.intersection(sentence_words):
                relevant_sentences.append(sentence.strip())
        
        # If we found relevant sentences, use them to construct an answer
        if relevant_sentences:
            # Take the most relevant sentence (first one that matches)
            answer_sentence = relevant_sentences[0]
            
            # Clean up the sentence
            answer_sentence = re.sub(r'\s+', ' ', answer_sentence).strip()
            
            # If it's a complete sentence, return it
            if len(answer_sentence) > 10:  # Ensure it's substantial
                return f"Based on what I learned: {answer_sentence}"
        
        # If no direct match, try to find related information
        # Look for sentences that might be related to the topic
        topic_words = []
        for word in question_words:
            if len(word) > 3:  # Only consider substantial words
                topic_words.append(word)
        
        if topic_words:
            related_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower().strip()
                if not sentence_lower:
                    continue
                    
                # Check if sentence contains any topic words
                for topic_word in topic_words:
                    if topic_word in sentence_lower:
                        related_sentences.append(sentence.strip())
                        break
            
            if related_sentences:
                # Return the first related sentence
                answer_sentence = related_sentences[0]
                answer_sentence = re.sub(r'\s+', ' ', answer_sentence).strip()
                if len(answer_sentence) > 10:
                    return f"Based on what I learned: {answer_sentence}"
        
        return None  # No relevant information found
        
    except Exception as e:
        print(f"Error answering from content: {e}")
        return None


def _generate_response_text(transcript: str) -> str:
    t = transcript.strip().lower()
    if not t:
        return "I did not receive any input."
    
    # More flexible greeting detection
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    if any(greeting in t for greeting in greetings):
        return "Hello! How can I assist you today?"
    
    # Handle common variations and typos
    if "gello" in t or "hallo" in t or "helo" in t:
        return "Hello! How can I assist you today?"
    
    # Name and identity questions
    if "who are you" in t or ("what" in t and "are you" in t):
        return "I'm your local Audio Training & Response assistant, running fully offline."
    
    if "what is your name" in t or "what's your name" in t:
        return "My name is ATR Assistant. I'm your local voice assistant that runs completely offline."
    
    if "what can you do" in t or "what do you do" in t:
        return "I can listen to your voice, understand what you say, and respond with both text and speech. I can also help with basic questions and have conversations with you."
    
    # Time questions
    if "time" in t and ("what" in t or "current" in t):
        try:
            import datetime
            return f"The current time is {datetime.datetime.now().strftime('%H:%M')}"
        except Exception:
            return "I could not fetch the current time."
    
    # How are you questions
    if "how are you" in t:
        return "I'm doing well, thank you for asking! How can I help you today?"
    
    # Thank you responses
    if "thank" in t or "thanks" in t:
        return "You're welcome! Is there anything else I can help you with?"
    
    # Goodbye responses
    if "bye" in t or "goodbye" in t or "see you" in t:
        return "Goodbye! Have a great day!"
    
    # Weather questions
    if "weather" in t:
        return "I can't check the weather since I run completely offline, but I'm here to help with other questions!"
    
    # Help questions
    if "help" in t:
        return "I can help you with conversations, answer questions, tell you the time, and more. Just ask me anything!"
    
    # If it looks like a filename (contains brackets), provide helpful response
    if "[" in transcript and "]" in transcript:
        return "I received your audio message, but I had trouble understanding what you said. Could you try speaking more clearly or type your message instead?"
    
    # No hardcoded knowledge - let the system learn from audio content only
    
    # For questions that don't match specific patterns, provide a helpful response
    if "?" in transcript:
        return f"That's an interesting question about '{transcript.replace('?', '').strip()}'. I'm still learning, but I'm here to help with conversations and basic questions!"
    
    # For statements, acknowledge and ask for more
    return f"I understand you said '{transcript}'. That's interesting! Is there anything specific you'd like to know or discuss?"


async def _transcribe_whisper_safe(path: str) -> Optional[str]:
    global _whisper_model
    try:
        if _whisper_model is None:
            print("ERROR: Whisper model not pre-loaded! Falling back to lazy loading...")
            import whisper
            _whisper_model = whisper.load_model("small")
        
        print(f"Transcribing audio file: {path}")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Check if file is WebM and convert to WAV if needed
        if path.lower().endswith('.webm'):
            print("Converting WebM to WAV for Whisper compatibility...")
            import tempfile
            import subprocess
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                wav_path = tmp_wav.name
            
            try:
                # Use ffmpeg to convert WebM to WAV
                cmd = ['ffmpeg', '-i', path, '-ar', '16000', '-ac', '1', '-y', wav_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"Converted to WAV: {wav_path}")
                    path = wav_path
                else:
                    print(f"FFmpeg conversion failed: {result.stderr}")
                    # Try to use original file anyway
                    
            except FileNotFoundError:
                print("FFmpeg not found, trying original file...")
                # Try to use original file anyway
        
        # Run transcription with more verbose output
        result = _whisper_model.transcribe(path, fp16=(device == "cuda"), verbose=True)
        text = (result or {}).get("text", "").strip()
        print(f"Whisper transcription result: '{text}'")
        
        if text:
            return text
        else:
            print("Whisper returned empty text")
            return None
            
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        print(f"Audio file path: {path}")
        print(f"File exists: {os.path.exists(path)}")
        if os.path.exists(path):
            print(f"File size: {os.path.getsize(path)} bytes")
        return None


async def _tts_synthesize_safe(text: str) -> Optional[bytes]:
    # Try Piper first (lightweight local binary)
    wav_bytes = _tts_with_piper(text)
    if wav_bytes is not None:
        return wav_bytes

    # If Piper unavailable, try Coqui TTS (requires MSVC build tools on Windows)
    global _tts_model, _tts_vocoder
    try:
        if _tts_model is None:
            from TTS.api import TTS as CoquiTTS  # type: ignore
            _tts_model = CoquiTTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        wav = _tts_model.tts(text)
        import numpy as np
        import wave
        import struct
        sample_rate = 22050
        wav = np.asarray(wav, dtype=np.float32)
        wav_int16 = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)
        with io.BytesIO() as buf:
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(wav_int16.tobytes())
            return buf.getvalue()
    except Exception as e:
        print("TTS synthesis failed (Coqui):", e)
        return None


def _tts_with_piper(text: str) -> Optional[bytes]:
    try:
        print(f"DEBUG: Checking Piper files - EXE: {PIPER_EXE}, MODEL: {PIPER_MODEL}")
        print(f"DEBUG: EXE exists: {os.path.isfile(PIPER_EXE)}, MODEL exists: {os.path.isfile(PIPER_MODEL)}")
        if not (os.path.isfile(PIPER_EXE) and os.path.isfile(PIPER_MODEL)):
            print("DEBUG: Piper files not found, returning None")
            return None
        import tempfile
        import subprocess
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
            out_path = tf.name
        try:
            # Pass text via stdin to avoid shell quoting issues
            cmd = [PIPER_EXE,
                   "-m", os.path.basename(PIPER_MODEL),
                   "-f", os.path.basename(out_path),
                   "--espeak_data", "espeak-ng-data"]

            # If model JSON config exists, pass it explicitly
            model_json = os.path.basename(PIPER_MODEL) + ".json"
            model_json_abs = os.path.join(PIPER_DIR, model_json)
            if os.path.isfile(model_json_abs):
                cmd.extend(["-c", model_json])

            # Ensure DLLs are found by Piper by running inside the Piper directory and augmenting PATH
            env = os.environ.copy()
            env["PATH"] = f"{PIPER_DIR};{env.get('PATH','')}"

            result = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=PIPER_DIR,
                env=env,
            )
            print(f"DEBUG: Piper command: {cmd}")
            print(f"DEBUG: Piper return code: {result.returncode}")
            print(f"DEBUG: Piper stdout: {result.stdout.decode(errors='ignore')}")
            print(f"DEBUG: Piper stderr: {result.stderr.decode(errors='ignore')}")
            if result.returncode != 0:
                print("Piper TTS failed:", result.stderr.decode(errors='ignore'))
                return None
            # Read back the wav from Piper's output path inside PIPER_DIR
            out_abs = os.path.join(PIPER_DIR, os.path.basename(out_path))
            if os.path.isfile(out_abs):
                with open(out_abs, 'rb') as f:
                    return f.read()
            return None
        finally:
            try:
                # Clean temp file both relative (inside PIPER_DIR) and absolute
                out_abs = os.path.join(PIPER_DIR, os.path.basename(out_path))
                if os.path.isfile(out_abs):
                    os.remove(out_abs)
                if os.path.isfile(out_path):
                    os.remove(out_path)
            except Exception:
                pass
    except Exception as e:
        print("Piper invocation error:", e)
        return None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


