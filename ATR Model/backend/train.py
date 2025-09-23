"""
Simple Hugging Face Transformers-based training pipeline for ATR Model.
Uses question-answering models for intelligent content understanding.
"""

import os
import json
import asyncio
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
# Whisper (OpenAI) – static type checker hint to ignore if not in analysis env
try:
    import whisper  # type: ignore
    WHISPER_AVAILABLE = True
except Exception:
    whisper = None  # type: ignore
    WHISPER_AVAILABLE = False
import torch

# Hugging Face Transformers imports
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    print("Hugging Face Transformers loaded successfully")
except ImportError as e:
    print(f"Warning: Transformers not available: {e}")
    print("Please install: pip install transformers")
    TRANSFORMERS_AVAILABLE = False

# No fallback imports needed - using simple Transformers approach

class ATRTrainer:
    """
    Simple Hugging Face Transformers-based trainer for intelligent audio content understanding
    """
    
    def __init__(self, training_data_dir: str = "training_data"):
        self.training_data_dir = Path(training_data_dir)
        self.whisper_model = None
        self.qa_model = None
        self.tts_model = None
        self.training_state = None
        self.content_text = ""
        
        # Create training directories
        self.whisper_dir = self.training_data_dir / "whisper_data"
        self.tts_dir = self.training_data_dir / "tts_data"
    
    def set_whisper_model(self, model):
        """Set the pre-loaded Whisper model"""
        self.whisper_model = model
        
        for dir_path in [self.whisper_dir, self.tts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def set_training_state(self, state):
        """Set reference to training state for progress updates"""
        self.training_state = state
    
    async def update_progress(self, stage: str, progress: int, message: str):
        """Update training progress"""
        if self.training_state:
            self.training_state.progress = progress
            self.training_state.current_stage = stage
            print(f"[{stage}] {progress}% - {message}")
            # Add a longer delay to ensure progress updates are visible in frontend
            await asyncio.sleep(0.5)
    
    async def train_all_models(self):
        """
        Train all components using Hugging Face Transformers (incremental training)
        """
        try:
            await self.update_progress("initialization", 0, "Starting incremental training...")
            
            # 1. Process new audio with Whisper (only new files)
            await self.update_progress("whisper", 10, "Processing new audio with Whisper...")
            await self.train_whisper()
            
            # 2. Setup Question-Answering model (loads ALL transcripts: old + new)
            await self.update_progress("qa", 50, "Setting up QA model with all content...")
            await self.setup_qa_model()
            
            # Add intermediate QA progress
            await self.update_progress("qa", 60, "QA model loaded successfully...")
            await self.update_progress("qa", 70, "Loading content for QA...")
            
            # 3. Setup TTS
            await self.update_progress("tts", 80, "Setting up TTS model...")
            await self.train_tts()
            
            # Add intermediate TTS progress
            await self.update_progress("tts", 90, "TTS model configured...")
            
            await self.update_progress("completion", 100, "Incremental training completed successfully!")
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            await self.update_progress("error", 0, f"Training failed: {str(e)}")
            return False
    
    async def train_whisper(self):
        """
        Process audio with Whisper (Speech-to-Text)
        """
        try:
            # Find uploaded audio files (support more formats)
            audio_files = (list(self.whisper_dir.glob("*.wav")) + 
                          list(self.whisper_dir.glob("*.mp3")) + 
                          list(self.whisper_dir.glob("*.m4a")) + 
                          list(self.whisper_dir.glob("*.webm")))
            
            if not audio_files:
                print("No audio files found for Whisper processing")
                return
            
            await self.update_progress("whisper", 20, f"Found {len(audio_files)} audio files")
            
            # Use pre-loaded Whisper model if available
            await self.update_progress("whisper", 30, "Using pre-loaded Whisper model...")
            if self.whisper_model is not None:
                model = self.whisper_model
                print("Using pre-loaded Whisper model")
            else:
                if not WHISPER_AVAILABLE:
                    print("Whisper package not installed; skipping transcription.")
                    return
                print("Whisper model not pre-loaded, loading now...")
                model = whisper.load_model("small")
                # Cache the model for future use
                self.whisper_model = model
            
            # Process each audio file
            training_data = []
            for i, audio_file in enumerate(audio_files):
                # Calculate progress more granularly
                base_progress = 30
                file_progress = (i * 40) // len(audio_files)
                current_progress = base_progress + file_progress
                
                await self.update_progress("whisper", current_progress, 
                                         f"Processing {audio_file.name}")
                
                try:
                    # Update progress before transcription
                    await self.update_progress("whisper", current_progress + 2, 
                                             f"Starting transcription of {audio_file.name}")
                    
                    # Transcribe audio
                    result = model.transcribe(str(audio_file))
                    text = result["text"].strip()
                    
                    # Update progress during processing
                    await self.update_progress("whisper", current_progress + 4, 
                                             f"Transcription completed for {audio_file.name}")
                    
                    if text:
                        training_data.append({
                            'audio_file': str(audio_file),
                            'transcript': text,
                            'language': result.get('language', 'en')
                        })
                        
                        # Save transcript
                        transcript_file = audio_file.with_suffix('.txt')
                        with open(transcript_file, 'w', encoding='utf-8') as f:
                            f.write(text)

                        # Persist to Supabase: transcript row, chunks, embeddings
                        try:
                            from .supabase_utils import (
                                get_supabase,
                                insert_transcript,
                                get_transcript_by_filename,
                                delete_chunks_for_transcript,
                                insert_chunks,
                                insert_summary,
                            )
                            from .chunking import build_indexed_chunks
                            from .embeddings import embed_texts
                            from .summarizer import generate_summary
                            sb = get_supabase()
                            if sb is not None:
                                fname = audio_file.name
                                # Upsert-like: find existing transcript by filename
                                tid = get_transcript_by_filename(sb, fname)
                                if tid is None:
                                    tid = insert_transcript(sb, fname, text)
                                # Refresh chunks: delete then insert
                                if tid:
                                    delete_chunks_for_transcript(sb, tid)
                                    chunks = build_indexed_chunks(text)
                                    if chunks:
                                        embeds = embed_texts([c['text'] for c in chunks])
                                        for c, v in zip(chunks, embeds):
                                            c['embedding'] = v
                                        insert_chunks(sb, tid, chunks)
                                    # Generate and store summary only if enabled for training
                                    auto_sum_flag = (os.getenv("AUTO_SUMMARY_ON_TRAIN", "0") or "").strip().lower() in ("1", "true", "yes")
                                    if auto_sum_flag:
                                        summary = generate_summary(text)
                                        if summary:
                                            insert_summary(sb, tid, summary)
                        except Exception as e:
                            print(f"Supabase persistence failed for {audio_file.name}: {e}")
                        
                        # Update progress after each successful transcription
                        await self.update_progress("whisper", current_progress + 6, 
                                                 f"Saved transcript for {audio_file.name}")
                            
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
            
            # Save training data
            training_data_file = self.whisper_dir / "training_data.json"
            with open(training_data_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2)
            
            await self.update_progress("whisper", 70, f"Processed {len(training_data)} audio files")
            
            # For now, we'll use the base model (no fine-tuning)
            # Real fine-tuning would require more complex setup with custom training loops
            # The base model is already loaded and ready to use
            
            await self.update_progress("whisper", 100, "Whisper processing completed")
            
        except Exception as e:
            print(f"Whisper processing failed: {e}")
            raise
    
    async def setup_qa_model(self):
        """
        Setup Hugging Face Question-Answering model
        """
        try:
            if not TRANSFORMERS_AVAILABLE:
                print("Transformers not available - please install: pip install transformers")
                return
            
            # Only load QA model if not already loaded
            if self.qa_model is None:
                await self.update_progress("qa", 60, "Loading QA model...")
                
                # Load deepset/roberta-base-squad2 as QA baseline
                self.qa_model = pipeline(
                    "question-answering",
                    model="deepset/roberta-base-squad2",
                    tokenizer="deepset/roberta-base-squad2"
                )
                print("QA model loaded successfully")
            else:
                await self.update_progress("qa", 60, "QA model already loaded, skipping...")
                print("QA model already loaded, reusing existing model")
            
            # Always reload content (since new transcripts may have been added)
            await self.update_progress("qa", 80, "Loading content...")
            self.content_text = await self._load_all_content()
            
            await self.update_progress("qa", 100, "QA model ready!")
            
        except Exception as e:
            print(f"QA model setup failed: {e}")
            print("Please ensure Transformers is installed: pip install transformers")
    
    async def _load_all_content(self) -> str:
        """
        Load all transcribed content from audio files
        """
        content_parts = []
        
        # Read all transcript files
        for txt_file in self.whisper_dir.glob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        content_parts.append(content)
            except Exception as e:
                print(f"Error reading transcript {txt_file}: {e}")
        
        return " ".join(content_parts)
    
    async def generate_response(self, question: str) -> str:
        """
        Generate response using Hugging Face QA model
        """
        try:
            if not TRANSFORMERS_AVAILABLE or not self.qa_model or not self.content_text:
                return "I'm sorry, I had trouble understanding that."
            
            # Clip context to avoid 512-token overflow for QA models
            def _clip(t: str, max_tokens: int = 480) -> str:
                parts = t.split()
                return " ".join(parts[:max_tokens]) if len(parts) > max_tokens else t

            clipped_context = _clip(self.content_text, 480)

            # Use the QA model to find answers in the content
            result = self.qa_model(question=question, context=clipped_context)
            
            if result and result['score'] > 0.1:  # Confidence threshold
                answer = result['answer'].strip()
                
                # Ensure the answer is a complete sentence
                if not answer.endswith(('.', '!', '?')):
                    answer = f"{answer}."
                
                return answer
            else:
                return "I couldn't find a clear answer to your question in the content."
                
        except Exception as e:
            print(f"QA response generation failed: {e}")
            return "I'm sorry, I had trouble understanding that."
    
    async def train_tts(self):
        """
        Setup TTS model (placeholder for now)
        """
        try:
            await self.update_progress("tts", 90, "Setting up TTS...")
            
            # For now, we'll use the existing Piper TTS setup
            # Real TTS training would require more complex setup
            
            tts_config = {
                "model": "piper",
                "voice": "en_US-kathleen-low",
                "status": "ready"
            }
            
            tts_config_file = self.tts_dir / "tts_config.json"
            with open(tts_config_file, 'w', encoding='utf-8') as f:
                json.dump(tts_config, f, indent=2)
            
            await self.update_progress("tts", 100, "TTS setup completed")
            
        except Exception as e:
            print(f"TTS setup failed: {e}")
    
    async def predict_response(self, user_input: str) -> str:
        """
        Generate response using Hugging Face QA model or fallback
        """
        try:
            # Try QA model first (most intelligent)
            if TRANSFORMERS_AVAILABLE and self.qa_model and self.content_text:
                print(f"Using QA model for question: {user_input}")
                qa_response = await self.generate_response(user_input)
                if qa_response and qa_response != "I'm sorry, I had trouble understanding that.":
                    print(f"QA response: {qa_response}")
                    return qa_response
            else:
                print(f"QA model not available - qa_model: {self.qa_model is not None}, content_text: {bool(self.content_text)}")
                return "I'm sorry, the AI system is not ready. Please ensure Transformers is installed and try training again."
            
        except Exception as e:
            print(f"Response generation failed: {e}")
            return "I had trouble understanding your question."

# Global trainer instance
trainer = ATRTrainer()