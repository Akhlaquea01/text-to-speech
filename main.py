#!/usr/bin/env python3
"""
Hinglish Text-to-Speech System with Emotion Tags
Web-based TTS application with support for emotion styling via LLM processing.
Uses Google TTS (gTTS) - open source, no SSL issues.
"""

import os
import asyncio
import glob
import re
from pathlib import Path
from datetime import datetime

# Web interface imports
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel

# TTS imports
from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import speedup, normalize
import io

# Advanced TTS imports
try:
    import torch
    from transformers import VitsModel, AutoTokenizer
    import torchaudio
    ADVANCED_TTS_AVAILABLE = True
except ImportError:
    ADVANCED_TTS_AVAILABLE = False
    print("[WARN] Advanced TTS libraries not found. High-quality engine disabled.")


# Ollama import
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[WARN] Ollama not available. Emotion tag processing will use basic mapping.")


class TTSRequest(BaseModel):
    text: str
    emotion: str = "neutral"
    speed: float = 1.0
    use_llm: bool = True
    engine: str = "gtts"


class VeenaTTS:
    """Advanced TTS model using Hugging Face transformers"""

    def __init__(self, model_name="maya-research/veena-tts"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the TTS model and tokenizer"""
        if not ADVANCED_TTS_AVAILABLE:
            print("[ERROR] Advanced TTS libraries not available.")
            return

        try:
            print(f"[TTS] Loading advanced model: {self.model_name}")
            # Load model with memory-efficient dtype
            self.model = VitsModel.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16
            )
            # Use a proper tokenizer for VITS models
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base")
            except:
                # Fallback to a simpler approach
                self.tokenizer = None
            print("[OK] Advanced TTS model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Could not load advanced TTS model: {e}")
            self.model = None
            self.tokenizer = None

    async def generate_tts(self, text: str, output_file: str):
        """Generate speech using the advanced TTS model"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Advanced TTS model not loaded.")

        try:
            print(f"[TTS] Generating with advanced model...")

            inputs = self.tokenizer(text, return_tensors="pt")

            with torch.no_grad():
                waveform = self.model(**inputs).waveform

            # Save the waveform to a file
            torchaudio.save(output_file, waveform.unsqueeze(0), sample_rate=self.model.config.sampling_rate)

            print(f"[OK] Advanced audio saved to {output_file}")

        except Exception as e:
            print(f"[ERROR] Advanced TTS generation failed: {e}")
            raise


class HinglishTTS:
    """Main TTS application class for web interface"""
    
    def __init__(self):
        self.app = None
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        self.veena_tts = None

        # Advanced TTS disabled by default due to model compatibility issues
        # if ADVANCED_TTS_AVAILABLE:
        #     self.veena_tts = VeenaTTS()
        #     self.veena_tts.load_model()
        
        # Emotion to speech parameters mapping
        # Using gTTS with pydub for audio manipulation
        self.emotion_params = {
            "neutral": {
                "lang": "en",
                "tld": "com",
                "slow": False,
                "pitch_shift": 0,
                "speed_factor": 1.0
            },
            "enthusiastically": {
                "lang": "en",
                "tld": "com",
                "slow": False,
                "pitch_shift": 3,
                "speed_factor": 1.15
            },
            "sarcastically": {
                "lang": "en",
                "tld": "co.uk",
                "slow": False,
                "pitch_shift": -2,
                "speed_factor": 0.95
            },
            "thoughtfully": {
                "lang": "en",
                "tld": "com",
                "slow": True,
                "pitch_shift": -1,
                "speed_factor": 0.85
            },
            "laughs warmly": {
                "lang": "en",
                "tld": "com.au",
                "slow": False,
                "pitch_shift": 2,
                "speed_factor": 1.05
            },
            "dramatically": {
                "lang": "en",
                "tld": "co.uk",
                "slow": False,
                "pitch_shift": 4,
                "speed_factor": 1.1
            },
            "softly": {
                "lang": "en",
                "tld": "com",
                "slow": True,
                "pitch_shift": -3,
                "speed_factor": 0.9
            },
            "confidently": {
                "lang": "en",
                "tld": "com",
                "slow": False,
                "pitch_shift": 1,
                "speed_factor": 1.0
            },
            "whispers": {
                "lang": "en",
                "tld": "com",
                "slow": True,
                "pitch_shift": -4,
                "speed_factor": 0.8
            },
            "excitedly": {
                "lang": "en",
                "tld": "com.au",
                "slow": False,
                "pitch_shift": 5,
                "speed_factor": 1.2
            }
        }

    def cleanup_old_files(self):
        """Delete old output files before generating new ones"""
        try:
            # Clean up output directory
            pattern = str(self.output_dir / "output_*.mp3")
            for file in glob.glob(pattern):
                try:
                    os.remove(file)
                    print(f"[CLEANUP] Deleted old file: {file}")
                except Exception as e:
                    print(f"[WARN] Could not delete {file}: {e}")
            
            # Clean up temp files
            temp_pattern = str(self.output_dir / "temp_*.mp3")
            for file in glob.glob(temp_pattern):
                try:
                    os.remove(file)
                    print(f"[CLEANUP] Deleted temp file: {file}")
                except Exception as e:
                    print(f"[WARN] Could not delete {file}: {e}")
        except Exception as e:
            print(f"[WARN] Cleanup error: {e}")

    async def process_emotion_tags_with_llm(self, text: str) -> list:
        """
        Process text with emotion tags using LLM to add appropriate styling.
        Returns list of (text_segment, emotion) tuples.
        """
        if not OLLAMA_AVAILABLE:
            return [(text, "neutral")]
        
        try:
            prompt = f"""You are a text-to-speech emotion analyzer. Analyze the following text and add emotion tags where appropriate.

Available emotion tags: [enthusiastically], [sarcastically], [thoughtfully], [laughs warmly], [dramatically], [softly], [confidently], [whispers], [excitedly], [neutral]

Input text: {text}

Please return ONLY the text with emotion tags inserted at appropriate positions. Format: [emotion] text segment
Example: [enthusiastically] I love this! [thoughtfully] But maybe we should consider other options.

Important: Return ONLY the formatted text with emotion tags, no explanations or additional commentary."""

            print(f"[LLM] Processing emotion tags with LLM...")
            
            response = await asyncio.to_thread(
                ollama.chat,
                model="llama3",
                    messages=[{'role': 'user', 'content': prompt}]
                )
            
            processed_text = response['message']['content'].strip()
            print(f"[LLM] Processed: {processed_text}")
            
            # Parse the emotion-tagged text
            segments = self.parse_emotion_tags(processed_text)
            return segments
            
        except Exception as e:
            print(f"[WARN] LLM processing failed: {e}, using basic parsing")
            return [(text, "neutral")]

    def parse_emotion_tags(self, text: str) -> list:
        """
        Parse text with emotion tags into segments.
        Returns list of (text_segment, emotion) tuples.
        """
        segments = []
        pattern = r'\[([^\]]+)\]\s*([^\[]+)'
        matches = re.findall(pattern, text)
        
        if matches:
            for emotion, segment in matches:
                emotion = emotion.lower().strip()
                segment = segment.strip()
                if segment:
                    # Map emotion to available emotions
                    if emotion not in self.emotion_params:
                        emotion = "neutral"
                    segments.append((segment, emotion))
        else:
            # No tags found, return as neutral
            segments.append((text.strip(), "neutral"))
        
        return segments

    async def generate_tts_gtts(self, text: str, output_file: str, emotion: str = "neutral", speed: float = 1.0):
        """Generate speech using Google TTS with emotion parameters."""
        try:
            # Get parameters for emotion
            params = self.emotion_params.get(emotion, self.emotion_params["neutral"])
            
            print(f"[TTS] Generating with Google TTS")
            print(f"[TTS] Emotion: {emotion}")
            print(f"[TTS] Language: {params['lang']}, TLD: {params['tld']}")
            print(f"[TTS] Speed factor: {params['speed_factor'] * speed}")
            
            # Generate speech with gTTS
            tts = gTTS(
                text=text,
                lang=params['lang'],
                tld=params['tld'],
                slow=params['slow']
            )
            
            # Save to bytes buffer
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            # Load audio with pydub
            audio = AudioSegment.from_file(fp, format="mp3")
            
            # Apply speed adjustment
            final_speed = params['speed_factor'] * speed
            if final_speed != 1.0:
                audio = speedup(audio, playback_speed=final_speed)
            
            # Apply pitch shift (simulate using octave shift)
            if params['pitch_shift'] != 0:
                # Rough pitch adjustment by changing sample rate
                new_sample_rate = int(audio.frame_rate * (1 + params['pitch_shift'] * 0.01))
                audio = audio._spawn(audio.raw_data, overrides={
                    'frame_rate': new_sample_rate
                }).set_frame_rate(audio.frame_rate)
            
            # Normalize audio
            audio = normalize(audio)
            
            # Export to file
            audio.export(output_file, format="mp3")
            print(f"[OK] Audio segment saved to {output_file}")
            
        except Exception as e:
            print(f"[ERROR] Google TTS generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def generate_tts_advanced(self, text: str, output_file: str):
        """Generate speech using the advanced TTS model."""
        if not self.veena_tts or not self.veena_tts.model:
            raise HTTPException(status_code=500, detail="Advanced TTS model not available.")

        try:
            await self.veena_tts.generate_tts(text, output_file)
        except Exception as e:
            print(f"[ERROR] Advanced TTS generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def setup_web_interface(self):
        """Setup FastAPI web interface."""
        self.app = FastAPI(title="Hinglish Text-to-Speech with Emotions", version="2.0.0")
        
        # Setup templates and static files
        templates = Jinja2Templates(directory="templates")
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        @self.app.get("/", response_class=HTMLResponse)
        async def read_root(request: Request):
            return templates.TemplateResponse("index.html", {"request": request})
        
        @self.app.get("/emotions")
        async def get_emotions():
            """Return available emotions"""
            return {"emotions": list(self.emotion_params.keys())}
        
        @self.app.post("/synthesize")
        async def synthesize(request: TTSRequest):
            try:
                print(f"\n{'='*50}")
                print(f"New TTS Request")
                print(f"Text: {request.text}")
                print(f"Emotion: {request.emotion}")
                print(f"Speed: {request.speed}x")
                print(f"Use LLM: {request.use_llm}")
                print(f"Engine: {request.engine}")
                print(f"{'='*50}\n")

                self.cleanup_old_files()

                if request.engine == "advanced" and not (self.veena_tts and self.veena_tts.model):
                    raise HTTPException(status_code=400, detail="Advanced TTS engine not available.")

                # For advanced TTS, we don't use emotion tags yet
                if request.engine == "advanced":
                    segments = [(request.text, "neutral")]
                else:
                    if '[' not in request.text and request.use_llm and OLLAMA_AVAILABLE:
                        print("Using LLM to add emotion tags...")
                        segments = await self.process_emotion_tags_with_llm(request.text)
                    elif '[' in request.text:
                        print("Parsing existing emotion tags...")
                        segments = self.parse_emotion_tags(request.text)
                    else:
                        segments = [(request.text, request.emotion)]

                print(f"Generated {len(segments)} segment(s)")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = str(self.output_dir / f"output_{timestamp}.mp3")

                # Combine text from all segments
                combined_text = " ".join([seg[0] for seg in segments])
                
                if request.engine == "advanced":
                    await self.generate_tts_advanced(combined_text, output_file)
                else:
                    # Use gTTS with the primary emotion
                    primary_emotion = segments[0][1]
                    if request.emotion != "neutral":
                        primary_emotion = request.emotion

                    await self.generate_tts_gtts(
                        combined_text,
                        output_file,
                        primary_emotion,
                        request.speed
                    )
                
                if not os.path.exists(output_file):
                    raise HTTPException(status_code=500, detail="Audio file not generated")
                
                print(f"\n[OK] Audio generation complete!")
                print(f"[INFO] Output file: {output_file}")
                print(f"[INFO] Size: {os.path.getsize(output_file) / 1024:.1f} KB\n")
                
                return FileResponse(
                    output_file,
                    media_type="audio/mpeg",
                    filename=f"tts_{timestamp}.mp3"
                )
                
            except Exception as e:
                print(f"\n[ERROR] TTS synthesis error: {e}\n")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))
        
        return self.app

    def run_web_server(self, host="0.0.0.0", port=8000):
        """Run the web server."""
        if not self.app:
            self.setup_web_interface()
        
        print(f"\n{'='*60}")
        print(f"[TTS] Hinglish Text-to-Speech with Emotion Tags")
        print(f"[TTS] Using Google TTS (gTTS) - No SSL issues!")
        print(f"{'='*60}")
        print(f"[INFO] Server: http://{host}:{port}")
        print(f"[INFO] Google TTS (gTTS): Available")
        print(f"[INFO] Advanced TTS: {'Available' if self.veena_tts and self.veena_tts.model else 'Not available'}")
        print(f"[INFO] Available emotions: {', '.join(self.emotion_params.keys())}")
        print(f"[INFO] LLM processing: {'Available' if OLLAMA_AVAILABLE else 'Not available'}")
        print(f"{'='*60}\n")
        
        uvicorn.run(self.app, host=host, port=port)


def main():
    """Main application entry point."""
    tts = HinglishTTS()
    tts.run_web_server()


if __name__ == "__main__":
    main()
