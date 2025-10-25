#!/usr/bin/env python3
"""
Hinglish Text-to-Speech System
A unified application that provides both web interface and command-line functionality.
Supports multiple TTS engines and automatic hardware detection.
"""

import argparse
import sys
import os
import subprocess
import asyncio
import torch
from pathlib import Path

# Web interface imports
try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.templating import Jinja2Templates
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    from pydantic import BaseModel
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    print("⚠️  Web interface not available. Install FastAPI for web features.")

# TTS imports
try:
    from gtts import gTTS
    import edge_tts
    import soundfile as sf
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing TTS package: {e}")
    TTS_AVAILABLE = False

# Advanced TTS imports (optional)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from snac import SNAC
    ADVANCED_TTS_AVAILABLE = True
except ImportError:
    ADVANCED_TTS_AVAILABLE = False
    print("⚠️  Advanced TTS models not available. Using basic TTS engines.")

# Ollama import
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama not available. Text normalization will be skipped.")

# Audio processing
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️  pydub not available - audio format conversion may be limited")


class TTSRequest(BaseModel):
    text: str
    normalize: bool = False
    speaker: str = "kavya"
    tts_engine: str = "advanced"
    voice_style: str = "natural"
    speed: float = 1.0


class HinglishTTS:
    """Main TTS application class"""
    
    def __init__(self):
        self.app = None
        self.advanced_models_loaded = False
        self.model = None
        self.tokenizer = None
        self.snac_model = None
        
        # Token constants for advanced TTS
        self.START_OF_SPEECH_TOKEN = 128257
        self.END_OF_SPEECH_TOKEN = 128258
        self.START_OF_HUMAN_TOKEN = 128259
        self.END_OF_HUMAN_TOKEN = 128260
        self.START_OF_AI_TOKEN = 128261
        self.END_OF_AI_TOKEN = 128262
        self.AUDIO_CODE_BASE_OFFSET = 128266
        
        self.speakers = ["kavya", "agastya", "maitri", "vinaya"]
        
        # Voice mappings for Edge TTS
        self.voice_map = {
            "natural": "en-US-AriaNeural",
            "conversational": "en-US-JennyNeural", 
            "friendly": "en-US-GuyNeural",
            "professional": "en-US-DavisNeural",
            "expressive": "en-US-AmberNeural",
            "calm": "en-US-MichelleNeural",
            "energetic": "en-US-BrandonNeural",
            "hindi": "hi-IN-SwaraNeural",
            "default": "en-US-AriaNeural"
        }
        
        self.style_params = {
            "natural": {"pitch": "+0Hz", "volume": "+0%"},
            "conversational": {"pitch": "+5Hz", "volume": "+0%"},
            "friendly": {"pitch": "+10Hz", "volume": "+5%"},
            "professional": {"pitch": "-5Hz", "volume": "+0%"},
            "expressive": {"pitch": "+15Hz", "volume": "+10%"},
            "calm": {"pitch": "-10Hz", "volume": "-5%"},
            "energetic": {"pitch": "+20Hz", "volume": "+15%"}
        }

    def check_ollama_availability(self):
        """Check if Ollama is installed and running."""
        if not OLLAMA_AVAILABLE:
            return False
            
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ Ollama is available")
                return True
            else:
                print("⚠️  Ollama is not running. Please start Ollama service.")
                return False
        except FileNotFoundError:
            print("❌ Ollama is not installed. Please install Ollama first:")
            print("   Visit: https://ollama.ai/")
            return False
        except Exception as e:
            print(f"⚠️  Error checking Ollama: {e}")
            return False

    def normalize_text_with_ollama(self, text, model="llama3"):
        """Normalize Hinglish text using Ollama LLM."""
        if not OLLAMA_AVAILABLE:
            print("⚠️  Ollama not available, using original text")
            return text
            
        try:
            prompt = f"""Please normalize and improve the following Hinglish text (mix of Hindi and English). 
            Fix punctuation, improve flow, and make it sound natural for text-to-speech.
            Only return the corrected text, no explanations:
            
            {text}"""
            
            print(f"🧠 Normalizing text with Ollama ({model})...")
            
            try:
                response = ollama.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                normalized_text = response['message']['content'].strip()
            except Exception as e:
                print(f"⚠️  Ollama client failed, trying subprocess: {e}")
                result = subprocess.run(
                    ["ollama", "run", model, prompt], 
                    capture_output=True, 
                    text=True,
                    timeout=60
                )
                if result.returncode != 0:
                    raise Exception(f"Ollama subprocess failed: {result.stderr}")
                normalized_text = result.stdout.strip()
            
            # Clean up the response
            lines = normalized_text.split('\n')
            for line in lines:
                if line.strip() and not line.strip().startswith(('Here', 'The', 'Normalized', 'Corrected')):
                    normalized_text = line.strip()
                    break
            
            return normalized_text
            
        except Exception as e:
            print(f"⚠️  Ollama normalization failed: {e}")
            print("📝 Using original text without normalization...")
            return text

    def detect_language(self, text):
        """Simple language detection based on character analysis."""
        hindi_chars = 0
        english_chars = 0
        
        for char in text:
            if '\u0900' <= char <= '\u097F':  # Hindi Unicode range
                hindi_chars += 1
            elif char.isalpha():
                english_chars += 1
        
        if hindi_chars > english_chars:
            return "hi"
        elif english_chars > hindi_chars:
            return "en"
        else:
            return "mixed"  # Hinglish

    async def generate_tts_edge(self, text, output_file="output.mp3", speed=1.0, voice="en-US-AriaNeural", pitch="+0Hz", volume="+0%"):
        """Generate speech using Edge TTS."""
        try:
            print("🎙️ Initializing Edge TTS...")
            print(f"🎵 Using voice: {voice}")
            print(f"🎚️ Speed: {speed}x, Pitch: {pitch}, Volume: {volume}")
            
            communicate = edge_tts.Communicate(
                text, 
                voice, 
                rate=f"+{int((speed-1)*100)}%",
                pitch=pitch,
                volume=volume
            )
            
            print("🎤 Generating speech...")
            await communicate.save(output_file)
            print(f"✅ Audio saved to {output_file}")
            
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / 1024
                print(f"📁 File size: {file_size:.1f} KB")
                
        except Exception as e:
            print(f"❌ Edge TTS generation failed: {e}")
            raise

    def generate_tts_gtts(self, text, output_file="output.mp3", speed=1.0, lang="auto"):
        """Generate speech using Google TTS."""
        try:
            print("🎙️ Initializing Google TTS...")
            
            if lang == "auto":
                hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
                english_chars = sum(1 for char in text if char.isalpha())
                
                print(f"🔍 Hindi chars: {hindi_chars}, English chars: {english_chars}")
                
                if hindi_chars > 0:
                    lang = "hi"
                else:
                    lang = "en"
            
            print(f"🌍 Language: {lang}")
            
            tts = gTTS(text=text, lang=lang, slow=False)
            
            print("🎤 Generating speech...")
            temp_file = "temp_tts.mp3"
            tts.save(temp_file)
            
            if output_file.endswith('.mp3'):
                os.rename(temp_file, output_file)
            else:
                if PYDUB_AVAILABLE:
                    audio = AudioSegment.from_mp3(temp_file)
                    audio.export(output_file, format="mp3")
                    os.remove(temp_file)
                else:
                    os.rename(temp_file, output_file)
            
            print(f"✅ Audio saved to {output_file}")
            
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / 1024
                print(f"📁 File size: {file_size:.1f} KB")
                
        except Exception as e:
            print(f"❌ Google TTS generation failed: {e}")
            raise

    def load_advanced_models(self):
        """Load advanced TTS models if available."""
        if not ADVANCED_TTS_AVAILABLE or self.advanced_models_loaded:
            return
            
        try:
            print("Loading advanced TTS models...")
            
            # Check if CUDA is available
            if torch.cuda.is_available():
                print("Using GPU acceleration...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    "maya-research/veena-tts",
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
            else:
                print("Using CPU (slower but compatible)...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    "maya-research/veena-tts",
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                )
                self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cpu()
            
            self.tokenizer = AutoTokenizer.from_pretrained("maya-research/veena-tts", trust_remote_code=True)
            self.advanced_models_loaded = True
            print("✅ Advanced models loaded successfully!")
            
        except Exception as e:
            print(f"⚠️  Advanced models not available: {e}")
            print("Using basic TTS engines instead...")

    def decode_snac_tokens(self, snac_tokens):
        """De-interleave and decode SNAC tokens to audio."""
        if not snac_tokens or len(snac_tokens) % 7 != 0:
            return None

        snac_device = next(self.snac_model.parameters()).device
        codes_lvl = [[] for _ in range(3)]
        llm_codebook_offsets = [self.AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]

        for i in range(0, len(snac_tokens), 7):
            codes_lvl[0].append(snac_tokens[i] - llm_codebook_offsets[0])
            codes_lvl[1].append(snac_tokens[i+1] - llm_codebook_offsets[1])
            codes_lvl[1].append(snac_tokens[i+4] - llm_codebook_offsets[4])
            codes_lvl[2].append(snac_tokens[i+2] - llm_codebook_offsets[2])
            codes_lvl[2].append(snac_tokens[i+3] - llm_codebook_offsets[3])
            codes_lvl[2].append(snac_tokens[i+5] - llm_codebook_offsets[5])
            codes_lvl[2].append(snac_tokens[i+6] - llm_codebook_offsets[6])

        hierarchical_codes = []
        for lvl_codes in codes_lvl:
            tensor = torch.tensor(lvl_codes, dtype=torch.int32, device=snac_device).unsqueeze(0)
            if torch.any((tensor < 0) | (tensor > 4095)):
                raise ValueError("Invalid SNAC token values")
            hierarchical_codes.append(tensor)

        with torch.no_grad():
            audio_hat = self.snac_model.decode(hierarchical_codes)

        return audio_hat.squeeze().clamp(-1, 1).cpu().numpy()

    def generate_speech_advanced(self, text, speaker="kavya", temperature=0.4, top_p=0.9, voice_style="natural"):
        """Generate speech using advanced TTS models with natural voice output."""
        if not self.advanced_models_loaded:
            raise ValueError("Advanced models not loaded")
        
        # Adjust generation parameters based on voice style for more natural output
        style_params = {
            "natural": {"temperature": 0.4, "top_p": 0.9, "repetition_penalty": 1.05},
            "conversational": {"temperature": 0.5, "top_p": 0.85, "repetition_penalty": 1.1},
            "friendly": {"temperature": 0.6, "top_p": 0.8, "repetition_penalty": 1.15},
            "professional": {"temperature": 0.3, "top_p": 0.95, "repetition_penalty": 1.02},
            "expressive": {"temperature": 0.7, "top_p": 0.75, "repetition_penalty": 1.2},
            "calm": {"temperature": 0.2, "top_p": 0.98, "repetition_penalty": 1.0},
            "energetic": {"temperature": 0.8, "top_p": 0.7, "repetition_penalty": 1.25}
        }
        
        params = style_params.get(voice_style, style_params["natural"])
        temperature = params["temperature"]
        top_p = params["top_p"]
        repetition_penalty = params["repetition_penalty"]
        
        print(f"🎭 Generating {voice_style} speech with speaker {speaker}")
        print(f"🎚️ Parameters: temp={temperature}, top_p={top_p}, rep_penalty={repetition_penalty}")
            
        prompt = f"<spk_{speaker}> {text}"
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

        input_tokens = [
            self.START_OF_HUMAN_TOKEN,
            *prompt_tokens,
            self.END_OF_HUMAN_TOKEN,
            self.START_OF_AI_TOKEN,
            self.START_OF_SPEECH_TOKEN
        ]

        input_ids = torch.tensor([input_tokens], device=self.model.device)
        max_tokens = min(int(len(text) * 1.3) * 7 + 21, 700)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=[self.END_OF_SPEECH_TOKEN, self.END_OF_AI_TOKEN]
            )

        generated_ids = output[0][len(input_tokens):].tolist()
        snac_tokens = [
            token_id for token_id in generated_ids
            if self.AUDIO_CODE_BASE_OFFSET <= token_id < (self.AUDIO_CODE_BASE_OFFSET + 7 * 4096)
        ]

        if not snac_tokens:
            raise ValueError("No audio tokens generated")

        audio = self.decode_snac_tokens(snac_tokens)
        return audio

    async def generate_tts_async(self, text, output_file="output.mp3", speed=1.0, voice=None, tts_engine="edge", voice_style="natural"):
        """Generate speech using the specified TTS engine (async version for web interface)."""
        if tts_engine == "edge":
            # Use Edge TTS with enhanced voice options
            detected_lang = self.detect_language(text)
            print(f"🌍 Detected language: {detected_lang}")
            
            if detected_lang == "hi" or detected_lang == "mixed" or voice_style == "hindi":
                hindi_voice_map = {
                    "natural": "hi-IN-SwaraNeural",
                    "conversational": "hi-IN-MadhurNeural", 
                    "friendly": "hi-IN-SwaraNeural",
                    "professional": "hi-IN-MadhurNeural",
                    "expressive": "hi-IN-SwaraNeural",
                    "calm": "hi-IN-MadhurNeural",
                    "energetic": "hi-IN-SwaraNeural",
                    "hindi": "hi-IN-SwaraNeural"
                }
                selected_voice = hindi_voice_map.get(voice_style, "hi-IN-SwaraNeural")
            else:
                selected_voice = self.voice_map.get(voice_style, self.voice_map["default"])
            
            params = self.style_params.get(voice_style, self.style_params["natural"])
            print(f"🎭 Voice style: {voice_style}")
            
            await self.generate_tts_edge(
                text, 
                output_file, 
                speed, 
                selected_voice,
                params["pitch"],
                params["volume"]
            )
            
        elif tts_engine == "gtts":
            self.generate_tts_gtts(text, output_file, speed, "auto")
            
        elif tts_engine == "advanced":
            if not self.advanced_models_loaded:
                self.load_advanced_models()
            
            if self.advanced_models_loaded:
                print("🎙️ Using advanced TTS models...")
                print(f"🎭 Speaker: {voice}")
                print(f"🎚️ Voice style: {voice_style}")
                
                # Use the speaker parameter for advanced TTS
                audio = self.generate_speech_advanced(text, voice, voice_style=voice_style)
                
                # Apply speed adjustment if needed
                if speed != 1.0:
                    print(f"⚡ Adjusting speed to {speed}x...")
                    # Simple speed adjustment by resampling
                    import numpy as np
                    original_rate = 24000
                    new_rate = int(original_rate * speed)
                    audio_resampled = np.interp(
                        np.linspace(0, len(audio), int(len(audio) * original_rate / new_rate)),
                        np.arange(len(audio)),
                        audio
                    )
                    audio = audio_resampled
                
                sf.write(output_file, audio, 24000)
                print(f"✅ Audio saved to {output_file}")
                
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file) / 1024
                    print(f"📁 File size: {file_size:.1f} KB")
            else:
                print("⚠️  Advanced models not available, falling back to Edge TTS")
                await self.generate_tts_async(text, output_file, speed, voice, "edge", voice_style)
        else:
            raise ValueError(f"Unsupported TTS engine: {tts_engine}")

    def generate_tts(self, text, output_file="output.mp3", speed=1.0, voice=None, tts_engine="edge", voice_style="natural"):
        """Generate speech using the specified TTS engine (sync version for CLI)."""
        if tts_engine == "edge":
            # Use Edge TTS with enhanced voice options
            detected_lang = self.detect_language(text)
            print(f"🌍 Detected language: {detected_lang}")
            
            if detected_lang == "hi" or detected_lang == "mixed" or voice_style == "hindi":
                hindi_voice_map = {
                    "natural": "hi-IN-SwaraNeural",
                    "conversational": "hi-IN-MadhurNeural", 
                    "friendly": "hi-IN-SwaraNeural",
                    "professional": "hi-IN-MadhurNeural",
                    "expressive": "hi-IN-SwaraNeural",
                    "calm": "hi-IN-MadhurNeural",
                    "energetic": "hi-IN-SwaraNeural",
                    "hindi": "hi-IN-SwaraNeural"
                }
                selected_voice = hindi_voice_map.get(voice_style, "hi-IN-SwaraNeural")
            else:
                selected_voice = self.voice_map.get(voice_style, self.voice_map["default"])
            
            params = self.style_params.get(voice_style, self.style_params["natural"])
            print(f"🎭 Voice style: {voice_style}")
            
            # Create a new event loop for CLI mode
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.generate_tts_edge(
                    text, 
                    output_file, 
                    speed, 
                    selected_voice,
                    params["pitch"],
                    params["volume"]
                ))
            finally:
                loop.close()
            
        elif tts_engine == "gtts":
            self.generate_tts_gtts(text, output_file, speed, "auto")
            
        elif tts_engine == "advanced":
            if not self.advanced_models_loaded:
                self.load_advanced_models()
            
            if self.advanced_models_loaded:
                print("🎙️ Using advanced TTS models...")
                print(f"🎭 Speaker: {voice}")
                print(f"🎚️ Voice style: {voice_style}")
                
                # Use the speaker parameter for advanced TTS
                audio = self.generate_speech_advanced(text, voice, voice_style=voice_style)
                
                # Apply speed adjustment if needed
                if speed != 1.0:
                    print(f"⚡ Adjusting speed to {speed}x...")
                    # Simple speed adjustment by resampling
                    import numpy as np
                    original_rate = 24000
                    new_rate = int(original_rate * speed)
                    audio_resampled = np.interp(
                        np.linspace(0, len(audio), int(len(audio) * original_rate / new_rate)),
                        np.arange(len(audio)),
                        audio
                    )
                    audio = audio_resampled
                
                sf.write(output_file, audio, 24000)
                print(f"✅ Audio saved to {output_file}")
                
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file) / 1024
                    print(f"📁 File size: {file_size:.1f} KB")
            else:
                print("⚠️  Advanced models not available, falling back to Edge TTS")
                self.generate_tts(text, output_file, speed, voice, "edge", voice_style)
        else:
            raise ValueError(f"Unsupported TTS engine: {tts_engine}")

    def setup_web_interface(self):
        """Setup FastAPI web interface."""
        if not WEB_AVAILABLE:
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
            
        self.app = FastAPI(title="Hinglish Text-to-Speech", version="1.0.0")
        
        # Setup templates and static files
        templates = Jinja2Templates(directory="templates")
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        @self.app.get("/", response_class=HTMLResponse)
        async def read_root(request: Request):
            return templates.TemplateResponse("index.html", {"request": request})
        
        @self.app.post("/synthesize")
        async def synthesize(request: TTSRequest):
            try:
                text_to_synthesize = request.text
                
                if request.normalize and OLLAMA_AVAILABLE:
                    try:
                        text_to_synthesize = self.normalize_text_with_ollama(request.text)
                    except Exception as e:
                        print(f"Ollama normalization failed: {e}")
                        print("Using original text without normalization...")

                print(f"Generating speech for: {text_to_synthesize}")
                
                # Generate audio using async version
                await self.generate_tts_async(
                    text_to_synthesize,
                    "output.mp3",
                    request.speed,
                    request.speaker,
                    request.tts_engine,
                    request.voice_style
                )
                output_filename = "output.mp3"
                
                print(f"Audio saved to {output_filename}")
                return FileResponse(output_filename, media_type="audio/mpeg")
                
            except Exception as e:
                print(f"TTS synthesis error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        return self.app

    def run_web_server(self, host="0.0.0.0", port=8000):
        """Run the web server."""
        if not self.app:
            self.setup_web_interface()
        
        print(f"🌐 Starting web server at http://{host}:{port}")
        print("📝 Enter Hinglish text and select a speaker to generate speech!")
        uvicorn.run(self.app, host=host, port=port)

    def run_cli(self, args):
        """Run command-line interface."""
        print("🎯 Hinglish Text-to-Speech System")
        print("=" * 40)
        print(f"📝 Input text: {args.text}")
        print(f"📁 Output file: {args.output}")
        print(f"⚡ Speed: {args.speed}x")
        print(f"🎭 Voice style: {args.voice_style}")
        print(f"🔧 TTS Engine: {args.tts_engine}")
        print()
        
        # Check Ollama availability if normalization is enabled
        if not args.skip_normalization and OLLAMA_AVAILABLE:
            if not self.check_ollama_availability():
                print("⚠️  Proceeding without text normalization...")
                args.skip_normalization = True
        
        try:
            # Step 1: Text normalization
            if args.skip_normalization:
                normalized_text = args.text
                print("📝 Using original text (normalization skipped)")
            else:
                normalized_text = self.normalize_text_with_ollama(args.text, args.model)
                print(f"📝 Normalized text: {normalized_text}")
            
            print()
            
            # Step 2: Generate TTS
            self.generate_tts(
                text=normalized_text,
                output_file=args.output,
                speed=args.speed,
                voice=args.speaker,
                tts_engine=args.tts_engine,
                voice_style=args.voice_style
            )
            
            print()
            print("🎉 Success! Your Hinglish text has been converted to speech.")
            
        except KeyboardInterrupt:
            print("\n⏹️  Process interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Hinglish Text-to-Speech System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --text "Aaj meeting hai at 5 pm"
  python main.py --text "Kal subah office jana hai and I'm not ready yet!" --speed 1.2
  python main.py --text "Hello world" --output my_audio.mp3
  python main.py --text "Welcome to our meeting" --voice-style conversational
  python main.py --text "This is important" --voice-style professional
  python main.py --text "Let's have fun!" --voice-style energetic
  python main.py --web  # Start web interface
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--web", 
        action="store_true", 
        help="Start web interface instead of command line"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host for web interface (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port for web interface (default: 8000)"
    )
    
    # CLI arguments
    parser.add_argument(
        "--text", 
        type=str, 
        help="Hinglish text input to convert to speech"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output.mp3", 
        help="Output audio file path (default: output.mp3)"
    )
    parser.add_argument(
        "--speed", 
        type=float, 
        default=1.0, 
        help="Speech speed multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--voice", 
        type=str, 
        help="Path to reference voice file for cloning (optional)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="llama3", 
        help="Ollama model for text normalization (default: llama3)"
    )
    parser.add_argument(
        "--skip-normalization", 
        action="store_true", 
        help="Skip text normalization with Ollama"
    )
    parser.add_argument(
        "--voice-style", 
        type=str, 
        default="natural", 
        choices=["natural", "conversational", "friendly", "professional", "expressive", "calm", "energetic", "hindi"],
        help="Voice style for more natural speech (default: natural)"
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="kavya",
        choices=["kavya", "agastya", "maitri", "vinaya"],
        help="Speaker for the advanced TTS engine (default: kavya)"
    )
    parser.add_argument(
        "--tts-engine", 
        type=str, 
        default="advanced",
        choices=["edge", "gtts", "advanced"],
        help="TTS engine to use (default: advanced)"
    )
    parser.add_argument(
        "--force-hindi", 
        action="store_true", 
        help="Force Hindi voice selection for mixed content"
    )
    
    args = parser.parse_args()
    
    # Initialize TTS system
    tts = HinglishTTS()
    
    # Check if web mode is requested
    if args.web:
        if not WEB_AVAILABLE:
            print("❌ Web interface not available. Install FastAPI: pip install fastapi uvicorn")
            sys.exit(1)
        tts.run_web_server(host=args.host, port=args.port)
    else:
        # CLI mode
        if not args.text:
            print("❌ --text argument is required for CLI mode")
            print("Use --web to start web interface or provide --text for CLI")
            sys.exit(1)
        tts.run_cli(args)


if __name__ == "__main__":
    main()