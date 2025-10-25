#!/usr/bin/env python3
"""
Hinglish Text-to-Speech System
Converts mixed Hindi-English text to natural-sounding MP3 audio using Ollama for text normalization and Coqui TTS for speech synthesis.
"""

import subprocess
import argparse
import sys
import os
from pathlib import Path

try:
    from gtts import gTTS
    import ollama
    import edge_tts
    import asyncio
    # Try to import pydub, but handle gracefully if it fails
    try:
        from pydub import AudioSegment
        PYDUB_AVAILABLE = True
    except ImportError:
        PYDUB_AVAILABLE = False
        print("⚠️  pydub not available - audio format conversion may be limited")
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)


def normalize_text_with_ollama(text, model="llama3"):
    """
    Normalize Hinglish text using Ollama LLM for better punctuation and flow.
    
    Args:
        text (str): Raw Hinglish text input
        model (str): Ollama model to use (default: llama3)
    
    Returns:
        str: Normalized text with proper punctuation and flow
    """
    try:
        # Create a prompt for text normalization
        prompt = f"""Please normalize and improve the following Hinglish text (mix of Hindi and English). 
        Fix punctuation, improve flow, and make it sound natural for text-to-speech.
        Only return the corrected text, no explanations:
        
        {text}"""
        
        print(f"🧠 Normalizing text with Ollama ({model})...")
        
        # Use ollama Python client if available, otherwise fallback to subprocess
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            normalized_text = response['message']['content'].strip()
        except Exception as e:
            print(f"⚠️  Ollama client failed, trying subprocess: {e}")
            # Fallback to subprocess method
            result = subprocess.run(
                ["ollama", "run", model, prompt], 
                capture_output=True, 
                text=True,
                timeout=60
            )
            if result.returncode != 0:
                raise Exception(f"Ollama subprocess failed: {result.stderr}")
            normalized_text = result.stdout.strip()
        
        # Clean up the response (remove any extra formatting)
        lines = normalized_text.split('\n')
        # Find the actual text (skip any model responses like "Here's the normalized text:")
        for line in lines:
            if line.strip() and not line.strip().startswith(('Here', 'The', 'Normalized', 'Corrected')):
                normalized_text = line.strip()
                break
        
        return normalized_text
        
    except Exception as e:
        print(f"⚠️  Ollama normalization failed: {e}")
        print("📝 Using original text without normalization...")
        return text


async def generate_tts_edge(text, output_file="output.mp3", speed=1.0, voice="en-US-AriaNeural", pitch="+0Hz", volume="+0%"):
    """
    Generate speech from text using Edge TTS (Microsoft's neural voices).
    
    Args:
        text (str): Text to convert to speech
        output_file (str): Output file path
        speed (float): Speech speed multiplier
        voice (str): Voice to use (default: en-US-AriaNeural)
        pitch (str): Voice pitch adjustment
        volume (str): Voice volume adjustment
    """
    try:
        print("🎙️ Initializing Edge TTS...")
        print(f"🎵 Using voice: {voice}")
        print(f"🎚️ Speed: {speed}x, Pitch: {pitch}, Volume: {volume}")
        
        # Create TTS communication with enhanced parameters
        communicate = edge_tts.Communicate(
            text, 
            voice, 
            rate=f"+{int((speed-1)*100)}%",
            pitch=pitch,
            volume=volume
        )
        
        print("🎤 Generating speech...")
        
        # Generate speech
        await communicate.save(output_file)
        
        print(f"✅ Audio saved to {output_file}")
        
        # Show file info
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / 1024  # KB
            print(f"📁 File size: {file_size:.1f} KB")
        
    except Exception as e:
        print(f"❌ Edge TTS generation failed: {e}")
        raise


def generate_tts_gtts(text, output_file="output.mp3", speed=1.0, lang="auto"):
    """
    Generate speech from text using Google TTS (gTTS).
    
    Args:
        text (str): Text to convert to speech
        output_file (str): Output file path
        speed (float): Speech speed multiplier (not supported by gTTS)
        lang (str): Language code (en, hi, auto)
    """
    try:
        print("🎙️ Initializing Google TTS...")
        
        # Auto-detect language if needed
        if lang == "auto":
            hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
            english_chars = sum(1 for char in text if char.isalpha())
            
            print(f"🔍 Hindi chars: {hindi_chars}, English chars: {english_chars}")
            
            if hindi_chars > 0:  # If any Hindi characters are present, use Hindi
                lang = "hi"
            else:
                lang = "en"
        
        print(f"🌍 Language: {lang}")
        
        # Create TTS object
        tts = gTTS(text=text, lang=lang, slow=False)
        
        print("🎤 Generating speech...")
        
        # Generate speech
        temp_file = "temp_tts.mp3"
        tts.save(temp_file)
        
        # Convert to final format if needed
        if output_file.endswith('.mp3'):
            os.rename(temp_file, output_file)
        else:
            if PYDUB_AVAILABLE:
                audio = AudioSegment.from_mp3(temp_file)
                audio.export(output_file, format="mp3")
                os.remove(temp_file)
            else:
                # If pydub not available, just rename the file
                os.rename(temp_file, output_file)
        
        print(f"✅ Audio saved to {output_file}")
        
        # Show file info
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / 1024  # KB
            print(f"📁 File size: {file_size:.1f} KB")
        
    except Exception as e:
        print(f"❌ Google TTS generation failed: {e}")
        raise


def generate_tts(text, output_file="output.mp3", speed=1.0, voice=None, tts_engine="edge", voice_style="natural"):
    """
    Generate speech from text using the specified TTS engine.
    
    Args:
        text (str): Text to convert to speech
        output_file (str): Output file path
        speed (float): Speech speed multiplier
        voice (str): Voice to use (optional)
        tts_engine (str): TTS engine to use ("edge" or "gtts")
        voice_style (str): Voice style ("natural", "conversational", "friendly")
    """
    if tts_engine == "edge":
        # Use Edge TTS (Microsoft neural voices) with enhanced voice options
        voice_map = {
            # Natural, conversational voices
            "natural": "en-US-AriaNeural",  # Warm, natural female voice
            "conversational": "en-US-JennyNeural",  # Conversational, friendly
            "friendly": "en-US-GuyNeural",  # Warm, friendly male voice
            "professional": "en-US-DavisNeural",  # Professional male voice
            "expressive": "en-US-AmberNeural",  # Expressive, emotional
            "calm": "en-US-MichelleNeural",  # Calm, soothing
            "energetic": "en-US-BrandonNeural",  # Energetic, dynamic
            "hi": "hi-IN-SwaraNeural",  # Hindi voice
            "default": "en-US-AriaNeural"
        }
        
        # Voice style parameters for more natural speech
        style_params = {
            "natural": {"pitch": "+0Hz", "volume": "+0%"},
            "conversational": {"pitch": "+5Hz", "volume": "+0%"},
            "friendly": {"pitch": "+10Hz", "volume": "+5%"},
            "professional": {"pitch": "-5Hz", "volume": "+0%"},
            "expressive": {"pitch": "+15Hz", "volume": "+10%"},
            "calm": {"pitch": "-10Hz", "volume": "-5%"},
            "energetic": {"pitch": "+20Hz", "volume": "+15%"}
        }
        
        # Language detection for better voice selection
        def detect_language(text):
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
        
        detected_lang = detect_language(text)
        print(f"🌍 Detected language: {detected_lang}")
        
        # Adjust voice selection based on language
        if detected_lang == "hi" or detected_lang == "mixed" or voice_style == "hindi":
            # Use Hindi voices for Hindi or mixed content
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
            # Use English voices for English content
            selected_voice = voice or voice_map.get(voice_style, voice_map["default"])
        
        params = style_params.get(voice_style, style_params["natural"])
        
        print(f"🎭 Voice style: {voice_style}")
        
        # Run async function with enhanced parameters
        asyncio.run(generate_tts_edge(
            text, 
            output_file, 
            speed, 
            selected_voice,
            params["pitch"],
            params["volume"]
        ))
        
    elif tts_engine == "gtts":
        # Use Google TTS with auto language detection
        generate_tts_gtts(text, output_file, speed, "auto")
        
    else:
        raise ValueError(f"Unsupported TTS engine: {tts_engine}")


def check_ollama_availability():
    """Check if Ollama is installed and running."""
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


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Hinglish Text-to-Speech System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --text "Aaj meeting hai at 5 pm"
  python app.py --text "Kal subah office jana hai and I'm not ready yet!" --speed 1.2
  python app.py --text "Hello world" --output my_audio.mp3
  python app.py --text "Welcome to our meeting" --voice-style conversational
  python app.py --text "This is important" --voice-style professional
  python app.py --text "Let's have fun!" --voice-style energetic
        """
    )
    
    parser.add_argument(
        "--text", 
        type=str, 
        required=True, 
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
        "--tts-engine", 
        type=str, 
        default="edge", 
        choices=["edge", "gtts"],
        help="TTS engine to use (default: edge)"
    )
    parser.add_argument(
        "--force-hindi", 
        action="store_true", 
        help="Force Hindi voice selection for mixed content"
    )
    
    args = parser.parse_args()
    
    print("🎯 Hinglish Text-to-Speech System")
    print("=" * 40)
    print(f"📝 Input text: {args.text}")
    print(f"📁 Output file: {args.output}")
    print(f"⚡ Speed: {args.speed}x")
    print(f"🎭 Voice style: {args.voice_style}")
    print(f"🔧 TTS Engine: {args.tts_engine}")
    print()
    
    # Check Ollama availability if normalization is enabled
    if not args.skip_normalization:
        if not check_ollama_availability():
            print("⚠️  Proceeding without text normalization...")
            args.skip_normalization = True
    
    try:
        # Step 1: Text normalization
        if args.skip_normalization:
            normalized_text = args.text
            print("📝 Using original text (normalization skipped)")
        else:
            normalized_text = normalize_text_with_ollama(args.text, args.model)
            print(f"📝 Normalized text: {normalized_text}")
        
        print()
        
        # Step 2: Generate TTS
        generate_tts(
            text=normalized_text,
            output_file=args.output,
            speed=args.speed,
            voice=args.voice,
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


if __name__ == "__main__":
    main()
