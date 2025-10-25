# 🎯 Hinglish Text-to-Speech System

A local **Hinglish Text-to-Speech (TTS)** system that converts mixed Hindi-English text into natural-sounding MP3 audio files. The system uses **Ollama LLM** for intelligent text normalization and **Coqui TTS** for high-quality speech synthesis.

## ✨ Features

- 🗣️ **Mixed Language Support**: Handles Hinglish (Hindi + English) text seamlessly
- 🧠 **AI-Powered Normalization**: Uses Ollama LLM to fix punctuation and improve text flow
- 🎙️ **High-Quality TTS**: Generates natural-sounding speech using Coqui TTS
- 🏠 **Fully Offline**: No API keys or external cloud dependencies
- ⚡ **Customizable**: Adjustable speech speed and voice options
- 🎵 **Multiple Formats**: Outputs MP3 audio files

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+** (tested with Python 3.14)
2. **Ollama** installed and running (optional, for text normalization)
3. **Internet connection** (for Edge TTS and Google TTS)

### Installation

1. **Clone or download this project**
   ```bash
   git clone <repository-url>
   cd hinglish-tts
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and setup Ollama**
   ```bash
   # Install Ollama (visit https://ollama.ai/)
   # Then pull a model:
   ollama pull llama3
   # or
   ollama pull mistral
   ```

4. **Optional: Install Ollama** (for AI text normalization)
   - Visit https://ollama.ai/ and install Ollama
   - Pull a model: `ollama pull llama3` or `ollama pull mistral`
   - **Note**: The system works without Ollama, but text normalization will be skipped

### Usage

#### Basic Usage
```bash
python app.py --text "Aaj meeting hai at 5 pm"
```

#### Advanced Usage
```bash
# Custom output file and speed
python app.py --text "Kal subah office jana hai and I'm not ready yet!" --output my_audio.mp3 --speed 1.2

# Use different voice styles for more natural speech
python app.py --text "Welcome to our meeting" --voice-style conversational
python app.py --text "This is important" --voice-style professional
python app.py --text "Let's have fun!" --voice-style energetic

# Use different Ollama model
python app.py --text "Hello world" --model mistral

# Skip text normalization
python app.py --text "Perfect text already" --skip-normalization
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--text` | Hinglish text to convert (required) | - |
| `--output` | Output audio file path | `output.mp3` |
| `--speed` | Speech speed multiplier | `1.0` |
| `--voice` | Path to reference voice file | None |
| `--voice-style` | Voice style for natural speech | `natural` |
| `--tts-engine` | TTS engine to use | `edge` |
| `--model` | Ollama model for normalization | `llama3` |
| `--skip-normalization` | Skip AI text normalization | False |

## 📁 Project Structure

```
hinglish-tts/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🔧 How It Works

1. **Text Input**: You provide mixed Hindi-English text
2. **AI Normalization**: Ollama LLM improves punctuation and flow
3. **Speech Synthesis**: Coqui TTS converts text to natural speech
4. **Audio Output**: Saves as MP3 file

### Example Workflow

```bash
# Input
python app.py --text "Kal subah office jana hai and I'm not ready yet!"

# AI Normalization (Ollama)
# Input:  "Kal subah office jana hai and I'm not ready yet!"
# Output: "Kal subah office jaana hai, and I'm not ready yet!"

# TTS Generation (Coqui)
# Output: output.mp3 (natural-sounding speech)
```

## 🛠️ Troubleshooting

### Common Issues

1. **"Ollama is not installed"**
   - Install Ollama from https://ollama.ai/
   - Make sure Ollama service is running

2. **"Missing required package"**
   - Run: `pip install -r requirements.txt`

3. **"TTS generation failed"**
   - Ensure FFmpeg is installed
   - Check if TTS models are downloaded (first run may take time)

4. **"Ollama subprocess failed"**
   - Make sure Ollama is running: `ollama list`
   - Try a different model: `--model mistral`

### Performance Tips

- **First run**: May take longer as TTS models are downloaded
- **Model size**: Larger Ollama models give better normalization but use more RAM
- **Speed**: Use `--skip-normalization` for faster processing

## 🎵 Supported Models

### Ollama Models (for text normalization)
- `llama3` (recommended)
- `mistral`
- `codellama`
- Any other Ollama model

### TTS Engines (automatically managed)
- **Edge TTS** (Microsoft neural voices) - Default, high quality
- **Google TTS** (gTTS) - Alternative option

### Voice Styles (for more natural speech)
- **natural** - Warm, natural voice (default)
- **conversational** - Friendly, conversational tone
- **friendly** - Warm, approachable voice
- **professional** - Clear, professional tone
- **expressive** - Emotional, dynamic voice
- **calm** - Soothing, relaxed voice
- **energetic** - Dynamic, enthusiastic voice
- **hindi** - Hindi voice for Hindi text

### Language Support
- **Automatic Language Detection** - Detects Hindi vs English text
- **Hindi Voices** - Uses `hi-IN-SwaraNeural` and `hi-IN-MadhurNeural` for Hindi text
- **Mixed Language** - Handles Hinglish (Hindi + English) text intelligently
- **Google TTS** - Alternative engine with Hindi support

## 📝 Example Texts

Try these Hinglish examples:

```bash
# Business
python app.py --text "Aaj client meeting hai at 3 PM, please be ready"

# Daily conversation  
python app.py --text "Kal subah gym jana hai, phir office"

# Mixed technical
python app.py --text "API integration complete hai, ab testing karna hai"

# Pure Hindi
python app.py --text "नमस्ते, यह एक टेस्ट है" --voice-style hindi

# Hindi with Google TTS
python app.py --text "आज मीटिंग है" --tts-engine gtts
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.

---

**Made with ❤️ for the Hinglish-speaking community**
