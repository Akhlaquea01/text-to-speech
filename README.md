# 🎯 Hinglish Text-to-Speech System

A unified **Hinglish Text-to-Speech (TTS)** application that converts mixed Hindi-English text into natural-sounding audio files. Features both web interface and command-line functionality with multiple TTS engines.

## ✨ Features

- 🗣️ **Mixed Language Support**: Handles Hinglish (Hindi + English) text seamlessly
- 🧠 **AI-Powered Normalization**: Uses Ollama LLM to fix punctuation and improve text flow
- 🎙️ **Multiple TTS Engines**: Edge TTS, Google TTS, and Advanced AI models
- 🌐 **Web Interface**: Beautiful web UI for easy text-to-speech conversion
- 💻 **Command Line**: Full CLI support for automation and scripting
- 🏠 **Fully Offline**: No API keys or external cloud dependencies
- ⚡ **Auto Hardware Detection**: Automatically uses GPU when available
- 🎵 **Multiple Formats**: Outputs MP3/WAV audio files

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** (tested with Python 3.14)
2. **8GB+ RAM** (16GB+ recommended for advanced models)
3. **GPU with CUDA** (optional, but recommended for advanced TTS)

### Installation

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**
   ```bash
   # Web interface (recommended)
   python main.py --web
   
   # Command line
   python main.py --text "Aaj meeting hai at 5 pm"
   ```

3. **Open your browser** and go to: `http://localhost:8000` (for web interface)

## 📖 Usage

### Web Interface (Recommended)

Start the web server:
```bash
python main.py --web
```

Then open `http://localhost:8000` in your browser to use the interactive interface.

### Command Line Usage

#### Basic Usage
```bash
python main.py --text "Aaj meeting hai at 5 pm"
```

#### Advanced Usage
```bash
# Custom output file and speed
python main.py --text "Kal subah office jana hai and I'm not ready yet!" --output my_audio.mp3 --speed 1.2

# Use different voice styles for more natural speech
python main.py --text "Welcome to our meeting" --voice-style conversational
python main.py --text "This is important" --voice-style professional
python main.py --text "Let's have fun!" --voice-style energetic

# Use different TTS engines
python main.py --text "Hello world" --tts-engine gtts
python main.py --text "Advanced speech" --tts-engine advanced

# Use different Ollama model for normalization
python main.py --text "Hello world" --model mistral

# Skip text normalization
python main.py --text "Perfect text already" --skip-normalization
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--web` | Start web interface | False |
| `--text` | Hinglish text to convert (required for CLI) | - |
| `--output` | Output audio file path | `output.mp3` |
| `--speed` | Speech speed multiplier | `1.0` |
| `--voice-style` | Voice style for natural speech | `natural` |
| `--tts-engine` | TTS engine to use | `edge` |
| `--model` | Ollama model for normalization | `llama3` |
| `--skip-normalization` | Skip AI text normalization | False |
| `--host` | Host for web interface | `0.0.0.0` |
| `--port` | Port for web interface | `8000` |

## 🔧 TTS Engines

### 1. Edge TTS (Default)
- **High Quality**: Microsoft neural voices
- **Fast**: Quick generation
- **Multilingual**: Supports Hindi and English
- **Voice Styles**: Natural, conversational, professional, etc.

### 2. Google TTS (gTTS)
- **Reliable**: Google's text-to-speech
- **Language Detection**: Automatic Hindi/English detection
- **Offline**: No internet required after setup

### 3. Advanced AI Models
- **Best Quality**: State-of-the-art AI models
- **Multiple Speakers**: 4 different voice options
- **GPU Accelerated**: Fast processing with CUDA
- **Resource Intensive**: Requires significant RAM/GPU

## 🎭 Voice Styles

- **natural** - Warm, natural voice (default)
- **conversational** - Friendly, conversational tone
- **friendly** - Warm, approachable voice
- **professional** - Clear, professional tone
- **expressive** - Emotional, dynamic voice
- **calm** - Soothing, relaxed voice
- **energetic** - Dynamic, enthusiastic voice
- **hindi** - Hindi voice for Hindi text

## 📁 Project Structure

```
text-to-speech/
├── main.py              # Main executable (web + CLI)
├── requirements.txt      # Python dependencies
├── templates/           # Web interface templates
│   └── index.html       # Main web page
├── static/              # Web interface assets
│   ├── style.css        # Styling
│   └── script.js        # JavaScript
└── README.md           # This file
```

## 🔧 How It Works

1. **Text Input**: You provide mixed Hindi-English text
2. **AI Normalization**: Ollama LLM improves punctuation and flow (optional)
3. **Language Detection**: Automatically detects Hindi vs English content
4. **Speech Synthesis**: Selected TTS engine converts text to natural speech
5. **Audio Output**: Saves as MP3/WAV file

### Example Workflow

```bash
# Input
python main.py --text "Kal subah office jana hai and I'm not ready yet!"

# AI Normalization (Ollama)
# Input:  "Kal subah office jana hai and I'm not ready yet!"
# Output: "Kal subah office jaana hai, and I'm not ready yet!"

# TTS Generation (Edge TTS)
# Output: output.mp3 (natural-sounding speech)
```

## 🛠️ Troubleshooting

### Common Issues

1. **"Missing required package"**
   - Run: `pip install -r requirements.txt`

2. **"Web interface not available"**
   - Install FastAPI: `pip install fastapi uvicorn`

3. **"Advanced models not available"**
   - Install PyTorch: `pip install torch transformers`
   - Ensure sufficient GPU memory (8GB+)

4. **"Ollama normalization failed"**
   - Install Ollama from https://ollama.ai/
   - Make sure Ollama service is running

### Performance Tips

- **First run**: May take longer as TTS models are downloaded
- **Memory**: Use `--tts-engine edge` for lower memory usage
- **Speed**: Use `--skip-normalization` for faster processing
- **GPU**: Advanced models work best with CUDA GPU

## 📝 Example Texts

Try these Hinglish examples:

```bash
# Business
python main.py --text "Aaj client meeting hai at 3 PM, please be ready"

# Daily conversation  
python main.py --text "Kal subah gym jana hai, phir office"

# Mixed technical
python main.py --text "API integration complete hai, ab testing karna hai"

# Pure Hindi
python main.py --text "नमस्ते, यह एक टेस्ट है" --voice-style hindi

# Hindi with Google TTS
python main.py --text "आज मीटिंग है" --tts-engine gtts
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.

---

**Made with ❤️ for the Hinglish-speaking community**