# 🎙️ Hinglish Text-to-Speech with Emotion Tags

A modern web-based Text-to-Speech application that supports emotional styling and intelligent emotion detection using AI. Convert your text to natural-sounding speech with emotions like enthusiasm, sarcasm, thoughtfulness, and more!

## ✨ Features

- 🎭 **Emotion Tags**: Add emotional styling to your speech with tags like `[enthusiastically]`, `[sarcastically]`, `[thoughtfully]`, etc.
- 🧠 **AI-Powered Emotion Detection**: Automatically detect and apply appropriate emotions using Ollama LLM (optional)
- 🎵 **Multiple Voice Styles**: 10 different emotion presets with unique voice characteristics
- ⚡ **Adjustable Speed**: Control speech speed from 0.5x to 2.0x
- 🌐 **Modern Web Interface**: Beautiful, responsive UI with real-time feedback
- 🔄 **Async Processing**: Fully asynchronous for responsive performance
- 🗑️ **Auto Cleanup**: Automatically removes old audio files before generating new ones
- 📥 **Download Support**: Download generated audio files

## 🎭 Available Emotions

1. **😃 Enthusiastically** - Energetic and excited
2. **😏 Sarcastically** - Dry and ironic
3. **🤔 Thoughtfully** - Contemplative and measured
4. **😄 Laughs Warmly** - Friendly and joyful
5. **🎭 Dramatically** - Expressive and theatrical
6. **🌸 Softly** - Gentle and quiet
7. **💪 Confidently** - Strong and assured
8. **🤫 Whispers** - Very soft and intimate
9. **🎉 Excitedly** - High energy and animated
10. **😐 Neutral** - Standard speaking style

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd tts
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Ollama (Optional - for AI emotion detection)**
   - Visit [ollama.ai](https://ollama.ai) and follow installation instructions
   - Pull the llama3 model:
   ```bash
   ollama pull llama3
   ```

## 🎯 Usage

### Starting the Server

Simply run:
```bash
python main.py
```

The server will start at `http://localhost:8000`

### Using Emotion Tags

You can use emotion tags in your text in two ways:

1. **Manual Tags**: Insert emotion tags directly in your text
   ```
   [enthusiastically] I love this feature! [thoughtfully] But we should consider the implications.
   ```

2. **AI Auto-Detection**: Enable "Use AI to auto-detect emotions" and the LLM will automatically add appropriate emotion tags to your text.

### Web Interface

1. Enter your text in the text area
2. (Optional) Click emotion tag buttons to insert emotions at cursor position
3. Adjust speed slider (0.5x - 2.0x)
4. Select default emotion for untagged text
5. Toggle AI emotion detection on/off
6. Click "Generate Speech"
7. Listen to or download the generated audio

## 📁 Project Structure

```
tts/
├── main.py                 # FastAPI backend with emotion processing
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Web interface
├── static/
│   ├── script.js          # Frontend JavaScript
│   └── style.css          # Modern UI styling
└── output/                # Generated audio files (auto-created)
```

## 🔧 Configuration

### Emotion Voice Mapping

Each emotion is mapped to specific voice characteristics in `main.py`:
- Voice type (different neural voices)
- Pitch adjustment
- Volume adjustment
- Speaking rate

You can customize these mappings by editing the `emotion_voice_map` dictionary in the `HinglishTTS` class.

## 🛠️ Technical Details

### Backend
- **Framework**: FastAPI (async)
- **TTS Engine**: Edge TTS (Microsoft Azure Neural Voices)
- **LLM Integration**: Ollama (for emotion detection)
- **Audio Format**: MP3

### Frontend
- Vanilla JavaScript (no frameworks)
- Modern CSS with CSS Grid and Flexbox
- Responsive design for mobile and desktop
- Real-time status updates and feedback

## 📝 API Endpoints

### `GET /`
Returns the web interface

### `GET /emotions`
Returns list of available emotions
```json
{
  "emotions": ["neutral", "enthusiastically", "sarcastically", ...]
}
```

### `POST /synthesize`
Generate speech from text
```json
{
  "text": "Your text here",
  "emotion": "enthusiastically",
  "speed": 1.0,
  "use_llm": true
}
```

Returns: MP3 audio file

## 🎨 UI Features

- **Character Counter**: Real-time character count
- **Emotion Tag Buttons**: Quick insertion of emotion tags
- **Speed Slider**: Visual speed adjustment
- **Status Messages**: Color-coded feedback (info/success/error)
- **Audio Player**: Built-in playback with controls
- **Download Button**: Easy audio file download
- **Keyboard Shortcuts**: Ctrl/Cmd + Enter to generate

## 🔄 File Management

The application automatically:
- Creates an `output/` directory for generated files
- Deletes old output files before generating new ones
- Uses timestamps for unique filenames
- Cleans up temporary files after generation

## ⚠️ Requirements

- Python 3.8+
- Internet connection (for Edge TTS)
- Ollama (optional, for AI emotion detection)

## 🐛 Troubleshooting

### "Ollama not available" warning
This is normal if you haven't installed Ollama. The app will work without it, but won't have AI emotion detection.

### Audio not playing
Some browsers block auto-play. Click the play button on the audio player.

### Generation is slow
- First request may be slower as Edge TTS initializes
- LLM processing adds 2-5 seconds if enabled
- Consider disabling LLM for faster generation

## 🤝 Contributing

Contributions are welcome! Some ideas for improvements:
- Add more emotion types
- Support for multiple languages
- Audio segment concatenation for multi-emotion text
- Voice cloning support
- Batch processing

## 📄 License

MIT License - feel free to use and modify!

## 🙏 Acknowledgments

- **Edge TTS**: Microsoft Azure Neural Voices
- **Ollama**: Local LLM inference
- **FastAPI**: Modern Python web framework

---

Made with ❤️ for natural-sounding speech synthesis
