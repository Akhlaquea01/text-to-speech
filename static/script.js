const textInput = document.getElementById('text-input');
const normalizeCheckbox = document.getElementById('normalize-checkbox');
const convertBtn = document.getElementById('convert-btn');
const audioOutput = document.getElementById('audio-output');
const speakerSelect = document.getElementById('speaker-select');
const ttsEngineSelect = document.getElementById('tts-engine-select');
const voiceStyleSelect = document.getElementById('voice-style-select');
const speedSlider = document.getElementById('speed-slider');
const speedValue = document.getElementById('speed-value');

// Update speed display
speedSlider.addEventListener('input', () => {
    speedValue.textContent = speedSlider.value;
});

convertBtn.addEventListener('click', async () => {
    const text = textInput.value;
    const normalize = normalizeCheckbox.checked;
    const speaker = speakerSelect.value;
    const ttsEngine = ttsEngineSelect.value;
    const voiceStyle = voiceStyleSelect.value;
    const speed = parseFloat(speedSlider.value);

    if (!text.trim()) {
        alert('Please enter some text to convert to speech.');
        return;
    }

    // Show loading state
    convertBtn.disabled = true;
    convertBtn.textContent = 'Converting...';

    try {
        const response = await fetch('/synthesize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                normalize: normalize,
                speaker: speaker,
                tts_engine: ttsEngine,
                voice_style: voiceStyle,
                speed: speed
            })
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            audioOutput.src = url;
            audioOutput.play();
        } else {
            const errorData = await response.json();
            console.error('TTS synthesis failed:', errorData);
            alert('TTS synthesis failed: ' + (errorData.detail || 'Unknown error'));
        }
    } catch (error) {
        console.error('Network error:', error);
        alert('Network error: ' + error.message);
    } finally {
        // Reset button state
        convertBtn.disabled = false;
        convertBtn.textContent = 'Convert to Speech';
    }
});