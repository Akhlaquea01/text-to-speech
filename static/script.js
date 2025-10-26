const textInput = document.getElementById('text-input');
const normalizeCheckbox = document.getElementById('normalize-checkbox');
const convertBtn = document.getElementById('convert-btn');
const audioOutput = document.getElementById('audio-output');
const speakerSelect = document.getElementById('speaker-select');
const engineSelect = document.getElementById('engine-select');

convertBtn.addEventListener('click', async () => {
    const text = textInput.value;
    const normalize = normalizeCheckbox.checked;
    const speaker = speakerSelect.value;
    const engine = engineSelect.value;

    const response = await fetch('/synthesize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: text,
            normalize: normalize,
            speaker: speaker,
            engine: engine
        })
    });

    if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        audioOutput.src = url;
        audioOutput.play();
    } else {
        console.error('TTS synthesis failed');
    }
});