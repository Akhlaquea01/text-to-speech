const textInput = document.getElementById('text-input');
const normalizeCheckbox = document.getElementById('normalize-checkbox');
const convertBtn = document.getElementById('convert-btn');
const audioOutput = document.getElementById('audio-output');
const engineSelect = document.getElementById('engine-select');

convertBtn.addEventListener('click', async () => {
    const text = textInput.value;
    const use_llm = normalizeCheckbox.checked;
    const engine = engineSelect.value;

    const response = await fetch('/synthesize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: text,
            emotion: "neutral",
            speed: 1.0,
            use_llm: use_llm,
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