const imgInput = document.getElementById('imageup');
const imgPreview = document.getElementById('img-preview');
const placeholder = document.getElementById('placeholder');
imgInput.addEventListener('change', function () {
    const [file] = this.files;
    if (file) {
        imgPreview.src = URL.createObjectURL(file);
        imgPreview.style.display = "block";
        placeholder.style.display = "none";
    } else {
        imgPreview.style.display = "none";
        placeholder.style.display = "flex";
    }
});

// Prompt suggestions fill
document.querySelectorAll('.suggestion-btn').forEach(btn => {
    btn.onclick = () => {
        const box = document.getElementById('promptbox');
        let val = box.value.trim();
        const newPrompt = btn.textContent;

        // Check if the prompt is already in the list (splitting by newlines)
        const currentPrompts = val ? val.split('\n').map(p => p.trim()) : [];

        if (!currentPrompts.includes(newPrompt)) {
            if (val) {
                box.value = val + '\n' + newPrompt;
            } else {
                box.value = newPrompt;
            }
        }
    };
});

document.getElementById("classify").onclick = async (e) => {
    e.preventDefault();
    const resultBox = document.getElementById('resultbox');
    const predLabel = document.querySelector('.pred-label');
    const explanationDiv = document.querySelector('.explanation');
    const probList = document.querySelector('.prob-list');
    const imgFile = imgInput.files[0];
    const prompt = document.getElementById('promptbox').value;

    if (!imgFile || !prompt.trim()) {
        predLabel.textContent = "Please provide both image and prompt.";
        if (explanationDiv) explanationDiv.textContent = "";
        resultBox.style.display = "block";
        return;
    }

    const btn = document.getElementById("classify");
    btn.disabled = true;
    predLabel.textContent = "Classifying, please wait...";
    if (explanationDiv) explanationDiv.textContent = "";
    probList.innerHTML = "";
    resultBox.style.display = "block";

    const fd = new FormData();
    fd.append('image', imgFile);
    fd.append('prompts', prompt);

    try {
        const resp = await fetch('http://127.0.0.1:5001/classify', { method: 'POST', body: fd });
        const data = await resp.json();

        if (!resp.ok) {
            predLabel.textContent = data.error || "Server error during classification.";
            console.error('Server error:', data);
        } else {
            // Display prediction and probabilities
            predLabel.textContent = `Best: ${data.prediction}`;
            if (data.explanation && explanationDiv) {
                explanationDiv.textContent = data.explanation;
            }
            probList.innerHTML = "";

            const probs = data.probabilities || [];
            for (const item of probs) {
                const li = document.createElement('li');
                li.className = 'prob-item';

                const label = document.createElement('div');
                label.className = 'prob-label';
                label.textContent = item.label;

                const barWrap = document.createElement('div');
                barWrap.className = 'prob-bar';
                const fill = document.createElement('div');
                fill.className = 'prob-fill';
                const pct = Math.round((item.probability || 0) * 100);
                // animate width
                requestAnimationFrame(() => { fill.style.width = pct + '%'; });

                barWrap.appendChild(fill);

                const value = document.createElement('div');
                value.className = 'prob-value';
                value.innerHTML = `<span>${pct}%</span> <span style="font-size:0.8em; opacity:0.7; margin-left:6px;">(Sim: ${item.similarity.toFixed(3)})</span>`;

                li.appendChild(label);
                li.appendChild(barWrap);
                li.appendChild(value);
                probList.appendChild(li);
            }
        }
    } catch (err) {
        console.error(err);
        predLabel.textContent = "Network error. Ensure the server is running and reachable.";
    } finally {
        btn.disabled = false;
    }
};
