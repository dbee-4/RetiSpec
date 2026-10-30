// ── UI Elements ──────────────────────────────────────────────────────────────
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const previewContainer = document.getElementById('previewContainer');
const previewImg = document.getElementById('previewImg');
const loader = document.getElementById('loader');
const resultsSection = document.getElementById('results');

// Severity-based color mapping
const SEVERITY_COLORS = {
    'No DR': '#22c55e',
    'Mild': '#eab308',
    'Moderate': '#f97316',
    'Severe': '#ef4444',
    'Proliferative': '#a855f7'
};

// ── File Handling ────────────────────────────────────────────────────────────

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', function() {
    if (this.files && this.files[0]) {
        handleFile(this.files[0]);
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewContainer.classList.remove('hidden');
        resultsSection.classList.add('hidden'); // Reset results on new upload
    };
    reader.readAsDataURL(file);
}

// ── Inference Call ───────────────────────────────────────────────────────────

analyzeBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) {
        alert("Please select or drop a retinal fundus image first.");
        return;
    }

    // UI State: Loading
    analyzeBtn.disabled = true;
    loader.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    const formData = new FormData();
    formData.append('image', file);
    formData.append('use_tta', true);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.detail || data.error || "Inference failed");
        }

        displayResults(data);
    } catch (err) {
        alert("Analysis failed: " + err.message);
    } finally {
        loader.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
});

// ── Result Rendering ─────────────────────────────────────────────────────────

function displayResults(data) {
    const color = SEVERITY_COLORS[data.prediction] || '#38bdf8';

    // 1. Update Banner
    document.getElementById('resultLabel').textContent = data.prediction;
    document.getElementById('resultLabel').style.color = color;
    document.getElementById('resultDescription').textContent = data.description;
    if (document.getElementById('englishExplanation')) {
        document.getElementById('englishExplanation').textContent = data.explanation;
    }
    document.getElementById('resultConfidence').textContent = `${data.confidence}%`;
    document.getElementById('resultBanner').style.borderLeftColor = color;

    // 2. Update Explainable AI Overlays
    document.getElementById('origImg').src = `data:image/png;base64,${data.original_img}`;
    document.getElementById('camImg').src = `data:image/png;base64,${data.gradcam_img}`;
    
    // Support both new gradient_input_img and backwards-compatible shap_img
    const gradXInputElem = document.getElementById('gradXInputImg') || document.getElementById('shapImg');
    const gradXInputData = data.gradient_input_img || data.shap_img;
    if (gradXInputElem && gradXInputData) {
        gradXInputElem.src = `data:image/png;base64,${gradXInputData}`;
    }

    // 3. Update Probability Distribution Bars
    const probContainer = document.getElementById('probBars');
    probContainer.innerHTML = ''; // Clear old bars

    Object.entries(data.all_probs).forEach(([label, value]) => {
        const barHtml = `
            <div class="prob-item">
                <div class="prob-meta">
                    <span>${label}</span>
                    <span>${value}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: 0%; background-color: ${SEVERITY_COLORS[label] || '#3b82f6'}"></div>
                </div>
            </div>
        `;
        probContainer.insertAdjacentHTML('beforeend', barHtml);
    });

    // Trigger bar animation after short rendering delay
    setTimeout(() => {
        const fills = document.querySelectorAll('.prob-bar-fill');
        Object.values(data.all_probs).forEach((value, idx) => {
            if (fills[idx]) {
                fills[idx].style.width = `${value}%`;
            }
        });
    }, 100);

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}
