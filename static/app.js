const fileInput = document.getElementById('fileInput');
const browseButton = document.getElementById('browseButton');
const dropZone = document.getElementById('dropZone');
const dropContent = document.getElementById('dropContent');
const previewWrap = document.getElementById('previewWrap');
const preview = document.getElementById('preview');
const fileName = document.getElementById('fileName');
const changeImage = document.getElementById('changeImage');
const analyzeButton = document.getElementById('analyzeButton');
const resultContent = document.getElementById('resultContent');
const results = document.getElementById('results');
const statusPill = document.getElementById('statusPill');
const statusText = document.getElementById('statusText');
let selectedFile = null;

function setStatus(state, text) {
  statusPill.className = `status-pill ${state}`;
  statusText.textContent = text;
}

function chooseFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) return showError('Please choose an image file.');
  if (file.size > 12 * 1024 * 1024) return showError('This image is larger than 12 MB.');
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  fileName.textContent = file.name;
  dropContent.hidden = true;
  previewWrap.hidden = false;
  analyzeButton.disabled = false;
  results.hidden = true;
  resultContent.hidden = false;
  resultContent.querySelector('h2').textContent = 'Ready for a closer look?';
  resultContent.querySelector('p').textContent = 'Your image is in. Let’s see what stands out.';
  analyzeButton.querySelector('span:first-child').textContent = 'Analyze image';
  setStatus('', 'IMAGE LOADED');
}

function showError(message) {
  resultContent.hidden = true;
  results.hidden = false;
  results.innerHTML = `<p class="error-message">${message}</p>`;
  setStatus('', 'NEEDS ATTENTION');
}

browseButton.addEventListener('click', () => fileInput.click());
changeImage.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', event => chooseFile(event.target.files[0]));
['dragenter', 'dragover'].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault(); dropZone.classList.add('dragover');
}));
['dragleave', 'drop'].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault(); dropZone.classList.remove('dragover');
}));
dropZone.addEventListener('drop', event => chooseFile(event.dataTransfer.files[0]));

analyzeButton.addEventListener('click', async () => {
  if (!selectedFile) return;
  analyzeButton.disabled = true;
  analyzeButton.querySelector('span:first-child').textContent = 'Taking a look…';
  setStatus('busy', 'READING IMAGE');
  const form = new FormData();
  form.append('image', selectedFile);
  try {
    const response = await fetch('/predict', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The image could not be analyzed.');
    renderResults(data.predictions);
    setStatus('done', 'READ COMPLETE');
  } catch (error) {
    showError(error.message || 'Something went wrong. Please try again.');
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.querySelector('span:first-child').textContent = 'Analyze again';
  }
});

function renderResults(predictions) {
  resultContent.hidden = true;
  results.hidden = false;
  const [winner, ...rest] = predictions;
  results.innerHTML = `<div class="winner-label">OUR BEST READ</div><div class="winner">${winner.class}</div>${[winner, ...rest].map((item, index) => `<div class="match"><span>${item.class}</span><div class="match-bar"><i style="width:${item.confidence}%"></i></div><strong>${item.confidence}%</strong></div>`).join('')}`;
}
