/* All images and results stay in this page's memory until reset or reload. */
const $ = id => document.getElementById(id);
let selectedFile = null, current = null, grayPixels = null, heatValues = null;
let showingOriginal = false, busy = false;
const palettes = {
  ember: [[18,23,57],[50,79,156],[24,160,182],[253,194,87],[244,79,57]],
  viridis: [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]]
};

function error(message) {
  $('error').textContent = message;
  $('error').classList.toggle('hidden', !message);
}
function clearResult() {
  current = null; grayPixels = null; heatValues = null; showingOriginal = false;
  $('results').classList.add('hidden'); $('empty-state').classList.remove('hidden');
  $('input-image').removeAttribute('src');
  for (const id of ['heatmap-canvas','overlay-canvas']) {
    $(id).getContext('2d').clearRect(0,0,224,224);
  }
  $('status').classList.remove('loading');
  $('status').replaceChildren(Object.assign(document.createElement('i')), document.createTextNode('Awaiting image'));
}
function selectFile(file) {
  if (busy) return;
  clearResult(); error(''); selectedFile = null;
  $('file-info').classList.add('hidden'); $('analyze-button').disabled = true;
  if (!file) return reset();
  if (!/\.(jpe?g|png)$/i.test(file.name)) return error('Choose a JPG, JPEG, or PNG image.');
  if (!file.size || file.size > 12 * 1024 * 1024) return error('Choose a nonempty image smaller than 12 MB.');
  selectedFile = file;
  $('file-info').classList.remove('hidden');
  $('file-name').textContent = file.name;
  $('file-details').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · ready to analyze`;
  $('analyze-button').disabled = false;
}
function reset() {
  selectedFile = null; $('image-file').value = '';
  $('file-info').classList.add('hidden'); $('analyze-button').disabled = true;
  clearResult(); error('');
}
function color(value, palette) {
  const anchors = palettes[palette], scaled = value / 255 * (anchors.length-1);
  const index = Math.min(Math.floor(scaled), anchors.length-2), fraction = scaled-index;
  return anchors[index].map((v,c) => Math.round(v+(anchors[index+1][c]-v)*fraction));
}
function renderMaps() {
  if (!grayPixels || !heatValues) return;
  const alpha = Number($('opacity').value) / 100, palette = $('palette').value;
  $('opacity-value').textContent = `${Math.round(alpha*100)}%`;
  const heatContext = $('heatmap-canvas').getContext('2d');
  const overlayContext = $('overlay-canvas').getContext('2d');
  const heat = heatContext.createImageData(224,224), overlay = overlayContext.createImageData(224,224);
  for (let i=0; i<heatValues.length; i++) {
    const rgb = color(heatValues[i],palette), weight = alpha*heatValues[i]/255;
    for (let c=0; c<3; c++) {
      heat.data[i*4+c] = rgb[c];
      overlay.data[i*4+c] = Math.round(grayPixels[i*4+c]*(1-weight)+rgb[c]*weight);
    }
    heat.data[i*4+3] = overlay.data[i*4+3] = 255;
  }
  heatContext.putImageData(heat,0,0); overlayContext.putImageData(overlay,0,0);
  // Use a data image for the legend to respect the page's no-inline-style CSP.
  const legend = document.createElement('canvas'); legend.width = 100; legend.height = 1;
  const ctx = legend.getContext('2d'), pixels = ctx.createImageData(100,1);
  for (let i=0;i<100;i++) { const rgb=color(i*255/99,palette); pixels.data.set([...rgb,255],i*4); }
  ctx.putImageData(pixels,0,0);
  $('legend-gradient').replaceChildren();
  const image = new Image(); image.src=legend.toDataURL(); image.width=65; image.height=6;
  $('legend-gradient').append(image);
}
function loadImage(src) {
  return new Promise((resolve,reject) => { const image=new Image(); image.onload=()=>resolve(image); image.onerror=reject; image.src=src; });
}
function download(blob,filename) {
  const url=URL.createObjectURL(blob), link=document.createElement('a');
  link.href=url; link.download=filename; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function fromBase64(value) {
  return Uint8Array.from(atob(value),character=>character.charCodeAt(0));
}
$('image-file').addEventListener('change', event=>selectFile(event.target.files[0]));
$('remove-file').addEventListener('click',reset);
const zone=$('dropzone');
for(const type of ['dragenter','dragover']) zone.addEventListener(type,event=>{event.preventDefault();zone.classList.add('dragging');});
for(const type of ['dragleave','drop']) zone.addEventListener(type,event=>{event.preventDefault();zone.classList.remove('dragging');});
zone.addEventListener('drop',event=>selectFile(event.dataTransfer.files[0]));
$('opacity').addEventListener('input',renderMaps);
$('palette').addEventListener('change',renderMaps);
$('toggle-input').addEventListener('click',()=>{
  if(!current) return;
  showingOriginal=!showingOriginal;
  $('input-image').src=showingOriginal?current.original:current.model_input;
  $('input-image').alt=showingOriginal?'Uploaded grayscale chest X-ray':'Square-resized grayscale model input';
  $('toggle-input').textContent=showingOriginal?'Show model input':'Show original';
  $('input-caption').textContent=showingOriginal?`Original proportions · ${current.result.original_width} × ${current.result.original_height}`:'Square-resized model input · 224 × 224';
});
$('download-overlay').addEventListener('click',()=>{
  if(current) $('overlay-canvas').toBlob(blob=>download(blob,'pneumonia_detector_overlay.png'),'image/png');
});
$('download-json').addEventListener('click',()=>{
  if(current) download(new Blob([JSON.stringify({...current.result,display_settings:{palette:$('palette').value,overlay_strength:Number($('opacity').value)/100}},null,2)],{type:'application/json'}),'pneumonia_detector_result.json');
});
$('download-bundle').addEventListener('click',()=>{
  if(current) download(new Blob([fromBase64(current.archive)],{type:'application/zip'}),'pneumonia_detector_figures_and_result.zip');
});
$('analyze-button').addEventListener('click',async()=>{
  if(!selectedFile||busy) return;
  busy=true; error(''); clearResult();
  $('analyze-button').disabled=true; $('remove-file').disabled=true; $('image-file').disabled=true;
  $('analyze-label').textContent='Analyzing X-Ray…'; $('status').classList.add('loading');
  $('status').replaceChildren(document.createElement('i'),document.createTextNode('Generating Grad-CAM'));
  try {
    const form=new FormData(); form.append('image',selectedFile);
    const response=await fetch('/api/analyze',{method:'POST',body:form});
    const payload=await response.json(); if(!response.ok) throw new Error(payload.error||'Analysis failed.');
    const image=await loadImage(payload.model_input), temp=document.createElement('canvas');
    temp.width=temp.height=224; const ctx=temp.getContext('2d');ctx.drawImage(image,0,0);
    grayPixels=ctx.getImageData(0,0,224,224).data;
    heatValues=fromBase64(payload.heatmap_values); current=payload;
    const result=payload.result, positive=result.prediction==='PNEUMONIA';
    $('prediction-banner').classList.toggle('normal',!positive);
    $('prediction-label').textContent=positive?'Pneumonia class':'Normal class';
    $('prediction-description').textContent=positive?'The model classifies this image as pneumonia.':'The model classifies this image as normal. This does not rule out disease.';
    $('prediction-score').textContent=result.pneumonia_score.toFixed(2);
    $('input-image').src=payload.model_input;
    $('toggle-input').textContent='Show original';
    $('input-caption').textContent='Square-resized model input · 224 × 224';
    $('opacity').value=55; $('palette').value='ember';renderMaps();
    $('result-note').textContent=result.zero_positive_map?'No positive Grad-CAM map was obtained. A blank map does not mean there is no disease.':'This heatmap explains the pneumonia logit, even for a normal prediction. Colors are normalized within this image; they do not show disease severity or a verified disease location.';
    $('image-meta').textContent=`${result.filename} · ${result.original_width} × ${result.original_height}`;
    $('time-meta').textContent=`${result.inference_seconds.toFixed(2)} s`;
    $('empty-state').classList.add('hidden'); $('results').classList.remove('hidden');
    $('status').replaceChildren(document.createElement('i'),document.createTextNode('Analysis complete'));
  } catch(exception) { clearResult(); error(exception.message||'Could not connect to the local app.'); }
  finally {busy=false;$('analyze-button').disabled=!selectedFile;$('remove-file').disabled=false;$('image-file').disabled=false;$('analyze-label').textContent='Analyze X-Ray and Predict';$('status').classList.remove('loading');}
});

async function initialize() {
  try {
    const response=await fetch('/api/model'); if(!response.ok) throw new Error('Cannot load model information.');
    const data=await response.json(), m=data.metrics;
    $('accuracy').replaceChildren(document.createTextNode((m.accuracy*100).toFixed(2)),Object.assign(document.createElement('span'),{textContent:'%'}));
    $('accuracy-ci').textContent=`95% CI: ${(m.accuracy_ci[0]*100).toFixed(2)}–${(m.accuracy_ci[1]*100).toFixed(2)}%`;
    $('recall').textContent=(m.recall*100).toFixed(2)+'%';
    $('specificity').textContent=(m.specificity*100).toFixed(2)+'%';
    $('auc').textContent=m.roc_auc.toFixed(2);$('f1').textContent=m.f1.toFixed(2);
    $('model-threshold').textContent=data.threshold===null?'Unavailable':data.threshold.toFixed(2);
$('device').textContent='Model Trained On: NVIDIA GeForce RTX 3050 6GB Laptop GPU';
    if(!data.ready) error('Interface preview only. Run app.py inside CPSC597Proj to load your model.');
  } catch(exception) {error(exception.message);}
}
initialize();
