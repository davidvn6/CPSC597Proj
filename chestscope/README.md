# ChestScope

A local Flask interface for the CPSC597 pneumonia-detection project. Uses the
square-trained ResNet-50, epoch 8, and its full-precision validation-selected
threshold. The choice of main model is retained from development. Letterbox
remains an ablation, rather than being selected using test results.

## Setup in Windows PowerShell

1. Extract the ZIP. Place its **chestscope** folder inside **CPSC597Proj**.
2. Open a terminal in CPSC597Proj.
3. Install Flask in your existing environment:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r .\chestscope\requirements-app.txt
   ```

4. Start the app:

   ```powershell
   .\.venv\Scripts\python.exe .\chestscope\app.py
   ```

5. Open **http://127.0.0.1:5000**. Ctrl+C stops the server.

Use the existing virtual environment containing your working CUDA PyTorch,
torchvision, Pillow, and NumPy. Do not reinstall PyTorch for this app.

## Required existing project files

- `results/threshold_selection/20261008_182555_424842/selected_thresholds.json`
- The square ResNet-50 checkpoint referenced by that protocol (your run
  `results/resnet50/20261007_172849_403643/best_model.pt`).

Training scripts are not imported. The app reconstructs ResNet50CNN and
strictly loads its state dictionary. ImageNet weights are not downloaded.
To launch against a project at another path, set `CPSC597_PROJECT_ROOT` to
that folder before running. An updated research protocol is not selected
automatically: the file above is deliberately pinned.

## Interface

- Drag/drop or browse for JPG/JPEG/PNG, up to 12 MB.
- Prediction, pneumonia score, and recorded decision threshold.
- Original-image toggle, model input, positive Grad-CAM, and overlay.
- Overlay-strength slider and two accessible alternative color maps.
- Test accuracy with 95% interval, recall, specificity, F1, and ROC-AUC.
- Download current overlay PNG, result JSON, or a figures/result ZIP.
- Clear the current image, responsive layout, loading and validation states.

The downloaded ZIP contains the default Ember figures at 55% overlay strength.
The individual overlay PNG reflects the current slider/color-map settings.
Result JSON includes these display settings plus checkpoint/protocol hashes.

## Prediction and explanation

Preprocessing matches the square-trained model: EXIF orientation, grayscale,
three channels, antialiased square resize to 224 x 224, floating-point scaling,
and the checkpoint's ImageNet mean/std normalization. The displayed overlay
aligns to that square model input. The original-image toggle shows the uploaded
image's proportions but does not change inference.

Prediction and Grad-CAM use one full-precision forward pass. The input is
repeated to the recorded evaluation batch size to reduce the batch-size-related
score drift observed during earlier single-image Grad-CAM. Only the first
image's pneumonia logit is differentiated. Grad-CAM uses `network.layer4`,
spatially averaged gradients, a weighted feature sum, ReLU, bilinear upsampling,
and per-image normalization. Color maps visualize relative positive contribution
only. They do not explain negative evidence for NORMAL or verify disease location.

The app loads the threshold from the frozen protocol, not a rounded number
typed into the app. It verifies the model's SHA-256 hash and checkpoint epoch.
No threshold sliders or test-based model switching are provided.

## Metrics provenance

`model_metrics.json` is a bundled snapshot of your final held-out results for
**ResNet50_square / validation_selected**. Its accuracy is 96.22%, recall
98.11%, specificity 91.18%, F1 0.9742, ROC-AUC 0.9929. These measure performance
on 874 test images, not accuracy for a newly uploaded image. The 95% accuracy
interval is 94.80–97.40%, from 2,000 candidate-group bootstrap replicates.
Candidate groups are not verified patients. Model training and threshold-selection
variability are not included in these intervals. Neither the app nor these
metrics establishes clinical suitability or generalization beyond this dataset.

## Local processing

Uploads and result archives are processed in memory. The app does not write
images to disk or keep an analysis history. Closing/reloading the page clears
the displayed result; browser downloads are saved only when you choose them.
Use de-identified research images. JPG/PNG validity does not establish that an
image is a chest X-ray; the app cannot reliably reject out-of-domain images.
The dataset used for this project is pediatric, and adult generalization is
unverified. The app is a research demonstration, not a diagnostic tool.

The server binds to 127.0.0.1, with debug/reloader disabled. It is intended for
your local project demonstration, not public hosting.

## Verification

`python chestscope/check_app.py` checks file validation, API responses,
oversized requests, no-model handling, and in-memory result downloads using a
clearly identified fake inference engine. It never reads the held-out dataset.
The real checkpoint is available on your computer, so actual checkpoint
inference must be checked there. Upload a de-identified research X-ray and
confirm the three figures, label, threshold, controls, and downloads appear.

## Troubleshooting

- **No module named flask:** use the `.venv` Python commands above.
- **Missing threshold protocol/checkpoint:** keep chestscope inside your project
  folder and preserve the results files at the paths above.
- **Checkpoint hash differs:** use the checkpoint frozen in the protocol;
  do not bypass the integrity check.
- **Address already in use:** stop the earlier app with Ctrl+C.
- **GPU memory error:** stop other training jobs before running the app. You
  can use CPU by setting `$env:CUDA_VISIBLE_DEVICES = "-1"` before launching.
- **Model says NORMAL:** this is a class prediction, not a statement that the
  person is healthy or that disease has been excluded.

## Git

Commit app code and the small metrics snapshot. Keep large checkpoints and
datasets out of ordinary Git tracking. No checkpoint is included in this ZIP.
