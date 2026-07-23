# Cattle Weight Estimator

A Streamlit web app that estimates live cattle body weight from a single
side-view photo, using two AI models:

- **Segmentation model** (`best.pt`) — locates the cow and the calibration sticker
- **Side keypoint model** (`best_model_side.pth`) — locates 15 side-view skeletal keypoints

Both models are downloaded automatically from Google Drive on first run and
cached to disk (`models/`), so they only download once per deployment.

## How it works

1. Enter the animal's **Tag ID** and upload a **side-view image** (the
   calibration sticker of known size must be visible in the frame).
2. The app detects the cow + sticker, calibrates pixels→cm, and localizes
   keypoints.
3. It computes:
   - `Linear_Body_Depth` (body_top ↔ body_bottom)
   - `Linear_Chest_Height` (chest_top ↔ chest_bottom)
   - `Body_Length` = `0.9334×raw_body_length + 0.4344×linear_body_depth − 7.75`
   - `Heart_Girth` = `1.588×linear_chest_height + 73.43`
   - **Weight (kg)** = `(Heart_Girth² × Body_Length) / 10840`
4. Weight is shown prominently on screen.
5. Sidebar:
   - **Output** — annotated image (Body Length + Heart Girth lines only),
     downloadable with the Tag ID in the filename.
   - **Logs** — running table of every animal tested (Tag ID, Date, Time,
     Linear Body Depth, Linear Chest Height, Body Length, Heart Girth,
     Weight), downloadable as CSV.

## Project structure

```
cattle_weight_app/
├── app.py                    # Streamlit UI
├── utils.py                  # Model loading, inference, trait/weight math, annotation
├── requirements.txt
├── packages.txt               # apt packages needed by OpenCV/Ultralytics on Streamlit Cloud
├── .streamlit/
│   └── config.toml            # White + #2A2866 theme
├── assets/
│   └── (put logo.png here)
├── models/                    # auto-downloaded on first run (gitignored)
└── logs/
    └── cattle_weight_logs.csv # created automatically, running log
```

## Before you deploy

1. **Add your logo**: drop a file at `assets/logo.png`. It shows automatically
   in the sidebar; if it's missing, a text placeholder is shown instead.
2. **Make sure the Google Drive files are shared as "Anyone with the link"**
   (Viewer access) — `gdown` cannot download files that require sign-in.
   - Segmentation model: `best.pt`
   - Side keypoint model: `best_model_side.pth`
3. Optionally add a `.gitignore` with:
   ```
   models/
   logs/
   __pycache__/
   ```
   so the large model weights and local logs aren't committed to git — they
   are re-downloaded / recreated automatically at runtime.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io → **New app**.
3. Select your repo, branch, and set **Main file path** to `app.py`.
4. Deploy. On first load, the app will download both models from Google
   Drive (this can take a minute or two) — subsequent reruns reuse the
   cached files for that session.

### Notes on Streamlit Cloud

- The free tier has limited RAM/CPU; a ResNet-101-backboned KeypointRCNN
  plus a YOLO segmentation model can be heavy. If you see out-of-memory
  errors, consider requesting more resources or trimming model size.
- Streamlit Cloud's filesystem is **ephemeral** — `logs/cattle_weight_logs.csv`
  persists only while the app instance stays awake. For permanent log
  storage across restarts, connect a database or cloud storage bucket
  (e.g. Google Sheets, S3, or a small hosted DB) and swap out the
  `logs_df` read/write logic in `app.py`.
