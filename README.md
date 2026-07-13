# Oil Spill Detection & Classification — Streamlit App

A Streamlit web app that uses two transformer-based segmentation models —
**SegFormer-B2** and **Mask2Former (Swin-Small)** — to detect and classify
oil spills in aerial/satellite imagery (LADOS dataset classes: Background,
Oil, Emulsion, Sheen, Ship, Oil-platform).

This repo contains **only the inference app**. Model training happens
separately (see `training_fixed.ipynb`, run on Kaggle) and the resulting
checkpoints are hosted on the Hugging Face Hub, not in this repo.

## Repo contents

```
oil-spill-streamlit/
├── app.py             # Streamlit UI
├── model_utils.py     # Model loading + inference logic
├── requirements.txt
├── .gitignore
└── README.md
```

## One-time setup before deploying

### 1. Train the models (if you haven't already)

Run `training_fixed.ipynb` on Kaggle (GPU enabled). It downloads LADOS,
trains SegFormer-B2 and Mask2Former, and saves the best checkpoints to
`/kaggle/working/`.

You'll need two Kaggle secrets set up first (Notebook menu → Add-ons → Secrets):
- `ROBOFLOW_API_KEY` — your Roboflow API key
- `HF_TOKEN` — a Hugging Face **write** access token (create one at
  https://huggingface.co/settings/tokens)

### 2. Upload checkpoints to Hugging Face Hub

The last cell of `training_fixed.ipynb` does this automatically — it
creates a model repo (e.g. `your-username/oil-spill-segformer-mask2former`)
and uploads both `.pth` files there.

If you'd rather do it manually:
```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload your-username/oil-spill-segformer-mask2former \
    segformer_lados_best.pth segformer_lados_best.pth
huggingface-cli upload your-username/oil-spill-segformer-mask2former \
    mask2former_lados_best.pth mask2former_lados_best.pth
```

### 3. Point this app at your Hugging Face repo

Open `model_utils.py` and change:
```python
HF_REPO_ID = "YOUR_HF_USERNAME/oil-spill-segformer-mask2former"
```
to your actual Hugging Face repo ID.

> **Architecture note:** `model_utils.py` builds Mask2Former with
> **Swin-Small** dimensions to match the checkpoint-loading code that was
> already verified to work. If you trained with Swin-Base instead, update
> `M2F_CKPT` and remove the manual `backbone_config` overrides in
> `load_mask2former()`.

If your Hugging Face model repo is **private**, the app also needs a read
token at runtime — add `HF_TOKEN` as a secret in Streamlit Cloud (see step 5)
and `huggingface_hub` will pick it up automatically from the environment.

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/oil-spill-streamlit.git
cd oil-spill-streamlit
pip install -r requirements.txt
streamlit run app.py
```

This will open `http://localhost:8501` in your browser. The first run
downloads both checkpoints from Hugging Face Hub (cached afterwards).

## Deploy to Streamlit Community Cloud

1. Push this folder to a **new GitHub repository**:
   ```bash
   cd oil-spill-streamlit
   git init
   git add .
   git commit -m "Initial commit: oil spill detection Streamlit app"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/oil-spill-streamlit.git
   git push -u origin main
   ```
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"** → select your repo, branch `main`, main file path `app.py`.
4. Click **Deploy**.
5. If your Hugging Face model repo is private, go to your app's
   **Settings → Secrets** in Streamlit Cloud and add:
   ```toml
   HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
   ```
6. Your app will be live at `https://YOUR_USERNAME-oil-spill-streamlit.streamlit.app`
   (or similar) within a minute or two. No need to keep anything running on
   your own machine — Streamlit Cloud hosts it continuously.

## Notes

- The app downloads model weights at startup and caches them with
  `@st.cache_resource`, so they only download once per app instance, not
  per visitor.
- CPU inference works but is slower than GPU; Streamlit Community Cloud's
  free tier is CPU-only, so expect a few seconds per prediction.
- Never commit `.pth` checkpoint files or API keys/tokens directly into this
  repo — GitHub will reject files over 100MB, and committing secrets is a
  security risk even in private repos.
