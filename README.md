# FAR-VLM Streamlit App

Interactive dashboard presenting the FAR-VLM thesis/paper (Israel Agyekum, MPhil Data
Science, University of Ghana): abstract, architecture, all results tables and figures,
faithfulness/ablation analysis, external validation, and qualitative analysis.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud (free)

1. Create a new GitHub repository (public or private) and push this entire `farvlm_app`
   folder to it (`app.py`, `requirements.txt`, and the `assets/` folder — all of it,
   the images and PDF are needed at runtime).

   ```bash
   cd farvlm_app
   git init
   git add .
   git commit -m "FAR-VLM Streamlit dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub
   account.

3. Click **"New app"**, choose your repository, branch `main`, and set the main file
   path to `app.py`.

4. Click **"Deploy"**. The first build takes 1-2 minutes. You'll get a public URL like
   `https://<your-app-name>.streamlit.app` that you can share with your supervisors or
   include in your thesis defence.

## Folder contents

- `app.py` — the Streamlit app
- `requirements.txt` — Python dependencies
- `assets/` — all figures (architecture diagram, training curves, per-pathology charts,
  attention heatmaps, failure cases) and the final MICCAI paper PDF (downloadable from
  the app's sidebar)

## Notes

- All numbers in the app are taken directly from the FAR-VLM MICCAI paper and its
  underlying result JSON files (Tables 1-6, `farvlm_results.json`, `experiment_summary.json`,
  `openi_validation.json`, `per_pathology_deletion.json`). Nothing is fabricated or
  interpolated.
- The per-pathology classification AUC-ROC chart (Results tab) is shown as the original
  image (`fig4_per_path_auc.png`) rather than a rebuilt data table, because the exact
  14-label AUC-ROC values were not saved to a data file — only three values (Pneumothorax
  0.9905, Pleural Effusion 0.9446, Fracture 0.8505) are stated explicitly in the paper text
  and are surfaced as callouts. If you still have your per-label evaluation logs, send them
  over and the app can be updated to show the exact full table.
