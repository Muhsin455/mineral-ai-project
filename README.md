# MineralAI — AI for Mineral Exploration Using Satellite & Seismic Data

## What this project includes
- Streamlit interface for uploading a labelled CSV dataset
- Random Forest training with missing-value imputation
- Held-out test-set metrics: accuracy, weighted precision/recall/F1
- Confusion matrix, classification report, feature importance
- New-sample prediction form and CSV export

## Important honesty note
This is a working ML application scaffold, not a completed geological discovery system. It does not automatically download or process raw satellite imagery, and it does not include verified mineral labels. You must supply a real dataset and document its source. The app must not be described as using real satellite/seismic data until those data are actually included and validated.

## Dataset format
Upload a CSV with:
- One target/label column (e.g. a verified mineral occurrence class)
- One or more numeric feature columns extracted from real data
- Each row representing a labelled location/sample

Example column names could be `spectral_index`, `elevation_m`, `magnetic_feature`, `seismic_attribute`, `mineral_label`, but only use columns supported by your real sources. Do not invent values or labels.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Run:
   `streamlit run app.py`

## Deploy to Streamlit Community Cloud
1. Upload `app.py` and `requirements.txt` to your GitHub repository.
2. Open Streamlit Community Cloud and select the repository, branch, and `app.py`.
3. Deploy. Do not commit private credentials or passwords.

## Public data source leads
- ISRO Bhoonidhi: https://bhoonidhi.nrsc.gov.in/bhoonidhi/home.html
  The portal provides access to satellite products; account registration may be required for some downloads, and some products are priced/on-demand.
- NRSC Geosciences sensor information: https://www.nrsc.gov.in/nrscnew/Apps_geo_sensors.php
  This lists sensors used for mineral exploration, including optical and hyperspectral sources.
- National Geoscience Data Repository: https://geodataindia.gov.in/
  Official Indian geoscience repository. Some functions/data may require login or access approval.

## Scientific cautions
- A DEM/elevation product alone is not a mineral detector.
- Satellite images, seismic waveform data and mineral occurrence labels are different data types; do not treat them as interchangeable.
- Only use seismic data that actually corresponds to a relevant geophysical survey or a scientifically justified attribute.
- Do not report test metrics from a small or biased dataset as real-world accuracy.
- A high model score can be caused by leakage, duplicates, spatial autocorrelation or unbalanced labels.
- Predictions are decision-support only and require independent geological validation.

## Suggested next development
1. Finalize a study area only after confirming public mineral labels exist there.
2. Obtain real satellite data and matching labels.
3. Create a reproducible feature extraction notebook/pipeline.
4. Evaluate using a spatially separated test set.
5. Add a map only after data coordinates and CRS are validated.
