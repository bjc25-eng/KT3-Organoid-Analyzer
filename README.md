# KT3 PDO + PSC Microwell Analyzer

A browser-based Streamlit application for processing large microscopy image sets from AWS S3 or direct browser upload.

## Current architecture

- **GitHub** stores the application code.
- **Streamlit Community Cloud** hosts the browser app.
- **AWS S3** stores large microscopy inputs and can persist complete result packages.
- `analysis_core.py` retains the validated KT3 segmentation/measurement logic.
- `processor.py` adds S3 I/O and one-PDO-per-crop export without changing the scientific thresholds.
- `app.py` is the S3-backed single/batch analyzer.
- `pages/02_Longitudinal_Experiment.py` preserves the existing longitudinal multi-condition / ML-export workflow.

## What it does

- Detects fully visible microwells and indexes them by column,row.
- Detects green PDO-like objects and measures 2D equivalent circular diameter.
- Includes the conservative touching-PDO split used for the corrected KT3 analysis.
- Detects red PSC-like fluorescent foci in each well.
- Creates indexed large-image QC overlays.
- Creates well-centred crops and **one PDO-centred crop per automatically detected PDO**.
- Generates PDO-size and PSC-frequency plots.
- Exports CSV tables and a downloadable ZIP.
- Can write the complete result tree and ZIP back to S3.

## Interpretation

- Calibration uses the physical microwell diameter entered by the user; the default is 100 µm.
- PDO size is a 2D equivalent circular diameter from segmented projected green area, not a true 3D organoid diameter.
- PSC values are automated PSC-like fluorescent foci and should not be interpreted as definitive single-cell counts.
- Automated results should be visually QC'd before use in a thesis or publication.
- Do not mix a fresh automated rerun with a manually QC'd thesis dataset if their object counts differ.

## Streamlit Secrets for AWS

In Streamlit Community Cloud, add these under **App settings → Secrets**. Do not commit real AWS keys to GitHub.

```toml
AWS_ACCESS_KEY_ID = "..."
AWS_SECRET_ACCESS_KEY = "..."
AWS_DEFAULT_REGION = "eu-west-2"
S3_BUCKET = "your-bucket-name"
S3_INPUT_PREFIX = "uploads/"
S3_OUTPUT_PREFIX = "results/"
```

`AWS_SESSION_TOKEN` may also be supplied if temporary AWS credentials are used.

The IAM identity used by the app needs permission to list/read the input objects and write result objects in the selected bucket/prefixes.

## Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Create a new app from this repository.
3. Select the `main` branch.
4. Set the main file path to `app.py`.
5. Add the AWS values above under Streamlit Secrets.
6. Deploy.

The main page is the S3-backed single/batch analyzer. The longitudinal experiment workflow appears as a second page in the Streamlit page menu.
