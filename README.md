# KT3 PDO + PSC Microwell Analyzer

A browser-based Streamlit application for processing large RGB fluorescence microscopy images of KT3 microwell arrays.

## What it does

- Detects fully visible microwells and indexes them by column,row.
- Detects green PDO-like objects and measures 2D equivalent circular diameter.
- Includes the conservative touching-PDO split used for the corrected KT3 analysis.
- Detects red PSC-like fluorescent foci in each well.
- Creates indexed large-image QC overlays.
- Saves raw and labelled crops of PDO-containing wells.
- Generates PDO-size and PSC-frequency plots.
- Exports CSV tables and a downloadable ZIP of the complete analysis.

## Interpretation

- Calibration uses the physical microwell diameter entered by the user; the default is 100 µm.
- PDO size is a 2D equivalent circular diameter from segmented projected green area, not a true 3D organoid diameter.
- PSC values are automated PSC-like fluorescent foci and should not be interpreted as definitive single-cell counts.
- Automated results should be visually QC'd before use in a thesis or publication.

## Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Create a new app from this repository.
3. Select the `main` branch.
4. Set the main file path to `app.py`.
5. Deploy.

The app processes uploaded microscopy images during the active Streamlit session and provides the results as a downloadable ZIP.