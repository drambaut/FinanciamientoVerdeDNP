# Green Finance Tracking – Streamlit Zero-Shot Classifier

This project is a Streamlit web application that helps the Departamento Nacional de Planeación (DNP) classify investment projects according to green finance taxonomies. It leverages a zero-shot multilingual `mDeBERTa` model to assign labels for the **Module**, **Climate Change (CC)**, **Disaster Risk Management (GRD)**, and **Biodiversity (BIO)** domains. The app accepts an Excel file containing project descriptions, runs cascaded classifiers, and provides categorized outputs ready for download.

## Key Features
- Interactive Streamlit interface styled with DNP branding.
- Excel upload (`.xlsx`) with validation for required columns (`bpin`, `texto`).
- Optional “Module” pre-classification switch to gate downstream classifiers.
- Zero-shot classification using Hugging Face transformers (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`).
- Hierarchical categorization:
  - Module → assigns Biodiversity, Climate Change, or Disaster Risk & Desasters.
  - CC & GRD → multi-level taxonomies (Category 1–3) with probability exports.
  - BIO → enriched outputs with Kunming-Montreal, CDB, and PAB mappings.
- Automatic export of results to `exports/<input_filename>/` with download buttons in the UI.

## Project Structure
- `app.py`: Streamlit entrypoint with UI, file handling, and result visualization.
- `modelo.py`: Processing pipeline with zero-shot classifiers and export helpers.
- `requirements.txt`: Python dependencies (Streamlit, pandas, transformers, torch, etc.).
- `exports/`: Auto-generated folder storing per-run Excel outputs.

## Prerequisites
- Python **3.10** or **3.11** (recommended for compatibility with `torch<2.5`).
- Virtual environment tool of your choice (`venv`, `conda`, or `pyenv`).
- Git LFS if large model artifacts are added manually (optional).

## Setup
```bash
# Clone the repository
git clone https://github.com/<your-org>/rastreo_financiamiento_verde.git
cd rastreo_financiamiento_verde

# (Recommended) create a Python 3.11 environment
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** If Python 3.11 is not available, install it via `pyenv` or `conda`. PyTorch wheels targeted by the project are not currently built for Python 3.13+ on macOS ARM.

## Running the App
```bash
source .venv/bin/activate  # ensure the environment is active
streamlit run app.py
```

Then open the provided localhost URL in your browser.

## Using the Classifier
1. Prepare an Excel file (`.xlsx`) with at least the columns:
   - `bpin`: Project identifier.
   - `texto`: Project description (Spanish text supported).
2. Launch the app and upload the file via “Cargar archivo Excel”.
3. (Optional) Enable “Procesar 'MÓDULO' antes de las ramas” to run the gating classifier.
4. Trigger one or more classification buttons:
   - `Módulo`
   - `GRD (Gestión de Riesgo de Desastres)`
   - `BIO (Biodiversidad)`
   - `CC (Cambio Climático)`
5. Review the interactive tables and download the generated Excel files from the UI.
6. All outputs are stored under `exports/<input_filename>/`.

## Outputs
- `modulo_results.xlsx` (+ `_mayor`) – Module-level probabilities and winning labels.
- `CC_cat{1,2,3}_results.xlsx` (+ `_mayor`) – Climate Change taxonomies.
- `GRD_cat{1,2,3}_results.xlsx` (+ `_mayor`) – Disaster Risk categories.
- `BIO_cat1_results.xlsx` – Biodiversity classification with extended mappings.
- Additional “_mayor” files include reduced columns focused on dominant labels.

## Model Notes
- The zero-shot pipeline runs per text per label, so large spreadsheets can take time.
- Set `clave = "NO"` in the UI if you already filtered rows by Module externally.
- Thresholding is handled inside `modelo.py` with percentile- or score-based binarization, depending on label count.
- GPU acceleration is supported when CUDA is available.

## Troubleshooting
- **Torch install fails on macOS ARM + Python 3.13:** use Python 3.11 (via `pyenv`, `conda`, etc.).
- **Out-of-memory:** process the Excel in batches or ensure the machine has enough RAM.
- **Empty outputs:** check that the uploaded file has the required columns and non-empty text.

## Roadmap Ideas
- Parameterize thresholds via the UI.
- Add batch progress indicators and processing statistics.
- Integrate domain-specific validation rules before exporting results.

## License
See `LICENSE` for details.