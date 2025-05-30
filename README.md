# RoMT: Robust Metamorphic Testing for LLM Hallucination Detection

This project implements the **RoMT (Robust Metamorphic Testing)** framework for detecting hallucinations in Large Language Models (LLMs), as described in the research paper. The system leverages metamorphic testing to analyze relative confidence across multiple activation paths in LLMs to identify hallucinated outputs without requiring manual annotation or external resources.

## Project Structure

### Code Modules
- **`data_processing.py`**:  
  Processes raw question data (`data.xlsx`) and generates multiple datasets using different metamorphic relations:
  - Original English answers (`sourcenewdata.csv`)
  - Spanish translations (`esnewdata.csv`)
  - Italian translations (`itnewdata.csv`)
  - Improved questions via GPT-3.5 (`renewdata.csv`)
  - Answers using a specialized GA prompt (`GAnewdata.csv`)

- **`consistency.py`**:  
  Checks consistency between the original answers (English) and transformed answers (translations/improvements):
  - Detects uncertainty markers ("I don't know")
  - Measures answer consistency
  - Computes confidence score differences

- **`main.py`**:  
  Main pipeline orchestrator:
  1. Processes input data
  2. Runs consistency checks
  3. Loads pre-trained GBDT model
  4. Generates final predictions (`new_predictions.csv`)

### Data Folders
- **`data/`**:  
  Contains all input/output datasets:
  - Raw input questions (`data.xlsx`)
  - Processed datasets (e.g., `sourcenewdata.csv`)
  - Consistency results (`consistency_results.csv`)
  - Final predictions (`new_predictions.csv`)

- **`model/`**:  
  Stores the pre-trained hallucination classifier:
  - GBDT model trained on HaluEval dataset (`gbdt_model.joblib`)

### Configuration
- **`requirements.txt`**:  
  Python dependencies (pandas, transformers, openai, etc.)

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place input data in `data/data.xlsx`

3. Set OpenAI API key in `data_processing.py` and `consistency.py`:
   ```python
   openai.api_key = 'your_api_key_here'
   ```

4. Run main pipeline:
   ```bash
   python main.py
   ```
