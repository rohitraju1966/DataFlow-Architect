# DataFlow Architect

**DataFlow Architect** is a Streamlit application that leverages AI to automatically generate comprehensive data science workflows and reports. Upload your dataset or document, choose your desired mode, and let the AI guide you through creating a full report or step-by-step workflow. View the app here: https://dataflow-architect-chpwn2xcy4yyarx8azsjct.streamlit.app/


---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Setting Up Environment Variables](#setting-up-environment-variables)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [License](#license)

---

## Features

- **Comprehensive Report Generation:**  
  Generate a full, AI-assisted report detailing data preprocessing, exploratory analysis, machine learning suggestions, and more.

- **Step-by-Step Guidance:**  
  Receive targeted, step-by-step instructions for specific workflow sections.

- **PDF Export:**  
  Download the generated report or guidance as a PDF.

- **Interactive & User-Friendly:**  
  Easily upload your dataset or document, select your mode, and follow clear prompts.

---

## Requirements

The application requires the following Python libraries:

```
streamlit
requests
pandas
markdown
WeasyPrint
python-dotenv
openai
mistralai
```

A complete `requirements.txt` file is included in the repository. You also need WeasyPrint and its dependencies (e.g., cairo, pango, libffi) installed on your system.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/<your-username>/DataFlow-Architect.git
   cd DataFlow-Architect
   ```

2. **Create a Virtual Environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or on Windows:
   venv\\Scripts\\activate
   ```

3. **Install the Required Libraries:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install WeasyPrint Dependencies:**  
   - On macOS with Homebrew:
     ```bash
     brew install cairo pango gdk-pixbuf libffi
     ```
   - On Ubuntu/Debian:
     ```bash
     sudo apt-get update
     sudo apt-get install -y libcairo2 libpango-1.0-0 libffi-dev
     ```
   - Or via conda:
     ```bash
     conda install -c conda-forge weasyprint
     ```

---

## Setting Up Environment Variables

This application uses OpenAI and Mistral APIs. Store your API keys in a local `.env` file **without** committing it to version control.

1. **Create a `.env` file** in the project root:

   ```bash
   touch .env
   ```

2. **Add your API keys** in the following format:

   ```dotenv
   OPENAI_API_KEY=sk-<your_openai_api_key>
   MISTRAL_API_KEY=mk-<your_mistralai_api_key>
   ```

3. **Ensure `.env` is ignored** by adding it to your `.gitignore`.

---

## Running the Application

1. **Activate your virtual environment** (if using one).

2. **Run the Streamlit app:**

   ```bash
   streamlit run Data_flow.py
   ```

3. **Open the app** in your browser (typically at [http://localhost:8501](http://localhost:8501)).

---

## Usage

1. **Select Your Mode:**
   - **Report Mode:**  
     Upload a CSV file and click "Generate My Report" to produce a comprehensive AI-assisted report. Then switch to the Report tab to view and download the report.
   - **Step-by-Step Guidance:**  
     Upload your dataset or document, select a workflow section using the clickable cards, and click "Proceed with Selection" to generate step-by-step guidance. Then switch to the Workflow tab to view and download the combined output.

2. **Download Options:**  
   Both modes allow you to download the generated output as a PDF.

---

**Happy DataFlowing!**
