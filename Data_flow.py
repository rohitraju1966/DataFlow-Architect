import streamlit as st
import requests
import pandas as pd
import time
import markdown
from weasyprint import HTML

# Import your backend class
from prompt import ReportGenerator

# Instantiate the backend processor
rg = ReportGenerator()

# ------------------ TYPEWRITER EFFECT FUNCTION ------------------
def typewriter_display(text, speed=0.001):  # faster speed (approx 30 sec for long text)
    placeholder = st.empty()
    display_text = ""
    for char in text:
        display_text += char
        placeholder.markdown(display_text)
        time.sleep(speed)

# ------------------ FUNCTION: Convert Markdown to PDF using WeasyPrint ------------------
def convert_md_to_pdf(md_text: str) -> bytes:
    """
    Convert a markdown string to PDF bytes using WeasyPrint.
    """
    html_text = markdown.markdown(md_text)
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; padding: 20px; }}
            h1, h2, h3 {{ color: #2C3E50; }}
            p {{ font-size: 0.95rem; }}
        </style>
    </head>
    <body>
        {html_text}
    </body>
    </html>
    """
    pdf_bytes = HTML(string=html_content).write_pdf()
    return pdf_bytes

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="DataFlow Architect", page_icon="⚡", layout="wide")

# ------------------ CUSTOM UI STYLES ------------------
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            color: #2C3E50;
        }
        .subheader {
            font-size: 1.5rem;
            color: #34495E;
        }
        .poll-container {
            text-align: center;
            margin-bottom: 20px;
        }
        /* 
         * Cards Grid: simple 2-column layout.
         */
        .option-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }
        /* 
         * Left-align text inside the clickable cards.
         */
        .option-grid .stButton > button {
            text-align: left !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>⚡ DataFlow Architect</h1>", unsafe_allow_html=True)

# ------------------ INTRODUCTION ------------------
st.markdown("""
<p style='text-align: center; font-size: 1.2rem;'>
Welcome to ⚡ DataFlow Architect! This application leverages AI to help you generate automated data science workflows.
You can either generate a comprehensive 📄 report of your dataset or receive 📜 step-by-step guidance for specific workflow sections.
Simply upload your dataset or document, choose your desired mode, and follow the prompts.
</p>
""", unsafe_allow_html=True)

# ------------------ SESSION STATE INITIALIZATION ------------------
if "selected_card" not in st.session_state:
    st.session_state["selected_card"] = None
if "dataset_uploaded" not in st.session_state:
    st.session_state["dataset_uploaded"] = False
if "df" not in st.session_state:
    st.session_state["df"] = None
if "step_report" not in st.session_state:
    st.session_state["step_report"] = ""
if "report_text" not in st.session_state:
    st.session_state["report_text"] = ""
if "report_displayed" not in st.session_state:
    st.session_state["report_displayed"] = False
if "step_report_displayed" not in st.session_state:
    st.session_state["step_report_displayed"] = False

# ------------------ CSV UPLOAD (REPORT MODE) ------------------
def handle_csv_upload():
    st.markdown("<h2 class='subheader'>📂 Upload Your Dataset</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv"], key="report_csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset loaded successfully!")
            st.dataframe(df)
            st.session_state["dataset_uploaded"] = True
            st.session_state["df"] = df
            return df
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
    return None

# ------------------ CLICKABLE CARD OPTIONS (STEP-BY-STEP GUIDANCE) ------------------
option_cards = [
    {
        "icon": "🧹",
        "title": "Data Cleaning & Preparation",
        "desc": "Handle missing values, outliers, and data transformations."
    },
    {
        "icon": "🔎",
        "title": "Exploratory Data Analysis (EDA)",
        "desc": "Visualize and summarize data to uncover patterns and trends."
    },
    {
        "icon": "🤖",
        "title": "Machine Learning Algorithm Selection",
        "desc": "Identify suitable ML models for your problem domain."
    },
    {
        "icon": "⚙️",
        "title": "Model Optimization & Feature Engineering",
        "desc": "Tune hyperparameters, engineer features, and improve performance."
    },
    {
        "icon": "🚀",
        "title": "Deployment & Real-World Considerations",
        "desc": "Implement model deployment strategies, scaling, and monitoring."
    },
]

def render_clickable_cards():
    st.markdown("<div class='option-grid'>", unsafe_allow_html=True)
    for card_info in option_cards:
        icon = card_info["icon"]
        title = card_info["title"]
        desc = card_info["desc"]
        label_str = f"{icon}  **{title}**\n{desc}"
        clicked = st.button(
            label=label_str,
            key=f"card_{title}",
            help=f"Click to select {title}",
            use_container_width=True
        )
        if clicked:
            st.session_state["selected_card"] = title
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ MAIN LAYOUT ------------------
st.markdown("<h2 class='subheader poll-container'>📊 How would you like to proceed?</h2>", unsafe_allow_html=True)
choice = st.radio("", ["📄 Report", "📜 Step-by-step Guidance"], horizontal=True)

if choice == "📄 Report":
    # REPORT MODE: Use two tabs: Input and Report.
    tab_input, tab_report = st.tabs(["Input", "Report"])
    with tab_input:
        df = handle_csv_upload()
        if df is not None:
            if st.button("Generate My Report", key="generate_report"):
                if st.session_state["df"] is None:
                    st.error("Please upload a dataset first.")
                else:
                    with st.spinner("Report is being generated, it might take a while, please wait!"):
                        report_path = rg.generate_output(st.session_state["df"], mode="report")
                        time.sleep(1)
                    try:
                        with open(report_path, "r", encoding="utf-8") as f:
                            report_text = f.read()
                        st.session_state["report_text"] = report_text
                        st.session_state["report_displayed"] = False  # Reset flag for fresh typing
                        st.info("Report generation complete. Please move to the Report tab to view the report.")
                    except Exception as e:
                        st.error(f"Error reading report file: {e}")
    with tab_report:
        st.markdown("<div id='report'></div>", unsafe_allow_html=True)
        if st.session_state["report_text"]:
            st.markdown("### Generated Report:")
            if not st.session_state["report_displayed"]:
                typewriter_display(st.session_state["report_text"])
                st.session_state["report_displayed"] = True
            else:
                st.markdown(st.session_state["report_text"])
            pdf_bytes = convert_md_to_pdf(st.session_state["report_text"])
            if pdf_bytes:
                st.download_button("Download Report as PDF", pdf_bytes, file_name="report.pdf", mime="application/pdf")
        else:
            st.markdown("### Report will appear here once generated.")
else:
    # STEP-BY-STEP GUIDANCE MODE: Use two tabs: Input and Workflow.
    tab_input, tab_workflow = st.tabs(["Input", "Workflow"])
    with tab_input:
        st.markdown("<h2 class='subheader'>📂 Upload a Document</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["txt", "pdf", "csv", "docx"], key="step_doc")
        if uploaded_file:
            if uploaded_file.type == "text/csv":
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success("CSV Dataset loaded successfully!")
                    st.dataframe(df)
                    st.session_state["dataset_uploaded"] = True
                    st.session_state["df"] = df
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
            else:
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(f"{FASTAPI_SERVER_URL}/upload/", files=files)
                if response.status_code == 200:
                    st.success("File uploaded successfully!")
                    st.session_state["dataset_uploaded"] = True
                else:
                    st.error("Error: File upload failed.")
    
        st.markdown("<h2 class='subheader'>🛠 Choose Your Section</h2>", unsafe_allow_html=True)
        render_clickable_cards()
        if st.session_state["selected_card"]:
            st.write(f"**Selected Section:** {st.session_state['selected_card']}")
        else:
            st.write("*No section selected yet.*")
    
        if st.button("Proceed with Selection", key="proceed"):
            if st.session_state["selected_card"] and st.session_state["dataset_uploaded"]:
                with st.spinner("Step is being generated, please wait..."):
                    step_output = rg.generate_output(st.session_state["df"], mode="step-by-step", selected_step=st.session_state["selected_card"])
                    time.sleep(1)
                st.session_state["step_report"] += "\n\n" + step_output
                st.session_state["selected_card"] = None
                st.session_state["step_report_displayed"] = False
                st.info("Step generation complete. Please move to the Workflow tab to view the update.")
            elif st.session_state["selected_card"] and not st.session_state["dataset_uploaded"]:
                st.warning("Please upload a dataset!")
            elif not st.session_state["selected_card"] and st.session_state["dataset_uploaded"]:
                st.warning("Please select a section!")
            else:
                st.warning("Please upload a dataset and select a section to start!")
    with tab_workflow:
        st.markdown("<div id='workflow'></div>", unsafe_allow_html=True)
        if st.session_state["step_report"]:
            st.markdown("### Combined Step-by-Step Report:")
            if not st.session_state["step_report_displayed"]:
                typewriter_display(st.session_state["step_report"])
                st.session_state["step_report_displayed"] = True
            else:
                st.markdown(st.session_state["step_report"])
            pdf_bytes = convert_md_to_pdf(st.session_state["step_report"])
            if pdf_bytes:
                st.download_button("Download Step-by-Step Report as PDF", pdf_bytes, file_name="step_report.pdf", mime="application/pdf")
        else:
            st.markdown("### No steps have been added yet.")
        st.markdown("#### New steps will be appended below. Scroll down to view the latest update.")
