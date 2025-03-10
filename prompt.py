import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from mistralai import Mistral

# Load environment variables from the .env file
load_dotenv(dotenv_path=".env")  # Ensure .env is in the current working directory

class ReportGenerator:
    def __init__(self):
        # Retrieve API keys from environment variables
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables.")
        
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize the Mistral client (if needed)
        self.ms_client = Mistral(api_key=self.MISTRAL_API_KEY)
        
        # Define missing value indicators and a system prompt for completions.
        self.MISSING_VALUE_INDICATORS = ["?", "NA", "N/A", "NaN", "NULL", "", "Unknown", "unknown", "missing", "Missing"]
        self.SYSTEM_PROMPT = {
            "role": "system",
            "content": (
                "You are an AI assistant helping generate structured and concise data science sections. "
                "Adhere to these rules:\n"
                "1. Use headings: # for main sections and ## or ### for subsections.\n"
                "2. Keep each section concise and to the point.\n"
                "3. Avoid repeating dataset overviews or the word 'report' in each section.\n"
                "4. Do not add extra lines before or after headings."
            )
        }
        
        # Define the workflow steps.
        self.workflow_steps = [
            "Data Cleaning & Preparation",
            "Exploratory Data Analysis (EDA)",
            "Machine Learning Algorithm Selection",
            "Model Optimization & Feature Engineering",
            "Deployment & Real-World Considerations"
        ]
        
        # Map each workflow step to its corresponding helper functions.
        self.step_to_functions = {
            "Data Cleaning & Preparation": [self.preprocess_data],
            "Exploratory Data Analysis (EDA)": [self.eda],
            "Machine Learning Algorithm Selection": [self.ml_suggestions],
            "Model Optimization & Feature Engineering": [self.feature_egr],
            "Deployment & Real-World Considerations": [self.model_deployment]
        }
    
    def load_dataset_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a dataset from a file into a DataFrame. Supports CSV, Excel, or JSON.
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, na_values=self.MISSING_VALUE_INDICATORS)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, na_values=self.MISSING_VALUE_INDICATORS)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
                df.replace(self.MISSING_VALUE_INDICATORS, pd.NA, inplace=True)
            else:
                raise ValueError("Unsupported file format. Use CSV, Excel, or JSON.")
            
            if df.empty:
                raise ValueError("The dataset is empty.")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def generate_user_prompt_with_dataset(self, df: pd.DataFrame, goal: str) -> str:
        """
        Builds a user prompt that includes a brief dataset overview and the final goal.
        """
        dataset_head = df.head().to_markdown()
        dataset_describe = df.describe().to_markdown()
        missing_vals = df.isnull().sum()
        missing_values_summary = missing_vals[missing_vals != 0].to_string()
        column_types = df.dtypes.to_string()
        
        prompt = (
            f"Dataset Head (first 5 rows):\n{dataset_head}\n\n"
            f"Dataset Statistics (describe):\n{dataset_describe}\n\n"
            f"Missing Values:\n{missing_values_summary if missing_values_summary.strip() else 'No missing values'}\n\n"
            f"Column Data Types:\n{column_types}\n\n"
            f"Task:\n{goal}"
        )
        return prompt.strip()
    
    def preprocess_data(self, df: pd.DataFrame) -> str:
        """
        Generates the Data Preprocessing & Cleaning section.
        """
        missing_values = df.isnull().sum()
        missing_values_pct = (missing_values / df.shape[0] * 100).round(2).astype(str) + "%"
        duplicate_count = df.duplicated().sum()
        unique_values = df.nunique()
        
        user_goal = (
            "# **Data Preprocessing & Cleaning**\n"
            "1. Show the first five rows of the dataset in a table.\n"
            f"2. Summarize missing values (Missing: {missing_values_pct.to_string()}) and duplicates (Duplicates: {duplicate_count}).\n"
            f"3. Summarize unique values per column: {unique_values.to_string()}.\n"
            "4. Explain any cleaning steps that might be needed."
        )
        user_prompt = self.generate_user_prompt_with_dataset(df, user_goal)
        response = self.client.chat.completions.create(
            messages=[self.SYSTEM_PROMPT, {"role": "user", "content": user_prompt}],
            model='gpt-4o-mini'
        )
        return response.choices[0].message.content
    
    def eda(self, df: pd.DataFrame) -> str:
        """
        Generates the Exploratory Data Analysis section.
        """
        user_goal = "# **Exploratory Data Analysis**\nSuggest EDA techniques (visualizations, correlation checks, outlier detection)."
        user_prompt = self.generate_user_prompt_with_dataset(df, user_goal)
        response = self.client.chat.completions.create(
            messages=[self.SYSTEM_PROMPT, {"role": "user", "content": user_prompt}],
            model='gpt-4o-mini'
        )
        return response.choices[0].message.content
    
    def ml_suggestions(self, df: pd.DataFrame) -> str:
        """
        Generates Machine Learning Algorithm Selection suggestions.
        """
        user_goal = "# **Machine Learning Suggestions**\nDiscuss supervised and unsupervised methods relevant to this dataset."
        user_prompt = self.generate_user_prompt_with_dataset(df, user_goal)
        response = self.client.chat.completions.create(
            messages=[self.SYSTEM_PROMPT, {"role": "user", "content": user_prompt}],
            model='gpt-4o-mini'
        )
        return response.choices[0].message.content
    
    def feature_egr(self, df: pd.DataFrame) -> str:
        """
        Generates Feature Engineering suggestions.
        """
        user_goal = "# **Feature Engineering**\nDiscuss feature creation, transformation, and selection approaches."
        user_prompt = self.generate_user_prompt_with_dataset(df, user_goal)
        response = self.client.chat.completions.create(
            messages=[self.SYSTEM_PROMPT, {"role": "user", "content": user_prompt}],
            model='gpt-4o-mini'
        )
        return response.choices[0].message.content
    
    def model_deployment(self, df: pd.DataFrame) -> str:
        """
        Generates Model Deployment & Data Drift suggestions.
        """
        user_goal = "# **Model Deployment & Data Drift**\nPropose strategies for deploying models and handling data drift."
        user_prompt = self.generate_user_prompt_with_dataset(df, user_goal)
        response = self.client.chat.completions.create(
            messages=[self.SYSTEM_PROMPT, {"role": "user", "content": user_prompt}],
            model='gpt-4o-mini'
        )
        return response.choices[0].message.content
    
    def conclusion(self, df: pd.DataFrame, combined: str) -> str:
        """
        Generates the Conclusion section.
        """
        user_goal = "# **Conclusion**\nProvide a succinct concluding section summarizing the overall insights."
        user_prompt = self.generate_user_prompt_with_dataset(df, user_goal + f"\nSections combined:\n{combined}")
        response = self.client.chat.completions.create(
            messages=[self.SYSTEM_PROMPT, {"role": "user", "content": user_prompt}],
            model='gpt-4o-mini'
        )
        return response.choices[0].message.content

    def generate_report_with_evaluation(self, df: pd.DataFrame, file_name: str = "automated_report.md") -> str:
        """
        Generates an AI-written full report (for Report mode) that includes:
        1. A static overview listing the workflow steps in numbered order.
        2. Detailed explanations for each step (generated in order).
        3. Evaluations for each step.
        4. A concluding section.
        The final report is assembled in proper Markdown format and saved to a file.
        """
        # Use all steps in full report mode.
        selected_steps = self.workflow_steps
        
        # 1) Generate a static workflow overview in order.
        overview = "\n".join(f"{i+1}. {step}" for i, step in enumerate(selected_steps))
        print("Workflow Overview:")
        print(overview)
        
        # 2) Generate detailed outputs for each step in order.
        detailed_outputs = {}
        completed_tasks = ""
        for step in selected_steps:
            detail_level = "moderately detailed"  # default detail level
            step_prompt = (
                f"Dataset columns: {df.columns.tolist()}.\n"
                f"Tasks previously covered: {completed_tasks if completed_tasks else 'None'}.\n\n"
                f"Provide a {detail_level} explanation for the '{step}' step. "
                "Include the following:\n"
                "- Specific tasks to be performed at this step (tailored to this dataset).\n"
                "- The reasoning behind each task.\n"
                "- Clear, explicit instructions on how to perform these tasks.\n"
                "Ensure the explanation is coherent and follows the order of the workflow."
            )
            
            step_response = self.client.chat.completions.create(
                messages=[self.SYSTEM_PROMPT, {"role": "user", "content": step_prompt}],
                model='gpt-4o-mini'
            )
            step_output = step_response.choices[0].message.content.strip()
            detailed_outputs[step] = step_output
            completed_tasks += f"{step}: {step_output}\n\n"
            print(f"Details for '{step}' generated.")
        
        # 3) Execute all helper functions for each step concurrently (if needed).
        selected_functions = []
        for step in selected_steps:
            selected_functions.extend(self.step_to_functions[step])
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda func: func(df), selected_functions))
        combined_report = "\n\n".join(results)
        
        # 4) Build an evaluation report that shows each step's evaluation.
        evaluation_report = "\n\n".join(
            [f"**{step} Evaluation:**\n{detail}" for step, detail in detailed_outputs.items()]
        )
        
        # 5) Generate a concluding section.
        final_section = self.conclusion(df, combined_report + "\n\n" + evaluation_report)
        
        # 6) Assemble the final Markdown report in the desired order.
        full_report = (
            "## Data Science Workflow Steps\n\n"
            f"{overview}\n\n"
            "## Detailed Steps and Explanations\n\n"
            f"{combined_report}\n\n"
            "## Step-by-Step Evaluations\n\n"
            f"{evaluation_report}\n\n"
            "## Conclusion\n\n"
            f"{final_section}"
        )
        
        # 7) Write the final report to a Markdown file.
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(full_report)
        
        print(f"\nMarkdown report saved as: {file_name}")
        return os.path.abspath(file_name)


    
    def generate_output(self, df: pd.DataFrame, mode: str, selected_step: str = None, output_file: str = None) -> str:
        """
        Unified method to generate a report based on UI inputs.
        
        Parameters:
            df: The input DataFrame.
            mode: "report" for full report mode, or "step-by-step" for a single section.
            selected_step: For step-by-step mode, the exact workflow step to generate.
            output_file: Optional file name to save the report.
        
        Returns:
            A string: Either the path to the generated report file or the generated report text.
        """
        if mode == "report":
            if output_file is None:
                output_file = "automated_report.md"
            return self.generate_report_with_evaluation(df, output_file)
        elif mode == "step-by-step":
            if selected_step is None or selected_step not in self.workflow_steps:
                return "Invalid or missing selected step."
            # Call the helper function for the selected step.
            output = self.step_to_functions[selected_step][0](df)
            # Optionally generate a conclusion for the step.
            conclusion_section = self.conclusion(df, output)
            combined = output + "\n\n" + conclusion_section
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(combined)
                return os.path.abspath(output_file)
            return combined
        else:
            return "Invalid mode selected. Choose either 'report' or 'step-by-step'."
    
    def run(self):
        """
        (Deprecated interactive method)
        For UI usage, call generate_output() with the appropriate parameters.
        """
        file_path = input("Please enter the dataset file path (CSV, Excel, or JSON): ").strip()
        df = self.load_dataset_from_file(file_path)
        if df is None:
            print("Dataset could not be loaded. Exiting.")
            return
        
        # For demonstration, generate a full report.
        report_path = self.generate_output(df, mode="report")
        print(f"Report generated at: {report_path}")

def main():
    rg = ReportGenerator()
    rg.run()

if __name__ == "__main__":
    main()
