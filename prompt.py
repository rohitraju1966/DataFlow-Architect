import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from mistralai import Mistral
import time, inspect

# Load environment variables from the .env file
load_dotenv(dotenv_path=".env")  # Ensure .env is in the current working directory

def simple_timer(label: str = None):
    """Return current time so we can measure how long something took."""
    if label is None:
        label = inspect.stack()[1].function 
    print(f"[START] {label}")
    return label, time.perf_counter()

def tidy_headings(md: str) -> str:
    """
    1. Removes duplicate headings.
    2. Adds numbers to second‑level headings (## 1., ## 2., …).
    All in < 10 lines so it’s easy to read :-)
    """
    seen, result, n = set(), [], 0
    for line in md.splitlines():
        if line.startswith("## "): # we only care about '## '
            text = line[3:].strip()
            if text.lower() in seen: # skip duplicates
                continue
            seen.add(text.lower())
            n += 1
            line = f"## {n}. {text.lstrip('0123456789. ')}"
        result.append(line)
    return "\n".join(result)
    
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
                "4. Do not add extra lines before or after headings.\n"
                "5. Do NOT repeat a heading already used."
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
        label , t0 = simple_timer()
        user_prompt = self.generate_user_prompt_with_dataset(df, user_goal)
        response = self.client.chat.completions.create(
            messages=[self.SYSTEM_PROMPT, {"role": "user", "content": user_prompt}],
            model='gpt-4o-mini'
        )
        print(f"[END]  {label}: {time.perf_counter()-t0:.2f}s")
        return response.choices[0].message.content
    
    def eda(self, df: pd.DataFrame) -> str:
        """
        Generates the Exploratory Data Analysis section.
        """
        user_goal = "# **Exploratory Data Analysis**\nSuggest EDA techniques (visualizations, correlation checks, outlier detection)."
        label, t0 = simple_timer()
        user_prompt = self.generate_user_prompt_with_dataset(df, user_goal)
        response = self.client.chat.completions.create(
            messages=[self.SYSTEM_PROMPT, {"role": "user", "content": user_prompt}],
            model='gpt-4o-mini'
        )
        print(f"[END]  {label}: {time.perf_counter()-t0:.2f}s")
        return response.choices[0].message.content
    
    def ml_suggestions(self, df: pd.DataFrame) -> str:
        """
        Generates Machine Learning Algorithm Selection suggestions.
        """
        user_goal = "# **Machine Learning Suggestions**\nDiscuss supervised and unsupervised methods relevant to this dataset."
        label, t0 = simple_timer()
        user_prompt = self.generate_user_prompt_with_dataset(df, user_goal)
        response = self.client.chat.completions.create(
            messages=[self.SYSTEM_PROMPT, {"role": "user", "content": user_prompt}],
            model='gpt-4o-mini'
        )
        print(f"[END]  {label}: {time.perf_counter()-t0:.2f}s")
        return response.choices[0].message.content
    
    def feature_egr(self, df: pd.DataFrame) -> str:
        """
        Generates Feature Engineering suggestions.
        """
        user_goal = "# **Feature Engineering**\nDiscuss feature creation, transformation, and selection approaches."
        label, t0 = simple_timer()
        user_prompt = self.generate_user_prompt_with_dataset(df, user_goal)
        response = self.client.chat.completions.create(
            messages=[self.SYSTEM_PROMPT, {"role": "user", "content": user_prompt}],
            model='gpt-4o-mini'
        )
        print(f"[END]  {label}: {time.perf_counter()-t0:.2f}s")
        return response.choices[0].message.content
    
    def model_deployment(self, df: pd.DataFrame) -> str:
        """
        Generates Model Deployment & Data Drift suggestions.
        """
        user_goal = "# **Model Deployment & Data Drift**\nPropose strategies for deploying models and handling data drift."
        label, t0 = simple_timer()
        user_prompt = self.generate_user_prompt_with_dataset(df, user_goal)
        response = self.client.chat.completions.create(
            messages=[self.SYSTEM_PROMPT, {"role": "user", "content": user_prompt}],
            model='gpt-4o-mini'
        )
        print(f"[END]  {label}: {time.perf_counter()-t0:.2f}s")
        return response.choices[0].message.content
    
    def conclusion(self, df: pd.DataFrame, combined: str) -> str:
        """
        Generates the Conclusion section.
        """
        user_goal = "# **Conclusion**\nProvide a succinct concluding section summarizing the overall insights."
        label, t0 = simple_timer()
        user_prompt = self.generate_user_prompt_with_dataset(df, user_goal + f"\nSections combined:\n{combined}")
        response = self.client.chat.completions.create(
            messages=[self.SYSTEM_PROMPT, {"role": "user", "content": user_prompt}],
            model='gpt-4o-mini'
        )
        print(f"[END]  {label}: {time.perf_counter()-t0:.2f}s")
        return response.choices[0].message.content

    def generate_report_with_evaluation(self, df: pd.DataFrame, file_name: str = "automated_report.md", ratings: dict = None, automated_mode: bool = False) -> str:
        """
        Generates a full AI-written report that includes:
          - A static workflow overview (list of steps)
          - Detailed explanations for each step
          - Evaluations for each step
          - A concluding section
        In full report mode, if a ratings dictionary is provided, it is used for all steps.
        If automated_mode is True, it will automatically use all sections without prompting.
        Otherwise, the user is prompted for individual ratings.
        The final report is saved as a Markdown file.
        """
        if ratings is None:
            ratings_dict = {}
            print("\nPlease rate your knowledge for the following workflow steps (1 = rookie, 5 = expert):")
            for step in self.workflow_steps:
                while True:
                    try:
                        rating = int(input(f"Rating for '{step}': "))
                        if rating < 1 or rating > 5:
                            raise ValueError
                        ratings_dict[step] = rating
                        break
                    except ValueError:
                        print("Invalid input. Please enter an integer between 1 and 5.")
        else:
            ratings_dict = ratings
        
        # Generate a workflow overview.
        overview_prompt = (
            f"Given a dataset with columns: {df.columns.tolist()}, "
            "list the key steps involved in the data science workflow without explanations."
        )
        overview_response = self.client.chat.completions.create(
            messages=[self.SYSTEM_PROMPT, {"role": "user", "content": overview_prompt}],
            model='gpt-4o-mini'
        )
        overview = overview_response.choices[0].message.content
        print("Workflow Overview:")
        print(overview)
        
        # Determine which sections to generate
        if automated_mode:
            # For automated mode (typically used with Streamlit), use all sections
            selected_steps = self.workflow_steps
        else:
            # Ask the user which section to generate.
            print("\nChoose the section from which to generate the detailed report.")
            print("Type 'all' to include every section, or type the exact name of one of the workflow steps:")
            for step in self.workflow_steps:
                print(f"- {step}")
            
            while True:
                chosen_section = input("Enter your choice: ").strip()
                if chosen_section.lower() == "all":
                    selected_steps = self.workflow_steps
                    break
                elif chosen_section in self.workflow_steps:
                    selected_steps = [chosen_section]
                    break
                else:
                    print("Invalid selection. Please enter a valid section name (exactly as shown) or 'all'.")
        
        # Generate detailed outputs for the selected steps.
        detailed_outputs = {}
        completed_tasks = ""
        for step in selected_steps:
            # Determine detail level based on the provided rating.
            rating = ratings_dict[step]
            if rating <= 2:
                detail_level = "extremely detailed, comprehensive, beginner-friendly, including reasoning and methods"
            elif rating <= 4:
                detail_level = "moderately detailed"
            else:
                detail_level = "concise"
            
            step_prompt = (
                f"Dataset columns: {df.columns.tolist()}.\n"
                f"Tasks previously covered: {completed_tasks if completed_tasks else 'None'}.\n\n"
                f"Please provide an explanation with the following detail level: {detail_level} specifically for the '{step}' step, clearly detailing:\n"
                "- What needs to be done at this step (tailored to this dataset).\n"
                "- The reasoning behind each task.\n"
                "- Clear, explicit instructions on how to perform these tasks.\n"
                "Ensure the explanation is coherent and follows the workflow order."
            )
            
            step_response = self.client.chat.completions.create(
                messages=[self.SYSTEM_PROMPT, {"role": "user", "content": step_prompt}],
                model='gpt-4o-mini'
            )
            step_output = step_response.choices[0].message.content.strip()
            detailed_outputs[step] = step_output
            completed_tasks += f"{step}: {step_output}\n\n"
            print(f"Details for '{step}' generated.")
        
        # Build a list of functions to run only for the selected steps.
        selected_functions = []
        for step in selected_steps:
            selected_functions.extend(self.step_to_functions[step])
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda func: func(df), selected_functions))
        combined_report = tidy_headings("\n\n".join(results)) 
        
        evaluation_report = "\n\n".join(
            [f"**{step} Evaluation:**\n{detail}" for step, detail in detailed_outputs.items()]
        )
        
        # Summarize each section to reduce token count
        summary_for_conclusion = ""
        for step, detail in detailed_outputs.items():
            summary_lines = detail.strip().splitlines()
            short_summary = "\n".join(summary_lines[:3])  # First 3 lines as summary
            summary_for_conclusion += f"{step} Summary:\n{short_summary}\n\n"

        final_section = self.conclusion(df, summary_for_conclusion.strip())
        
        # Assemble the final Markdown report.
        if len(selected_steps) > 1:  # Full report or multiple sections
            overview_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(self.workflow_steps))
            full_report = (
                "## Data Science Workflow Steps\n\n" +
                f"{overview_text}\n\n" +
                "## Detailed Steps and Explanations\n\n" +
                f"{combined_report}\n\n" +
                "## Step-by-Step Evaluations\n\n" +
                f"{evaluation_report}\n\n" +
                "## Conclusion\n\n" +
                f"{final_section}"
            )
        else:  # Single section
            full_report = (
                "## Detailed Step Explanation\n\n" +
                f"{combined_report}\n\n" +
                "## Step Evaluation\n\n" +
                f"{evaluation_report}\n\n" +
                "## Conclusion\n\n" +
                f"{final_section}"
            )
        
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(full_report)
        
        print(f"\nMarkdown report saved as: {file_name}")
        return os.path.abspath(file_name)
    
    def generate_output(self, df: pd.DataFrame, mode: str, selected_step: str = None, output_file: str = None, rating: int = None, feedback: str = '') -> str:
        """
        Unified method to generate a report based on UI inputs.
        
        Parameters:
            df: The input DataFrame.
            mode: "report" for full report mode, or "step-by-step" for a single section.
            selected_step: For step-by-step mode, the exact workflow step to generate.
            output_file: Optional file name to save the report.
            rating: Optional expertise rating for step-by-step mode (1=rookie, 5=expert) or overall rating for report mode.
            feedback: Human feedback from previous steps to be incorporated into the prompt.
        
        Returns:
            A string: Either the path to the generated report file or the generated report text.
        """
        if mode == "report":
            if output_file is None:
                output_file = "automated_report.md"
            if rating is not None:
                ratings_dict = {step: rating for step in self.workflow_steps}
                # Set automated_mode=True to bypass the section selection prompt
                return self.generate_report_with_evaluation(df, output_file, ratings=ratings_dict, automated_mode=True)
            else:
                return self.generate_report_with_evaluation(df, output_file, automated_mode=True)
        elif mode == "step-by-step":
            if selected_step is None or selected_step not in self.workflow_steps:
                return "Invalid or missing selected step."
            if rating is None:
                while True:
                    try:
                        rating = int(input(f"Rating for '{selected_step}' (1=rookie, 5=expert): "))
                        if rating < 1 or rating > 5:
                            raise ValueError
                        break
                    except ValueError:
                        print("Invalid input. Please enter an integer between 1 and 5.")
            if rating <= 2:
                detail_level = "extremely detailed, comprehensive, beginner-friendly, including reasoning and methods"
            elif rating <= 4:
                detail_level = "moderately detailed"
            else:
                detail_level = "concise"
            
            # Specialized prompts for step-by-step mode.
            specialized_prompts = {
                "Data Cleaning & Preparation": "# **Data Preprocessing & Cleaning**\n1. Show the first five rows of the dataset in a table.\n2. Summarize missing values and duplicates, with specifics.\n3. Summarize unique values per column.\n4. Explain any cleaning steps that might be needed.",
                "Exploratory Data Analysis (EDA)": "# **Exploratory Data Analysis**\nSuggest appropriate EDA techniques (visualizations, correlation checks, outlier detection) tailored for this dataset.",
                "Machine Learning Algorithm Selection": "# **Machine Learning Suggestions**\nDiscuss supervised and unsupervised methods relevant to this dataset and explain their suitability.",
                "Model Optimization & Feature Engineering": "# **Feature Engineering**\nDiscuss feature creation, transformation, and selection approaches, including any relevant optimizations.",
                "Deployment & Real-World Considerations": "# **Model Deployment & Data Drift**\nPropose strategies for deploying models and handling data drift, addressing practical considerations."
            }
            
            base_goal = specialized_prompts[selected_step]
            if feedback:
                full_goal = (
                    f"Context: In the previous steps, if the  response was unsatisfactory, the user was asked to answer a couple of questions. "
                    f"Below is the complete feedback provided by the user for each step encountered by the user, where each question (Q) is followed by the corresponding answer (Answer):\n {feedback}.\n"
                    f"Based on this information and given that the report should be generated with the following detail level: {detail_level} "
                    f"(i.e., determining how detailed or concise the explanation should be), please provide an explanation for the '{selected_step}' step. "
                    #f"Use the following guidance as a reference, and feel free to add, modify, or omit details as necessary: {base_goal}"
                )
            else:
                full_goal = (
                    f"Based on the required detail level: {detail_level} "
                    f"(i.e., determining how detailed or concise the explanation should be), please provide an explanation for the '{selected_step}' step. "
                    #f"Use the following guidance as a reference, and feel free to add, modify, or omit details as necessary: {base_goal}"
                )
            user_prompt = self.generate_user_prompt_with_dataset(df, full_goal)
            response = self.client.chat.completions.create(
                messages=[self.SYSTEM_PROMPT, {"role": "user", "content": user_prompt}],
                model='gpt-4o-mini'
            )
            output = response.choices[0].message.content.strip()
            conclusion_section = self.conclusion(df, output)
            combined = tidy_headings(output + "\n\n" + conclusion_section)
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
