import os
import openai
import pandas as pd
from typing import List, Dict, Any

try:
    import gspread
    from gspread_dataframe import get_as_dataframe
    from google.oauth2.service_account import Credentials
except ImportError:
    gspread = None
    get_as_dataframe = None
    Credentials = None

class OpenAIGPTMatcher:
    def __init__(self, spreadsheet_path: str, match_columns: List[str], openai_api_key: str, gpt_model: str = "gpt-4o", sheet_mode: bool = False, sheet_name: str = None, creds_json: str = None):
        """
        Initialize the OpenAIGPTMatcher.
        Args:
            spreadsheet_path: Path to the CSV file or Google Sheet ID.
            match_columns: List of columns to use for matching.
            openai_api_key: OpenAI API key string.
            gpt_model: OpenAI GPT model name.
            sheet_mode: If True, load from Google Sheets instead of CSV.
            sheet_name: Name of the worksheet in Google Sheets.
            creds_json: Path to Google service account JSON.
        """
        self.spreadsheet_path = spreadsheet_path # Path to the CSV file
        self.match_columns = match_columns
        self.gpt_model = gpt_model or "gpt-4o"
        openai.api_key = openai_api_key
        if sheet_mode:
            if not (gspread and get_as_dataframe and Credentials):
                raise ImportError("gspread, gspread_dataframe, and google-auth must be installed for Google Sheets support.")
            if not (sheet_name and creds_json):
                raise ValueError("sheet_name and creds_json are required for Google Sheets mode.")
            scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
            creds = Credentials.from_service_account_file(creds_json, scopes=scopes)
            gc = gspread.authorize(creds)
            ws = gc.open_by_key(spreadsheet_path).worksheet(sheet_name)
            self.demos_df = get_as_dataframe(ws, evaluate_formulas=True, header=0)
            self.demos_df = self.demos_df.dropna(how='all')
        else:
            try:
                self.demos_df = pd.read_csv(spreadsheet_path)
            except Exception as e:
                raise RuntimeError(f"Failed to read spreadsheet: {e}")
        if not all(col in self.demos_df.columns for col in match_columns):
            missing = [col for col in match_columns if col not in self.demos_df.columns]
            raise ValueError(f"Missing columns in data: {missing}")

    def _format_demo(self, row: pd.Series) -> str:
        """
        Format a demo row for GPT prompt.
        Args:
            row: Pandas Series representing a demo.
        Returns:
            Formatted string for GPT prompt.
        """
        return " | ".join([f"{col}: {row[col]}" for col in self.match_columns if col in row and pd.notna(row[col])])

    def find_best_demos(self, customer_need: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Use GPT-4o (ChatGPT o3) to reason about the best demo matches for a customer need.
        Args:
            customer_need: The client's need/problem as a string.
            top_k: Number of top results to return.
        Returns:
            List of dicts with explanation, demo_info, and rank.
        """
        if not customer_need or not isinstance(customer_need, str):
            raise ValueError("customer_need must be a non-empty string.")
        
        # Clean and normalize text for better matching
        def clean_text(text):
            if not text or str(text).lower() == 'nan':
                return ''
            return str(text).strip()
        
        cleaned_customer_need = clean_text(customer_need)
        
        # Only use the 'Client Problem' column for matching - clean the data
        client_problems = []
        for i, problem in enumerate(self.demos_df['Client Problem']):
            cleaned_problem = clean_text(problem)
            if cleaned_problem:  # Only include non-empty problems
                client_problems.append((i, cleaned_problem))
        
        # Check for exact matches first
        exact_matches = []
        for idx, problem in client_problems:
            if cleaned_customer_need.lower() == problem.lower():
                demo_info = self.demos_df.iloc[idx].to_dict()
                # Only use the 'Demo link ' column for the link
                demo_link = demo_info.get('Demo link ', '')
                exact_match = {
                    'rank': len(exact_matches) + 1,
                    'similarity_score': 1.0,
                    'explanation': 'EXACT MATCH: This demo addresses the identical problem statement.',
                    'demo_link': demo_link,
                    'demo_index': idx,
                    'demo_info': demo_info,
                    'client_name': demo_info.get('Name/Client', 'Unknown'),
                    'industry': demo_info.get('Industry', 'Unknown'),
                    'client_problem': demo_info.get('Client Problem', ''),
                    'solution': demo_info.get('Instalily AI Capabilities', ''),
                    'benefits': demo_info.get('Benefit to Client', '')
                }
                exact_matches.append(exact_match)
                if len(exact_matches) >= top_k:
                    return exact_matches
        # If we have exact matches, return them
        if exact_matches:
            return exact_matches
            
        prompt = (
            f"You are an expert AI demo matcher for a B2B AI company. Your job is to select the {top_k} most relevant past demos from the list below, given a specific client problem or need.\n"
            "You must use deep semantic reasoning, not just keyword matching. ONLY use the 'Client Problem' field for your analysis.\n"
            "Your goal is to find demos where the original client's problem is most similar in meaning and topic to the new client's need.\n"
            "IGNORE any demos that do not clearly relate to the main topic or keywords in the new client's need. If a demo is about a different business area (e.g., supply chain vs. customer engagement), do NOT select it.\n"
            "Only select demos that are truly relevant to the new client's need, even if the wording is different.\n"
            "For each selected demo, provide:\n"
            "  1) A similarity score (0.00-1.00) reflecting how well the demo matches the new client's need, based on your full understanding.\n"
            "  2) A concise explanation of why this demo is a strong match, referencing specific aspects of the new client's need and the demo problem. Start the reasoning with 'this demo is a strong match because...', but replace the word strong accordingly, (Strong, great, good, solid), but don't be afraid to use the word strong.\n"
            "  3) The demo_index number exactly as shown in the list.\n"
            "  4) Only use information present in the provided data. Do not invent or infer any details.\n"
            "\nClient Need: " + cleaned_customer_need + "\n\n"
            "Client Problems:\n" + "\n".join([f"[{idx}] {problem}" for idx, problem in client_problems]) +
            "\n\nRespond in JSON as a list of objects with keys: 'rank', 'similarity_score', 'explanation', and 'demo_index'."
        )
        try:
            response = openai.chat.completions.create(
                model=self.gpt_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Lower temperature for more consistent results
                max_tokens=1200
            )
            import json
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("No content returned from GPT-4o.")
            # Find the first and last square brackets to extract JSON
            start = content.find('[')
            end = content.rfind(']') + 1
            if start == -1 or end == -1:
                raise RuntimeError(f"Could not find JSON array in GPT-4o response: {content}")
            json_str = content[start:end]
            matches = json.loads(json_str)
        except Exception as e:
            raise RuntimeError(f"OpenAI GPT-4o matching failed: {e}")
        # Attach full demo info for each match
        results = []
        for match in matches:
            idx = match.get('demo_index', None)
            if idx is not None and 0 <= idx < len(self.demos_df):
                demo_info = self.demos_df.iloc[idx].to_dict()
                demo_link = demo_info.get('Demo link ', '')
            else:
                demo_info = {}
                demo_link = ''
            
            # Ensure all critical fields are present at the top level
            result = {
                'rank': match.get('rank', 0),
                'similarity_score': match.get('similarity_score', 0.0),
                'explanation': match.get('explanation', 'No explanation provided'),
                'demo_link': demo_link,
                'demo_index': idx,
                'demo_info': demo_info,
                'client_name': demo_info.get('Name/Client', 'Unknown'),
                'industry': demo_info.get('Industry', 'Unknown'),
                'client_problem': demo_info.get('Client Problem', ''),
                'solution': demo_info.get('Instalily AI Capabilities', ''),
                'benefits': demo_info.get('Benefit to Client', '')
            }
            results.append(result)
        return results

    def debug_matching(self, customer_need: str) -> str:
        """
        Debug method to show what's happening in the matching process.
        Args:
            customer_need: The client's need/problem as a string.
        Returns:
            Debug information as a formatted string.
        """
        def clean_text(text):
            if not text or str(text).lower() == 'nan':
                return ''
            return str(text).strip()
        
        debug_info = []
        debug_info.append("=== DEBUG OPENAI MATCHING PROCESS ===")
        debug_info.append(f"Original customer need: '{customer_need}'")
        
        cleaned_need = clean_text(customer_need)
        debug_info.append(f"Cleaned customer need: '{cleaned_need}'")
        
        debug_info.append("\n=== AVAILABLE CLIENT PROBLEMS ===")
        for idx, row in self.demos_df.iterrows():
            original_problem = row.get('Client Problem', '')
            cleaned_problem = clean_text(original_problem)
            company = row.get('Name/Client', 'Unknown')
            
            debug_info.append(f"[{idx}] {company}")
            debug_info.append(f"    Original: '{original_problem}'")
            debug_info.append(f"    Cleaned:  '{cleaned_problem}'")
            debug_info.append(f"    Empty: {cleaned_problem == ''}")
            debug_info.append(f"    Exact match: {cleaned_need.lower() == cleaned_problem.lower()}")
            debug_info.append("")
        
        return "\n".join(debug_info)

    def suggest_search_terms(self, user_query: str) -> str:
        """
        Given a user query (e.g., 'Suggest search terms for insurance demos'), use GPT to suggest relevant search terms or clarify/refine the query.
        Args:
            user_query: The user's question or request for help.
        Returns:
            A string with suggested search terms or advice.
        """
        prompt = (
            "You are an expert AI demo search assistant. Given the following user request, suggest 3-5 highly relevant search terms or phrases that would help them find the best demos in a B2B AI demo database. "
            "If the user is asking for clarification or advice, provide a concise, helpful response.\n"
            f"User request: {user_query}\n"
            "Respond with a short list of search terms or a brief answer."
        )
        try:
            response = openai.chat.completions.create(
                model=self.gpt_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200
            )
            content = response.choices[0].message.content
            return content.strip() if content else "(No response from AI)"
        except Exception as e:
            return f"Error: {e}"
