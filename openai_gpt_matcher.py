import os
import openai
import pandas as pd
from typing import List, Dict, Any

class OpenAIGPTMatcher:
    def __init__(self, spreadsheet_path: str, match_columns: List[str], openai_api_key: str, gpt_model: str = "gpt-4o"):
        """
        Initialize the OpenAIGPTMatcher.
        Args:
            spreadsheet_path: Path to the CSV file containing demo data.
            match_columns: List of columns to use for matching.
            openai_api_key: OpenAI API key string.
            gpt_model: OpenAI GPT model name.
        """
        self.spreadsheet_path = spreadsheet_path
        self.match_columns = match_columns
        self.gpt_model = gpt_model
        openai.api_key = openai_api_key
        try:
            self.demos_df = pd.read_csv(spreadsheet_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read spreadsheet: {e}")
        if not all(col in self.demos_df.columns for col in match_columns):
            missing = [col for col in match_columns if col not in self.demos_df.columns]
            raise ValueError(f"Missing columns in CSV: {missing}")

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
        Use GPT-4o to reason about the best demo matches for a customer need.
        Args:
            customer_need: The client's need/problem as a string.
            top_k: Number of top results to return.
        Returns:
            List of dicts with explanation, demo_info, and rank.
        """
        if not customer_need or not isinstance(customer_need, str):
            raise ValueError("customer_need must be a non-empty string.")
        demo_texts = [self._format_demo(row) for _, row in self.demos_df.iterrows()]
        prompt = (
            f"You are an expert AI demo matcher. Given a client need, select the {top_k} most relevant demos from the list below. "
            "For each, provide: 1) a similarity score (0-1), 2) a short explanation of why it matches, and 3) the company name and video link if available. "
            "Only select demos that are truly relevant.\n\n"
            f"Client Need: {customer_need}\n\n"
            "Demos:\n" + "\n".join([f"[{i+1}] {demo}" for i, demo in enumerate(demo_texts)]) +
            "\n\nRespond in JSON as a list of objects with keys: 'rank', 'similarity_score', 'explanation', 'company', 'video_link', and 'demo_index'."
        )
        try:
            response = openai.chat.completions.create(
                model=self.gpt_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
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
            else:
                demo_info = {}
            match['demo_info'] = demo_info
            results.append(match)
        return results
