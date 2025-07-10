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
            f"You are an expert sales engineer and AI solutions consultant at a B2B AI company. Your job is to help our sales team select the {top_k} most relevant past demos to show a new prospective client, based on their specific business problem or need.\n"
            "You must use deep semantic reasoning and business understanding, not just keyword matching. Focus most heavily on the 'Client Problem' field, but also consider 'Instalily AI Capabilities' and 'Benefit to Client' for additional context and business fit.\n"
            "Your goal is to find demos where the original client's problem is most similar to the new client's need, and where the AI solution and benefits would be persuasive to the new prospect.\n"
            "Only select demos that are truly relevant to the new client's need, even if the wording is different. Ignore demos that are only superficially related.\n"
            "For each selected demo, provide:\n"
            "  1) A similarity score (0.00-1.00) reflecting how well the demo matches the new client's need, based on your full business and technical understanding.\n"
            "  2) A concise, sales-ready explanation of why this demo is a strong match, referencing specific aspects of the new client's need and the demo fields. This explanation should help a salesperson justify showing this demo to the client.\n"
            "  3) The company name and video link, copied exactly as shown in the data.\n"
            "  4) Only use information present in the provided data. Do not invent or infer any details.\n"
            "\nAll output fields (company, video_link, etc.) must be word-for-word from the data.\n"
            "\nClient Need: " + customer_need + "\n\n"
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
