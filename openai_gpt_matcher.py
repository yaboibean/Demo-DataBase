import os
import pandas as pd
import openai
from dotenv import load_dotenv

class OpenAIGPTMatcher:
    def __init__(self, spreadsheet_path, match_columns, openai_api_key, model="gpt-4o"):
        self.spreadsheet_path = spreadsheet_path
        self.match_columns = match_columns
        self.model = model
        openai.api_key = openai_api_key
        self.demos_df = pd.read_csv(spreadsheet_path)

    def _row_to_text(self, row):
        text_parts = [f"{col}: {row[col]}" for col in self.match_columns if col in row and pd.notna(row[col])]
        return " | ".join(text_parts)

    def find_best_match(self, customer_need, top_k=3):
        # Prepare all demo texts
        demo_texts = [self._row_to_text(row) for _, row in self.demos_df.iterrows()]
        # Build prompt
        prompt = (
            f"A client has the following need: '{customer_need}'.\n"
            f"Here are {len(demo_texts)} past demos. For each, decide if it is a strong, possible, or weak match, and explain why.\n"
            f"Return the top {top_k} most relevant demos, with their company name, match strength, a similarity score from 0.00 (not similar) to 1.00 (very similar), and a short explanation.\n"
            f"For each match, include a 1-2 sentence reason why the demo is similar to the client need.\n"
            f"If there are no strong matches, include possible matches, but do not include demos that are completely unrelated.\n"
            f"Demos:\n"
        )
        for i, row in self.demos_df.iterrows():
            company = row.get('Name/Client', f'Demo {i+1}')
            demo_text = self._row_to_text(row)
            prompt += f"{i+1}. {company}: {demo_text}\n"
        prompt += ("\nFormat your answer as a numbered list: For each, give the company name, match strength (strong/possible/weak), a similarity score (0.00-1.00), and a 1-2 sentence reason why the demo is similar to the client need. If there are no strong or possible matches, say so clearly, but do not include demos that are completely unrelated."
                  )
        # Call OpenAI GPT model
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=900
        )
        return response.choices[0].message.content.strip()
