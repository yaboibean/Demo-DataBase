import os
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity

class OpenAIDemoMatcher:
    def __init__(self, spreadsheet_path, match_columns, openai_api_key, embedding_model="text-embedding-3-small"):
        self.spreadsheet_path = spreadsheet_path
        self.match_columns = match_columns
        self.embedding_model = embedding_model
        openai.api_key = openai_api_key
        self.demos_df = pd.read_csv(spreadsheet_path)
        self.demo_embeddings = self._create_embeddings()

    def _embed_texts(self, texts):
        # OpenAI API only allows up to 2048 tokens per request, so batch if needed
        embeddings = []
        for text in texts:
            response = openai.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embeddings.append(response.data[0].embedding)
        return np.array(embeddings)

    def _create_embeddings(self):
        demo_texts = []
        for _, row in self.demos_df.iterrows():
            text_parts = [str(row[col]) for col in self.match_columns if col in row and pd.notna(row[col])]
            demo_texts.append(" | ".join(text_parts))
        return self._embed_texts(demo_texts)

    def find_similar_demos(self, customer_need, top_k=5, min_score=0.3):
        need_embedding = self._embed_texts([customer_need])[0].reshape(1, -1)
        similarities = cosine_similarity(need_embedding, self.demo_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1]
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score < min_score:
                continue
            demo_info = self.demos_df.iloc[idx].to_dict()
            results.append({
                'similarity_score': float(score),
                'demo_info': demo_info,
                'rank': len(results) + 1
            })
            if len(results) >= top_k:
                break
        return results
