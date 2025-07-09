import os
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class OpenAIDemoMatcher:
    def __init__(self, spreadsheet_path: str, match_columns: List[str], openai_api_key: str, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the OpenAIDemoMatcher.
        Args:
            spreadsheet_path: Path to the CSV file containing demo data.
            match_columns: List of columns to use for matching.
            openai_api_key: OpenAI API key string.
            embedding_model: OpenAI embedding model name.
        """
        self.spreadsheet_path = spreadsheet_path
        self.match_columns = match_columns
        self.embedding_model = embedding_model
        openai.api_key = openai_api_key
        try:
            self.demos_df = pd.read_csv(spreadsheet_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read spreadsheet: {e}")
        if not all(col in self.demos_df.columns for col in match_columns):
            missing = [col for col in match_columns if col not in self.demos_df.columns]
            raise ValueError(f"Missing columns in CSV: {missing}")
        self.demo_embeddings = self._create_embeddings()

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts using the OpenAI API.
        Args:
            texts: List of strings to embed.
        Returns:
            np.ndarray of embeddings.
        """
        embeddings = []
        for text in texts:
            try:
                response = openai.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                raise RuntimeError(f"OpenAI embedding failed: {e}")
        return np.array(embeddings)

    def _create_embeddings(self) -> np.ndarray:
        """
        Create embeddings for all demos in the dataframe.
        Returns:
            np.ndarray of demo embeddings.
        """
        demo_texts = []
        for _, row in self.demos_df.iterrows():
            text_parts = [str(row[col]) for col in self.match_columns if col in row and pd.notna(row[col])]
            demo_texts.append(" | ".join(text_parts))
        if not demo_texts:
            raise ValueError("No demo texts found for embedding.")
        return self._embed_texts(demo_texts)

    def find_similar_demos(self, customer_need: str, top_k: int = 5, min_score: float = 0.3) -> List[Dict[str, Any]]:
        """
        Find the most similar demos to a customer need.
        Args:
            customer_need: The client's need/problem as a string.
            top_k: Number of top results to return.
            min_score: Minimum cosine similarity score to consider a match.
        Returns:
            List of dicts with similarity_score, demo_info, and rank.
        """
        if not customer_need or not isinstance(customer_need, str):
            raise ValueError("customer_need must be a non-empty string.")
        try:
            need_embedding = self._embed_texts([customer_need])[0].reshape(1, -1)
        except Exception as e:
            raise RuntimeError(f"Failed to embed customer need: {e}")
        similarities = cosine_similarity(need_embedding, self.demo_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1]
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score < min_score:
                continue
            demo_info = self.demos_df.iloc[idx].to_dict()
            # Build a rich summary for display
            summary = {
                'Client Problem': demo_info.get('Client Problem', ''),
                'Instalily AI Capabilities': demo_info.get('Instalily AI Capabilities', ''),
                'Benefit to Client': demo_info.get('Benefit to Client', ''),
                'Company Name': demo_info.get('Company Name', ''),
                'Industry': demo_info.get('Industry', ''),
                'Video Link': demo_info.get('Video Link', ''),
            }
            results.append({
                'similarity_score': float(score),
                'demo_info': summary,
                'rank': len(results) + 1
            })
            if len(results) >= top_k:
                break
        return results
