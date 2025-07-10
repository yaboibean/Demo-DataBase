import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class DemoMatcher:
    def __init__(self, spreadsheet_path: str, match_columns: List[str], embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the DemoMatcher.
        Args:
            spreadsheet_path: Path to the CSV file containing demo data.
            match_columns: List of columns to use for matching.
            embedding_model: SentenceTransformer model name.
        """
        self.spreadsheet_path = spreadsheet_path
        self.match_columns = match_columns
        self.embedding_model = embedding_model
        try:
            self.demos_df = pd.read_csv(spreadsheet_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read spreadsheet: {e}")
        if not all(col in self.demos_df.columns for col in match_columns):
            missing = [col for col in match_columns if col not in self.demos_df.columns]
            raise ValueError(f"Missing columns in CSV: {missing}")
        self.model = SentenceTransformer(self.embedding_model)
        self.demo_embeddings = self._create_embeddings()

    def _create_embeddings(self) -> np.ndarray:
        """
        Create embeddings for all demos in the dataframe.
        Returns:
            np.ndarray of demo embeddings.
        """
        # Only embed the 'Client Problem' column for matching
        demo_texts = []
        for _, row in self.demos_df.iterrows():
            text = str(row['Client Problem']) if 'Client Problem' in row and pd.notna(row['Client Problem']) else ''
            # Clean and normalize the text for better matching
            text = self._clean_text(text)
            demo_texts.append(text)
        if not demo_texts:
            raise ValueError("No demo texts found for embedding.")
        return self.model.encode(demo_texts, show_progress_bar=False)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for better matching.
        Args:
            text: Input text to clean
        Returns:
            Cleaned text
        """
        if not text or text == 'nan':
            return ''
        # Strip whitespace, normalize spaces, remove extra line breaks
        text = str(text).strip()
        text = ' '.join(text.split())  # Normalize whitespace
        return text

    def find_similar_demos(self, customer_need: str, top_k: int = 2, min_score: float = 0.1) -> List[Dict[str, Any]]:
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
        
        # Clean the customer need the same way we clean the demo texts
        cleaned_customer_need = self._clean_text(customer_need)
        
        # Only embed the user input for matching against 'Client Problem'
        need_embedding = self.model.encode([cleaned_customer_need])[0].reshape(1, -1)
        similarities = cosine_similarity(need_embedding, self.demo_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score < min_score:
                continue
            demo_info = self.demos_df.iloc[idx].to_dict()
            
            # Add debugging info
            original_problem = demo_info.get('Client Problem', '')
            cleaned_problem = self._clean_text(str(original_problem))
            
            results.append({
                'similarity_score': float(score),
                'demo_info': demo_info,
                'rank': len(results) + 1,
                'debug_info': {
                    'original_customer_need': customer_need,
                    'cleaned_customer_need': cleaned_customer_need,
                    'original_problem': original_problem,
                    'cleaned_problem': cleaned_problem,
                    'exact_text_match': cleaned_customer_need.lower() == cleaned_problem.lower()
                }
            })
            if len(results) >= top_k:
                break
        return results
    
    def get_detailed_analysis(self, customer_need: str, top_k: int = 2) -> str:
        """
        Get a detailed analysis of the most similar demos
        
        Args:
            customer_need: Description of current customer's problem/need
            top_k: Number of top matches to analyze
            
        Returns:
            Formatted string with detailed analysis
        """
        similar_demos = self.find_similar_demos(customer_need, top_k)
        
        if not similar_demos:
            return "No similar demos found."
        
        analysis = f"=== DEMO MATCHING ANALYSIS ===\n"
        analysis += f"Customer Need: {customer_need}\n"
        analysis += f"Found {len(similar_demos)} similar demos:\n\n"
        
        for demo in similar_demos:
            score = demo['similarity_score']
            info = demo['demo_info']
            rank = demo['rank']
            
            analysis += f"--- RANK {rank} (Similarity: {score:.3f}) ---\n"
            for col in self.match_columns:
                if col in info and pd.notna(info[col]):
                    analysis += f"{col}: {info[col]}\n"
            analysis += "\n"
        
        return analysis
    
    def debug_matching(self, customer_need: str) -> str:
        """
        Debug method to show what's happening in the matching process.
        Args:
            customer_need: The client's need/problem as a string.
        Returns:
            Debug information as a formatted string.
        """
        debug_info = []
        debug_info.append("=== DEBUG MATCHING PROCESS ===")
        debug_info.append(f"Original customer need: '{customer_need}'")
        
        cleaned_need = self._clean_text(customer_need)
        debug_info.append(f"Cleaned customer need: '{cleaned_need}'")
        
        debug_info.append("\n=== AVAILABLE CLIENT PROBLEMS ===")
        for idx, row in self.demos_df.iterrows():
            original_problem = row.get('Client Problem', '')
            cleaned_problem = self._clean_text(str(original_problem))
            company = row.get('Name/Client', 'Unknown')
            
            debug_info.append(f"[{idx}] {company}")
            debug_info.append(f"    Original: '{original_problem}'")
            debug_info.append(f"    Cleaned:  '{cleaned_problem}'")
            debug_info.append(f"    Empty: {cleaned_problem == ''}")
            debug_info.append(f"    Exact match: {cleaned_need.lower() == cleaned_problem.lower()}")
            debug_info.append("")
        
        # Try the actual matching
        try:
            results = self.find_similar_demos(customer_need, top_k=5, min_score=0.0)
            debug_info.append("=== MATCHING RESULTS ===")
            for result in results:
                debug_info.append(f"Rank {result['rank']}: Score {result['similarity_score']:.4f}")
                debug_info.append(f"  Company: {result['demo_info'].get('Name/Client', 'Unknown')}")
                if 'debug_info' in result:
                    debug_info.append(f"  Exact match: {result['debug_info']['exact_text_match']}")
                debug_info.append("")
        except Exception as e:
            debug_info.append(f"Error in matching: {e}")
        
        return "\n".join(debug_info)

    def create_sample_data(self, file_path: str = "sample_demo_database.csv"):
        """
        Create a sample spreadsheet with demo data for testing
        """
        sample_data = {
            'Company Name': [
                'TechCorp Inc',
                'FinanceFlow Ltd',
                'RetailMax',
                'HealthcarePlus',
                'ManufacturingPro',
                'EduTech Solutions',
                'LogisticsMaster',
                'RealEstate Pro',
                'FoodService Inc',
                'EnergyOptim'
            ],
            'Industry': [
                'Technology',
                'Finance',
                'Retail',
                'Healthcare',
                'Manufacturing',
                'Education',
                'Logistics',
                'Real Estate',
                'Food Service',
                'Energy'
            ],
            'Problem/Need': [
                'Automated code review and bug detection in software development',
                'Intelligent document processing for loan applications',
                'Customer service chatbot for e-commerce platform',
                'Medical record analysis and patient data extraction',
                'Predictive maintenance for manufacturing equipment',
                'Automated grading and student performance analysis',
                'Route optimization and delivery scheduling',
                'Property valuation and market analysis automation',
                'Inventory management and demand forecasting',
                'Energy consumption optimization and cost reduction'
            ],
            'Solution Provided': [
                'AI agent that reviews code, identifies bugs, and suggests improvements',
                'Document processing agent that extracts key information from loan docs',
                'Conversational AI that handles customer inquiries and support tickets',
                'Medical AI that processes patient records and generates insights',
                'Predictive analytics agent that monitors equipment health',
                'Educational AI that grades assignments and tracks student progress',
                'Logistics optimization agent that plans efficient delivery routes',
                'Real estate AI that analyzes property values and market trends',
                'Inventory management agent that predicts demand and manages stock',
                'Energy optimization agent that reduces consumption and costs'
            ],
            'Demo Type': [
                'Live Coding Demo',
                'Document Processing Demo',
                'Chatbot Interaction Demo',
                'Data Analysis Demo',
                'Predictive Analytics Demo',
                'Educational Platform Demo',
                'Route Planning Demo',
                'Market Analysis Demo',
                'Inventory Dashboard Demo',
                'Energy Monitoring Demo'
            ],
            'Demo Link/File': [
                'demos/techcorp_code_review.mp4',
                'demos/financeflow_doc_processing.mp4',
                'demos/retailmax_chatbot.mp4',
                'demos/healthcareplus_medical_ai.mp4',
                'demos/manufacturingpro_predictive.mp4',
                'demos/edutech_grading.mp4',
                'demos/logisticsmaster_routing.mp4',
                'demos/realestate_valuation.mp4',
                'demos/foodservice_inventory.mp4',
                'demos/energyoptim_monitoring.mp4'
            ],
            'Success Rate': [
                '85%',
                '92%',
                '78%',
                '88%',
                '91%',
                '83%',
                '89%',
                '86%',
                '90%',
                '87%'
            ],
            'Date': [
                '2024-01-15',
                '2024-02-03',
                '2024-02-20',
                '2024-03-10',
                '2024-03-25',
                '2024-04-12',
                '2024-04-28',
                '2024-05-15',
                '2024-05-30',
                '2024-06-10'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(file_path, index=False)
        print(f"Sample demo database created: {file_path}")
        return file_path

# Example usage
if __name__ == "__main__":
    # Use the real CSV file and only the 'Client Problem' column for matching
    csv_file = "Copy of Master File Demos Database - Demos Database.csv"
    matcher = DemoMatcher(csv_file, match_columns=["Client Problem"])
    # Example: Find similar demos for a new customer need
    customer_need = "We need an AI system to help with customer support and handle frequently asked questions"
    print("\n" + "="*50)
    print("DEMO MATCHING SYSTEM")
    print("="*50)
    analysis = matcher.get_detailed_analysis(customer_need, top_k=3)
    print(analysis)

