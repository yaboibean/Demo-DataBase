import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class DemoMatcher:
    def __init__(self, spreadsheet_path: str = None, match_columns=None):
        """
        Initialize the Demo Matcher system
        
        Args:
            spreadsheet_path: Path to the spreadsheet containing past demos
            match_columns: List of columns to use for semantic matching
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.demos_df = None
        self.demo_embeddings = None
        # Use user-specified columns or default to the three most relevant
        if match_columns is None:
            self.match_columns = ["Client Problem", "Instalily AI Capabilities", "Benefit to Client"]
        else:
            self.match_columns = match_columns
        if spreadsheet_path:
            self.load_spreadsheet(spreadsheet_path)
    
    def load_spreadsheet(self, file_path: str):
        """
        Load the spreadsheet with past demos
        
        Expected columns:
        - Company Name
        - Industry
        - Problem/Need
        - Solution Provided
        - Demo Type
        - Demo Link/File
        - Success Rate
        - Date
        """
        try:
            if file_path.endswith('.csv'):
                self.demos_df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.demos_df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
            print(f"Loaded {len(self.demos_df)} demos from {file_path}")
            print(f"Columns: {list(self.demos_df.columns)}")
            
            # Create embeddings for all past demos
            self._create_embeddings()
            
        except Exception as e:
            print(f"Error loading spreadsheet: {e}")
            return None
    
    def _create_embeddings(self):
        """Create embeddings for all past demos using only the specified columns"""
        if self.demos_df is None:
            print("No demo data loaded!")
            return
        
        # Combine relevant text fields for embedding
        demo_texts = []
        for _, row in self.demos_df.iterrows():
            text_parts = []
            for col in self.match_columns:
                if col in row and pd.notna(row[col]):
                    text_parts.append(str(row[col]))
            demo_texts.append(" | ".join(text_parts))
        
        print(f"Creating embeddings for {len(demo_texts)} demos using columns: {self.match_columns}")
        self.demo_embeddings = self.model.encode(demo_texts)
        print(f"Created embeddings for {len(demo_texts)} demos")
    
    def find_similar_demos(self, customer_need: str, top_k: int = 5) -> List[Dict]:
        """
        Find the most similar past demos for a given customer need
        
        Args:
            customer_need: Description of current customer's problem/need
            top_k: Number of top matches to return
            
        Returns:
            List of dictionaries containing demo information and similarity scores
        """
        if self.demos_df is None or self.demo_embeddings is None:
            print("No demo data or embeddings available!")
            return []
        
        # Create embedding for the customer need
        need_embedding = self.model.encode([customer_need])
        
        # Calculate similarities
        similarities = cosine_similarity(need_embedding, self.demo_embeddings)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            demo_info = self.demos_df.iloc[idx].to_dict()
            
            results.append({
                'similarity_score': float(similarity_score),
                'demo_info': demo_info,
                'rank': len(results) + 1
            })
        
        return results
    
    def get_detailed_analysis(self, customer_need: str, top_k: int = 5) -> str:
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
    # Use the real CSV file and relevant columns
    csv_file = "Copy of Master File Demos Database - Demos Database.csv"
    matcher = DemoMatcher(csv_file, match_columns=["Client Problem", "Instalily AI Capabilities", "Benefit to Client"])
    
    # Example: Find similar demos for a new customer need
    customer_need = "We need an AI system to help with customer support and handle frequently asked questions"
    
    print("\n" + "="*50)
    print("DEMO MATCHING SYSTEM")
    print("="*50)
    
    analysis = matcher.get_detailed_analysis(customer_need, top_k=3)
    print(analysis)

