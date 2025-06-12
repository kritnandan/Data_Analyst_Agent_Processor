import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pytesseract
import PyPDF2
from docx import Document
import requests
import json
from io import BytesIO
from typing import Optional, Dict, List, Tuple, Union
import streamlit as st

# Constants
SUPPORTED_FILE_TYPES = {
    'tabular': ['csv', 'xlsx'],
    'text': ['txt', 'docx', 'pdf'],
    'image': ['png', 'jpg', 'jpeg']
}

class DataAnalystAgent:
    """A data analysis agent that processes various file types and provides insights."""
    
    def __init__(self):
        """Initialize the agent with empty data structures."""
        self._reset_state()
        
    def _reset_state(self) -> None:
        """Reset all data attributes to their initial state."""
        self.data = None
        self.df = None  # For tabular data
        self.text_content = ""  # For text-based documents
        self.current_file_type = None
        self.visualizations = []
    
    def process_file(self, file_path: str) -> str:
        """
        Process an uploaded file based on its type.
        
        Args:
            file_path: Path to the file to be processed
            
        Returns:
            str: Success message with file type
            
        Raises:
            ValueError: If file type is unsupported
            IOError: If file cannot be read
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_ext = os.path.splitext(file_path)[1][1:].lower()
        self._reset_state()
        
        if file_ext in SUPPORTED_FILE_TYPES['tabular']:
            self._process_tabular_data(file_path)
        elif file_ext in SUPPORTED_FILE_TYPES['text']:
            self._process_text_data(file_path)
        elif file_ext in SUPPORTED_FILE_TYPES['image']:
            self._process_image_data(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported types: {SUPPORTED_FILE_TYPES}")
            
        self.current_file_type = file_ext
        return f"Successfully processed {os.path.basename(file_path)} as {file_ext} file"
    
    def _process_tabular_data(self, file_path: str) -> None:
        """Process CSV or Excel files into a pandas DataFrame."""
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.df = pd.read_excel(file_path)
            self.data = self.df.to_dict()
        except Exception as e:
            raise IOError(f"Error reading tabular file: {str(e)}")
    
    def _process_text_data(self, file_path: str) -> None:
        """Extract text content from text-based documents."""
        try:
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.text_content = f.read()
                    
            elif file_path.endswith('.docx'):
                doc = Document(file_path)
                self.text_content = '\n'.join(para.text for para in doc.paragraphs)
                
            elif file_path.endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    self.text_content = '\n'.join(page.extract_text() for page in reader.pages)
                    
            self.data = {"text_content": self.text_content}
        except Exception as e:
            raise IOError(f"Error reading text file: {str(e)}")
    
    def _process_image_data(self, file_path: str) -> None:
        """Process image files using OCR to extract text."""
        try:
            img = Image.open(file_path)
            self.text_content = pytesseract.image_to_string(img)
            self.data = {"text_content": self.text_content, "image": img}
        except Exception as e:
            raise IOError(f"Error processing image file: {str(e)}")
    
    def query_llama(self, prompt: str, api_key: str, model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8") -> str:
        """
        Send query to Together.ai's Llama model.
        
        Args:
            prompt: User's question or prompt
            api_key: Together.ai API key
            model: Model identifier
            
        Returns:
            str: Model's response
            
        Raises:
            Exception: If API request fails
        """
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
        # Prepare context based on processed data
        context = ""
        if self.df is not None:
            context += f"Data summary:\n{self.df.describe().to_string()}\n"
        if self.text_content:
            context += f"Extracted text (truncated):\n{self.text_content[:2000]}...\n"
    
        full_prompt = f"""You are a data analyst assistant. The user has provided some data and has the following question:
        
        {prompt}
        
        {context}
        
        Please provide a detailed, helpful response. If the question requires data analysis, 
        perform the analysis and explain your reasoning. If visualization would help, 
        suggest what kind of visualization to create.
        """
        
        payload = {
            "model": model,
            "prompt": full_prompt,
            "max_tokens": 1500,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(
                "https://api.together.xyz/inference",
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # Handle different response structures
            if "output" in result and "choices" in result["output"]:
                return result["output"]["choices"][0]["text"]
            elif "choices" in result:
                return result["choices"][0]["text"]
            else:
                raise ValueError("Unexpected API response format")
                
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")

    def create_visualization(self, viz_type: str = "auto") -> str:
        """
        Create visualizations based on data.
        
        Args:
            viz_type: Type of visualization ('auto', 'histogram', 'bar', etc.)
            
        Returns:
            str: Status message
        """
        if self.df is None:
            return "No tabular data available for visualization"
            
        self.visualizations = []
        
        if viz_type == "auto":
            self._create_auto_visualizations()
        elif viz_type == "histogram":
            self._create_histograms()
        elif viz_type == "bar":
            self._create_bar_plots()
        else:
            return f"Unsupported visualization type: {viz_type}"
            
        return f"Created {len(self.visualizations)} visualizations"
    
    def _create_auto_visualizations(self) -> None:
        """Create automatic visualizations based on data characteristics."""
        # Correlation heatmap for numeric data
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        if len(numeric_cols) >= 2:
            self._plot_heatmap(numeric_cols)
        
        # Individual column visualizations
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self._plot_histogram(col)
            elif self.df[col].nunique() < 20:
                self._plot_bar_chart(col)
    
    def _plot_heatmap(self, columns) -> None:
        """Create correlation heatmap for numeric columns."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.df[columns].corr(), 
            annot=True, 
            cmap='coolwarm',
            center=0
        )
        plt.title("Correlation Heatmap")
        self._save_viz("heatmap")
    
    def _plot_histogram(self, column: str) -> None:
        """Create histogram for numeric column."""
        plt.figure()
        sns.histplot(self.df[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        self._save_viz(f"hist_{column}")
    
    def _plot_bar_chart(self, column: str) -> None:
        """Create bar plot for categorical column."""
        plt.figure()
        self.df[column].value_counts().plot(kind='bar')
        plt.title(f"Count of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        self._save_viz(f"bar_{column}")
    
    def _save_viz(self, name: str) -> None:
        """Save visualization to buffer and store."""
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        self.visualizations.append((name, buf))
        plt.close()
    
    def get_visualization(self, index: int = 0) -> Optional[BytesIO]:
        """Get a visualization by index."""
        if not self.visualizations or index >= len(self.visualizations):
            return None
        return self.visualizations[index][1]
    
    def analyze_data(self, question: Optional[str] = None, api_key: Optional[str] = None) -> Dict:
        """
        Perform comprehensive data analysis.
        
        Args:
            question: Optional question for the LLM
            api_key: API key for LLM queries
            
        Returns:
            dict: Analysis results
        """
        analysis = {}
        
        if self.df is not None:
            analysis.update({
                "summary_stats": self.df.describe().to_dict(),
                "missing_values": self.df.isnull().sum().to_dict(),
                "data_types": self.df.dtypes.astype(str).to_dict(),
                "shape": self.df.shape,
                "columns": list(self.df.columns)
            })
        elif self.text_content:
            words = self.text_content.split()
            analysis.update({
                "word_count": len(words),
                "character_count": len(self.text_content),
                "unique_words": len(set(words)),
                "avg_word_length": sum(len(word) for word in words)/len(words) if words else 0
            })
        else:
            return {"error": "No data available for analysis"}
            
        if question and api_key:
            try:
                analysis["llm_response"] = self.query_llama(question, api_key)
            except Exception as e:
                analysis["llm_error"] = str(e)
                
        return analysis


def main():
    """Streamlit application for the Data Analyst Agent."""
    st.set_page_config(page_title="Data Analyst Agent", layout="wide")
    st.title("📊 Data Analyst Agent")
    st.markdown("Upload a file for analysis and visualization")
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = DataAnalystAgent()
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Together.ai API Key", type="password")
        st.markdown("---")
        st.markdown("### Supported File Types")
        st.markdown("- Tabular: CSV, Excel")
        st.markdown("- Text: TXT, PDF, DOCX")
        st.markdown("- Images: PNG, JPG (OCR)")
    
    # Main file upload and processing
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=sum(SUPPORTED_FILE_TYPES.values(), []),
        accept_multiple_files=False
    )
    
    if uploaded_file:
        # Create temp directory if it doesn't exist
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        try:
            # Save uploaded file temporarily
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the file
            with st.spinner("Processing file..."):
                status = st.session_state.agent.process_file(temp_path)
                st.success(status)
            
            # Create tabs for different functionalities
            tab1, tab2, tab3 = st.tabs(["📈 Analysis", "❓ Q&A", "📊 Visualizations"])
            
            with tab1:
                st.header("Data Analysis")
                analysis = st.session_state.agent.analyze_data()
                st.json(analysis)
                
            with tab2:
                st.header("Ask Questions")
                question = st.text_area("Enter your question about the data")
                if question and api_key:
                    with st.spinner("Generating answer..."):
                        try:
                            answer = st.session_state.agent.query_llama(question, api_key)
                            st.markdown("### Answer")
                            st.write(answer)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                elif question and not api_key:
                    st.warning("Please enter your API key in the sidebar to ask questions")
                
            with tab3:
                st.header("Data Visualizations")
                if st.button("Generate Automatic Visualizations"):
                    with st.spinner("Creating visualizations..."):
                        st.session_state.agent.create_visualization()
                        
                    for name, viz in st.session_state.agent.visualizations:
                        st.subheader(name.replace('_', ' ').title())
                        st.image(viz, use_column_width=True)
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    main()