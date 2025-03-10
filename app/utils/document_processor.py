import os
import logging
import re
from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF
from datetime import datetime
import pandas as pd
from rapidfuzz import process, fuzz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Process PDF documents for bond information extraction
    """
    def __init__(self, output_dir: str = "processed_docs"):
        """
        Initialize the document processor
        
        Args:
            output_dir (str): Directory to store processed documents
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Patterns for extracting information
        self.patterns = {
            "isin": r"ISIN:?\s*([A-Z0-9]{12})",
            "issuer": r"Issuer:?\s*([^\n]+)",
            "maturity_date": r"Maturity\s*Date:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{2,4})",
            "coupon_rate": r"Coupon:?\s*(\d+\.?\d*)\s*%",
            "face_value": r"Face\s*Value:?\s*(?:Rs\.?|INR)?\s*(\d+(?:,\d+)*(?:\.\d+)?)",
            "yield": r"Yield:?\s*(\d+\.?\d*)\s*%",
            "payment_frequency": r"Payment\s*Frequency:?\s*([^\n]+)",
            "rating": r"Rating:?\s*([A-Za-z0-9+]+(?:\s*\([^)]+\))?)"
        }
        
    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Process a PDF file to extract bond information
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            Dict[str, Any]: Extracted bond information
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"error": "File not found"}
        
        try:
            # Extract text from PDF
            text = self._extract_text(file_path)
            
            # Extract entities from text
            entities = self._extract_entities(text)
            
            # Extract tables from PDF
            tables = self._extract_tables(file_path)
            
            # Process cashflow table if available
            cashflow = None
            if "cashflow" in tables:
                cashflow = self._process_cashflow_table(tables["cashflow"])
            
            # Combine all extracted information
            result = {
                "metadata": entities,
                "cashflow": cashflow,
                "raw_text": text[:1000] + "..." if len(text) > 1000 else text,
                "processed_at": datetime.now().isoformat()
            }
            
            # Save processed result
            output_file = os.path.join(self.output_dir, os.path.basename(file_path).replace(".pdf", ".json"))
            import json
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return {"error": str(e)}
    
    def _extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF file
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text
        """
        text = ""
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            return ""
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from text using regex patterns
        
        Args:
            text (str): Text to extract entities from
            
        Returns:
            Dict[str, Any]: Extracted entities
        """
        entities = {}
        
        for entity_type, pattern in self.patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities[entity_type] = match.group(1).strip()
        
        # Process dates if found
        if "maturity_date" in entities:
            try:
                # Try different date formats
                date_formats = [
                    "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y",
                    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y"
                ]
                
                for date_format in date_formats:
                    try:
                        entities["maturity_date"] = datetime.strptime(
                            entities["maturity_date"], date_format
                        ).isoformat()
                        break
                    except ValueError:
                        continue
            except Exception as e:
                logger.warning(f"Could not parse maturity date: {entities['maturity_date']}, {str(e)}")
        
        # Convert numeric values
        for field in ["coupon_rate", "yield"]:
            if field in entities:
                try:
                    entities[field] = float(entities[field])
                except ValueError:
                    logger.warning(f"Could not convert {field} to float: {entities[field]}")
        
        if "face_value" in entities:
            try:
                # Remove commas and convert to float
                entities["face_value"] = float(entities["face_value"].replace(",", ""))
            except ValueError:
                logger.warning(f"Could not convert face_value to float: {entities['face_value']}")
        
        return entities
    
    def _extract_tables(self, file_path: str) -> Dict[str, List[List[str]]]:
        """
        Extract tables from a PDF file
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            Dict[str, List[List[str]]]: Extracted tables
        """
        tables = {}
        
        try:
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    # Extract tables using PyMuPDF
                    tab = page.find_tables()
                    if tab.tables:
                        for i, table in enumerate(tab.tables):
                            table_data = []
                            for row in range(table.rows):
                                row_data = []
                                for col in range(table.cols):
                                    cell = table.cell(row, col)
                                    if cell:
                                        row_data.append(cell.text.strip())
                                    else:
                                        row_data.append("")
                                table_data.append(row_data)
                            
                            # Determine table type based on headers
                            table_type = self._determine_table_type(table_data)
                            tables[table_type] = table_data
        
        except Exception as e:
            logger.error(f"Error extracting tables from PDF {file_path}: {str(e)}")
        
        return tables
    
    def _determine_table_type(self, table_data: List[List[str]]) -> str:
        """
        Determine the type of table based on headers
        
        Args:
            table_data (List[List[str]]): Table data
            
        Returns:
            str: Table type
        """
        if not table_data or not table_data[0]:
            return "unknown"
        
        # Convert headers to lowercase for easier matching
        headers = [h.lower() for h in table_data[0]]
        
        # Check for cashflow table
        cashflow_keywords = ["payment", "date", "amount", "coupon", "principal"]
        if any(keyword in " ".join(headers) for keyword in cashflow_keywords):
            return "cashflow"
        
        # Check for financial metrics table
        financial_keywords = ["ratio", "metric", "financial", "value"]
        if any(keyword in " ".join(headers) for keyword in financial_keywords):
            return "financial_metrics"
        
        # Check for issuer information table
        issuer_keywords = ["issuer", "company", "sector", "industry"]
        if any(keyword in " ".join(headers) for keyword in issuer_keywords):
            return "issuer_info"
        
        return "unknown"
    
    def _process_cashflow_table(self, table_data: List[List[str]]) -> Optional[List[Dict[str, Any]]]:
        """
        Process a cashflow table
        
        Args:
            table_data (List[List[str]]): Cashflow table data
            
        Returns:
            Optional[List[Dict[str, Any]]]: Processed cashflow data
        """
        if not table_data or len(table_data) < 2:
            return None
        
        # Get headers
        headers = [h.lower() for h in table_data[0]]
        
        # Find date and amount columns
        date_col = None
        amount_col = None
        
        for i, header in enumerate(headers):
            if "date" in header:
                date_col = i
            elif any(keyword in header for keyword in ["amount", "payment", "coupon"]):
                amount_col = i
        
        if date_col is None or amount_col is None:
            return None
        
        # Process rows
        cashflow_data = []
        for row in table_data[1:]:
            if len(row) <= max(date_col, amount_col):
                continue
            
            try:
                # Parse date
                date_str = row[date_col].strip()
                date_formats = [
                    "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y",
                    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y"
                ]
                
                payment_date = None
                for date_format in date_formats:
                    try:
                        payment_date = datetime.strptime(date_str, date_format).isoformat()
                        break
                    except ValueError:
                        continue
                
                if not payment_date:
                    continue
                
                # Parse amount
                amount_str = row[amount_col].strip()
                # Remove currency symbols and commas
                amount_str = re.sub(r'[^\d.]', '', amount_str)
                amount = float(amount_str) if amount_str else 0.0
                
                cashflow_data.append({
                    "payment_date": payment_date,
                    "amount": amount
                })
            
            except Exception as e:
                logger.warning(f"Error processing cashflow row {row}: {str(e)}")
        
        return cashflow_data
    
    def extract_entities_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from text
        
        Args:
            text (str): Text to extract entities from
            
        Returns:
            Dict[str, Any]: Extracted entities
        """
        return self._extract_entities(text)
    
    def match_entity_to_database(self, entity: str, entity_type: str, database: pd.DataFrame, 
                                column: str, threshold: float = 85.0) -> Optional[str]:
        """
        Match an extracted entity to a database entry
        
        Args:
            entity (str): Entity to match
            entity_type (str): Type of entity
            database (pd.DataFrame): Database to match against
            column (str): Column in database to match against
            threshold (float): Matching threshold
            
        Returns:
            Optional[str]: Matched entity or None if no match found
        """
        if entity is None or database is None or database.empty:
            return None
        
        # Get all values from the column
        values = database[column].dropna().unique().tolist()
        
        # Find the best match
        match = process.extractOne(entity, values, scorer=fuzz.token_sort_ratio)
        
        if match and match[1] >= threshold:
            return match[0]
        
        return None
    
    def generate_pdf_preview(self, file_path: str, output_path: str = None, max_pages: int = 5) -> str:
        """
        Generate a preview of a PDF document
        
        Args:
            file_path (str): Path to the PDF file
            output_path (str): Path to save the preview
            max_pages (int): Maximum number of pages to include in the preview
            
        Returns:
            str: Path to the preview file
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            # If output path not provided, create one
            if not output_path:
                preview_dir = os.path.join(self.output_dir, "previews")
                os.makedirs(preview_dir, exist_ok=True)
                
                # Create output filename
                base_name = os.path.basename(file_path)
                output_path = os.path.join(preview_dir, f"preview_{base_name}")
            
            # Open the PDF
            with fitz.open(file_path) as doc:
                # Create a new PDF for the preview
                preview_doc = fitz.open()
                
                # Add pages to the preview (up to max_pages)
                for i in range(min(max_pages, len(doc))):
                    preview_doc.insert_pdf(doc, from_page=i, to_page=i)
                
                # Add a watermark to indicate it's a preview
                for page in preview_doc:
                    rect = page.rect
                    page.insert_text(
                        rect.br - (150, 20),  # Bottom right with offset
                        "PREVIEW ONLY",
                        fontsize=12,
                        color=(0.7, 0.7, 0.7)  # Light gray
                    )
                
                # Save the preview
                preview_doc.save(output_path)
                preview_doc.close()
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error generating PDF preview for {file_path}: {str(e)}")
            return None
    
    def get_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata for a document
        
        Args:
            file_path (str): Path to the document
            
        Returns:
            Dict[str, Any]: Document metadata
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"error": "File not found"}
        
        try:
            metadata = {}
            
            # Get basic file information
            file_stats = os.stat(file_path)
            metadata["file_size"] = file_stats.st_size
            metadata["created_at"] = datetime.fromtimestamp(file_stats.st_ctime).isoformat()
            metadata["modified_at"] = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            
            # Get PDF-specific information
            if file_path.lower().endswith(".pdf"):
                with fitz.open(file_path) as doc:
                    metadata["page_count"] = len(doc)
                    metadata["title"] = doc.metadata.get("title", "")
                    metadata["author"] = doc.metadata.get("author", "")
                    metadata["subject"] = doc.metadata.get("subject", "")
                    metadata["keywords"] = doc.metadata.get("keywords", "")
                    metadata["creator"] = doc.metadata.get("creator", "")
                    metadata["producer"] = doc.metadata.get("producer", "")
            
            return metadata
        
        except Exception as e:
            logger.error(f"Error getting document metadata for {file_path}: {str(e)}")
            return {"error": str(e)} 