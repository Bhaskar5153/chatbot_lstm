import fitz # pip install PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
import re

class PDFExtraction:
    def __init__(self):
        pass

    

    def extract_text(self, pdf_path: str):
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError("PDF not found")
        
        doc = fitz.open(str(path))
        pages = []

        for page in doc:
            text = page.get_text()
            cleaned = self._clean_page(text)
            if cleaned:
                pages.append(cleaned)

        doc.close()

        return "\n".join(pages)
    
    def _clean_page(self, text):
        # Remove URLs (http/htpps)
        cleaned_text = re.sub(r"https?://\S+", "", text)

        # Remove page footers/headers
        cleaned_text = re.sub(r"SUPREME COURT OF INDIA", "", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"Page \d+ of \d+", "", cleaned_text, flags=re.IGNORECASE)

        # Remove standardlone numbers (page numbers like "554")
        cleaned_text = re.sub(r"^\s*\d+\s*$", "", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\d+.", "", cleaned_text)
        cleaned_text = re.sub(r"x+\s", "", cleaned_text)


        # Remove extra whitespaces
        cleaned_text = re.sub(r"\n{2,}", "\n", cleaned_text)
        cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)

        return cleaned_text.strip()
    



        

# pdf_extract = PDFExtraction()
# print(pdf_extract.extract_text(r"C:\Users\Priya Bhaskar\OneDrive\Documents\project_6_chat_lstm\chatbot_lstm\app\pdf\-0___jonew__judis__10796.pdf"))

