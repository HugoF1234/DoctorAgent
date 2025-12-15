"""
Module pour traiter les documents uploadés (PDF, texte)
"""

import PyPDF2
import io
from typing import List, Dict

def extract_text_from_pdf(file) -> str:
    """Extrait le texte d'un fichier PDF"""
    try:
        # Pour les fichiers Streamlit, utiliser BytesIO
        if hasattr(file, 'read'):
            file_bytes = file.read()
            file.seek(0)  # Réinitialiser pour les prochaines lectures
            pdf_file = io.BytesIO(file_bytes)
        else:
            pdf_file = file
        
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Erreur lors de l'extraction du PDF : {str(e)}"

def extract_text_from_txt(file) -> str:
    """Extrait le texte d'un fichier texte"""
    try:
        # Lire le fichier en mode texte
        if hasattr(file, 'read'):
            file.seek(0)  # Réinitialiser le pointeur
            text = file.read()
            if isinstance(text, bytes):
                text = text.decode('utf-8')
        elif isinstance(file, bytes):
            text = file.decode('utf-8')
        else:
            text = str(file)
        return text
    except Exception as e:
        return f"Erreur lors de la lecture du fichier texte : {str(e)}"

def process_uploaded_file(uploaded_file) -> Dict[str, str]:
    """
    Traite un fichier uploadé et retourne son contenu
    
    Returns:
        Dict avec 'name', 'type', 'content'
    """
    file_name = uploaded_file.name
    file_type = file_name.split('.')[-1].lower() if '.' in file_name else 'unknown'
    
    # Réinitialiser le pointeur du fichier
    uploaded_file.seek(0)
    
    if file_type == 'pdf':
        content = extract_text_from_pdf(uploaded_file)
    elif file_type in ['txt', 'text', 'md']:
        content = extract_text_from_txt(uploaded_file)
    else:
        content = f"Type de fichier non supporté : {file_type}"
    
    return {
        "name": file_name,
        "type": file_type,
        "content": content
    }

def process_multiple_files(uploaded_files: List) -> List[Dict[str, str]]:
    """Traite plusieurs fichiers uploadés"""
    processed_files = []
    for uploaded_file in uploaded_files:
        processed_files.append(process_uploaded_file(uploaded_file))
    return processed_files

