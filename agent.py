import google.generativeai as genai
from reasoning import ReasoningTechniques
import json
import re

class DiagnosticAgent:
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        
        try:
            models = genai.list_models()
            model_name = None
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    model_name = model.name.replace('models/', '')
                    break
            
            if model_name:
                self.model = genai.GenerativeModel(model_name)
            else:
                self.model = genai.GenerativeModel()
        except Exception as e:
            try:
                self.model = genai.GenerativeModel('gemini-pro')
            except:
                self.model = genai.GenerativeModel()
        
        self.reasoning = ReasoningTechniques(self.model)
    
    def process_user_input(self, user_input: str, reasoning_technique: str, 
                          conversation_history: list, diagnosis_state: dict, 
                          documents: list = None) -> dict:
        
        symptoms = self._extract_symptoms(user_input, diagnosis_state["symptoms"])
        diagnosis_state["symptoms"].extend(symptoms)
        
        if reasoning_technique == "react":
            return self._react_reasoning(user_input, conversation_history, diagnosis_state, documents)
        elif reasoning_technique == "cot":
            return self._cot_reasoning(user_input, conversation_history, diagnosis_state, documents)
        elif reasoning_technique == "tot":
            return self._tot_reasoning(user_input, conversation_history, diagnosis_state, documents)
        elif reasoning_technique == "self_correction":
            return self._self_correction_reasoning(user_input, conversation_history, diagnosis_state, documents)
        elif reasoning_technique == "hybrid":
            return self._hybrid_reasoning(user_input, conversation_history, diagnosis_state, documents)
        else:
            return self._react_reasoning(user_input, conversation_history, diagnosis_state, documents)
    
    def _extract_symptoms(self, user_input: str, existing_symptoms: list) -> list:
        prompt = f"""
        Analyse ce message et extrais uniquement les symptômes médicaux mentionnés.
        Liste les symptômes sous forme de liste simple, sans explication.
        
        Message: {user_input}
        Symptômes déjà identifiés: {', '.join(existing_symptoms) if existing_symptoms else 'Aucun'}
        
        Réponds uniquement avec une liste de symptômes séparés par des virgules, ou "Aucun nouveau symptôme" si aucun n'est trouvé.
        """
        
        try:
            response = self.model.generate_content(prompt)
            symptoms_text = response.text.strip()
            
            if "aucun" in symptoms_text.lower():
                return []
            
            symptoms = [s.strip() for s in symptoms_text.split(',')]
            return [s for s in symptoms if s and s not in existing_symptoms]
        except:
            return []
    
    def _react_reasoning(self, user_input: str, conversation_history: list, 
                        diagnosis_state: dict, documents: list = None) -> dict:
        
        context = self._build_context(conversation_history, diagnosis_state, documents)
        
        prompt = f"""
        Tu es un agent de diagnostic médical expert. Utilise la méthode ReAct (Reason + Act).
        
        IMPORTANT: Consulte attentivement TOUT l'historique de conversation ci-dessous. Ne perds aucune information des échanges précédents.
        
        CONTEXTE ACTUEL (inclut l'historique complet):
        {context}
        
        NOUVELLE INFORMATION DU PATIENT:
        {user_input}
        
        PROCÉDURE REACT:
        1. PENSÉE (Thought) : Analyse TOUTES les informations disponibles, y compris tout l'historique de conversation
        2. ACTION (Action) : Décide de la prochaine action en tenant compte de tout ce qui a été dit précédemment
        3. OBSERVATION (Observation) : Évalue ce que tu observes en référence à l'historique complet
        4. RÉPONSE (Response) : Formule ta réponse au patient en cohérence avec tout l'historique
        
        Format ta réponse ainsi:
        THOUGHT: [ton analyse]
        ACTION: [ton action]
        OBSERVATION: [ton observation]
        RESPONSE: [ta réponse au patient]
        
        Si tu as assez d'informations, génère des hypothèses de pathologies avec leurs probabilités et justifications.
        """
        
        response = self.model.generate_content(prompt)
        response_text = response.text
        
        reasoning_details = self._parse_react_response(response_text)
        message = reasoning_details.get("RESPONSE", response_text)
        
        if "hypotheses" in reasoning_details:
            diagnosis_state["hypotheses"] = reasoning_details["hypotheses"]
        
        return {
            "message": message,
            "reasoning_details": reasoning_details,
            "diagnosis_state": diagnosis_state
        }
    
    def _cot_reasoning(self, user_input: str, conversation_history: list, 
                      diagnosis_state: dict, documents: list = None) -> dict:
        
        context = self._build_context(conversation_history, diagnosis_state, documents)
        
        prompt = f"""
        Tu es un agent de diagnostic médical. Utilise le Chain of Thought (CoT) pour analyser étape par étape.
        
        IMPORTANT: Consulte attentivement TOUT l'historique de conversation ci-dessous. Ne perds aucune information des échanges précédents.
        
        CONTEXTE (inclut l'historique complet):
        {context}
        
        NOUVELLE INFORMATION:
        {user_input}
        
        PENSE ÉTAPE PAR ÉTAPE:
        1. Analyse TOUS les symptômes mentionnés dans l'historique complet
        2. Identifie les patterns et associations en tenant compte de tout l'historique
        3. Considère les pathologies possibles en référence à toutes les informations précédentes
        4. Évalue la probabilité de chaque pathologie en utilisant tout le contexte
        5. Détermine quelles informations supplémentaires sont nécessaires (sans répéter ce qui a déjà été demandé)
        6. Formule ta réponse ou tes questions en cohérence avec tout l'historique
        
        Présente ton raisonnement étape par étape, puis donne ta réponse finale au patient.
        """
        
        response = self.model.generate_content(prompt)
        response_text = response.text
        
        reasoning_details = self._extract_cot_reasoning(response_text)
        
        return {
            "message": reasoning_details.get("final_response", response_text),
            "reasoning_details": reasoning_details,
            "diagnosis_state": diagnosis_state
        }
    
    def _tot_reasoning(self, user_input: str, conversation_history: list, 
                      diagnosis_state: dict, documents: list = None) -> dict:
        
        context = self._build_context(conversation_history, diagnosis_state, documents)
        
        prompt = f"""
        Tu es un agent de diagnostic médical. Utilise le Tree of Thoughts (ToT) pour explorer plusieurs pistes.
        
        IMPORTANT: Consulte attentivement TOUT l'historique de conversation ci-dessous. Ne perds aucune information des échanges précédents.
        
        CONTEXTE (inclut l'historique complet):
        {context}
        
        NOUVELLE INFORMATION:
        {user_input}
        
        MÉTHODE TOT:
        1. GÉNÈRE 3-5 hypothèses de pathologies différentes en tenant compte de TOUT l'historique
        2. ÉVALUE chaque hypothèse (probabilité, cohérence, plausibilité) en référence à toutes les informations précédentes
        3. ÉLAGUE : Garde seulement les 2-3 hypothèses les plus prometteuses
        4. JUSTIFIE chaque hypothèse retenue en utilisant tout le contexte disponible
        5. DÉTERMINE les questions à poser pour affiner le diagnostic (sans répéter ce qui a déjà été demandé)
        
        Format:
        HYPOTHÈSES GÉNÉRÉES:
        - Pathologie 1: [nom] - Probabilité: [X%] - Justification: [raison]
        - Pathologie 2: [nom] - Probabilité: [X%] - Justification: [raison]
        ...
        
        HYPOTHÈSES RETENUES (après élagage):
        - [Liste des meilleures hypothèses]
        
        RÉPONSE AU PATIENT:
        [Ta réponse]
        """
        
        response = self.model.generate_content(prompt)
        response_text = response.text
        
        reasoning_details = self._parse_tot_response(response_text)
        
        if "hypotheses" in reasoning_details:
            diagnosis_state["hypotheses"] = reasoning_details["hypotheses"]
        
        return {
            "message": reasoning_details.get("response", response_text),
            "reasoning_details": reasoning_details,
            "diagnosis_state": diagnosis_state
        }
    
    def _self_correction_reasoning(self, user_input: str, conversation_history: list, 
                                  diagnosis_state: dict, documents: list = None) -> dict:
        
        context = self._build_context(conversation_history, diagnosis_state, documents)
        
        initial_prompt = f"""
        IMPORTANT: Consulte attentivement TOUT l'historique de conversation ci-dessous. Ne perds aucune information des échanges précédents.
        
        CONTEXTE (inclut l'historique complet):
        {context}
        
        NOUVELLE INFORMATION:
        {user_input}
        
        Génère une première analyse diagnostique basée sur TOUTES ces informations, en tenant compte de tout l'historique de conversation.
        """
        
        initial_response = self.model.generate_content(initial_prompt)
        initial_text = initial_response.text
        
        critique_prompt = f"""
        Tu es un agent de diagnostic médical. Critique cette première analyse et identifie:
        1. Les erreurs potentielles (hallucinations, logique incorrecte)
        2. Les informations manquantes
        3. Les incohérences avec l'historique complet de conversation
        4. Les améliorations possibles
        
        IMPORTANT: Vérifie que l'analyse tient compte de TOUT l'historique de conversation.
        
        PREMIÈRE ANALYSE:
        {initial_text}
        
        CONTEXTE COMPLET (inclut tout l'historique):
        {context}
        
        NOUVELLE INFORMATION:
        {user_input}
        
        Liste les problèmes identifiés et suggère des corrections en tenant compte de tout l'historique.
        """
        
        critique_response = self.model.generate_content(critique_prompt)
        critique_text = critique_response.text
        
        corrected_prompt = f"""
        IMPORTANT: Consulte attentivement TOUT l'historique de conversation ci-dessous.
        
        PREMIÈRE ANALYSE:
        {initial_text}
        
        CRITIQUE:
        {critique_text}
        
        CONTEXTE COMPLET (inclut tout l'historique):
        {context}
        
        Génère une version corrigée et améliorée de l'analyse diagnostique en tenant compte de la critique ET de tout l'historique de conversation.
        """
        
        corrected_response = self.model.generate_content(corrected_prompt)
        corrected_text = corrected_response.text
        
        reasoning_details = {
            "initial_analysis": initial_text,
            "critique": critique_text,
            "corrected_analysis": corrected_text
        }
        
        return {
            "message": corrected_text,
            "reasoning_details": reasoning_details,
            "diagnosis_state": diagnosis_state
        }
    
    def _hybrid_reasoning(self, user_input: str, conversation_history: list, 
                         diagnosis_state: dict, documents: list = None) -> dict:
        
        cot_result = self._cot_reasoning(user_input, conversation_history, diagnosis_state, documents)
        
        tot_result = self._tot_reasoning(user_input, conversation_history, diagnosis_state, documents)
        
        if tot_result["reasoning_details"].get("hypotheses"):
            synthesis_prompt = f"""
            IMPORTANT: Consulte attentivement TOUT l'historique de conversation pour maintenir la cohérence.
            
            Synthétise ces analyses pour donner une réponse finale cohérente au patient en tenant compte de tout l'historique.
            
            ANALYSE CoT:
            {cot_result['reasoning_details']}
            
            HYPOTHÈSES ToT:
            {tot_result['reasoning_details']}
            
            CONTEXTE COMPLET (inclut tout l'historique):
            {self._build_context(conversation_history, diagnosis_state, documents)}
            
            Formule une réponse claire et structurée qui est cohérente avec tout l'historique de conversation.
            """
            
            synthesis_response = self.model.generate_content(synthesis_prompt)
            
            return {
                "message": synthesis_response.text,
                "reasoning_details": {
                    "cot_analysis": cot_result["reasoning_details"],
                    "tot_hypotheses": tot_result["reasoning_details"],
                    "synthesis": synthesis_response.text
                },
                "diagnosis_state": diagnosis_state
            }
        
        return cot_result
    
    def _build_context(self, conversation_history: list, diagnosis_state: dict, documents: list = None) -> str:
        context_parts = []
        
        if diagnosis_state["symptoms"]:
            context_parts.append(f"Symptômes identifiés: {', '.join(diagnosis_state['symptoms'])}")
        
        if diagnosis_state["hypotheses"]:
            context_parts.append("Hypothèses actuelles:")
            for h in diagnosis_state["hypotheses"]:
                context_parts.append(f"  - {h.get('pathology', 'Inconnue')}: {h.get('probability', 'N/A')}")
        
        if documents:
            context_parts.append("\n📄 DOCUMENTS FOURNIS:")
            for doc in documents:
                context_parts.append(f"\n--- Document: {doc.get('name', 'Sans nom')} ({doc.get('type', 'unknown')}) ---")
                content = doc.get('content', '')
                if len(content) > 2000:
                    content = content[:2000] + "... [contenu tronqué]"
                context_parts.append(content)
        
        if conversation_history:
            context_parts.append("\nHISTORIQUE COMPLET DE LA CONVERSATION:")
            for i, msg in enumerate(conversation_history, 1):
                role_name = "PATIENT" if msg['role'] == 'user' else "AGENT"
                content = msg['content']
                context_parts.append(f"\n[{i}] {role_name}: {content}")
        
        return "\n".join(context_parts) if context_parts else "Aucun contexte précédent"
    
    def _parse_react_response(self, response_text: str) -> dict:
        result = {
            "THOUGHT": "",
            "ACTION": "",
            "OBSERVATION": "",
            "RESPONSE": ""
        }
        
        sections = ["THOUGHT", "ACTION", "OBSERVATION", "RESPONSE"]
        current_section = None
        current_text = []
        
        for line in response_text.split('\n'):
            line_upper = line.strip().upper()
            found_section = None
            
            for section in sections:
                if line_upper.startswith(section + ":"):
                    if current_section:
                        result[current_section] = "\n".join(current_text).strip()
                    current_section = section
                    current_text = [line.split(":", 1)[1].strip()] if ":" in line else []
                    found_section = section
                    break
            
            if not found_section and current_section:
                current_text.append(line)
        
        if current_section:
            result[current_section] = "\n".join(current_text).strip()
        
        if not result["RESPONSE"]:
            result["RESPONSE"] = response_text
        
        return result
    
    def _extract_cot_reasoning(self, response_text: str) -> dict:
        steps = []
        final_response = ""
        
        lines = response_text.split('\n')
        in_reasoning = True
        
        for line in lines:
            if re.match(r'^\d+\.', line.strip()):
                steps.append(line.strip())
            elif "réponse" in line.lower() or "conclusion" in line.lower():
                in_reasoning = False
            elif not in_reasoning:
                final_response += line + "\n"
        
        if not final_response:
            final_response = response_text
        
        return {
            "steps": steps,
            "final_response": final_response.strip()
        }
    
    def _parse_tot_response(self, response_text: str) -> dict:
        hypotheses = []
        response = ""
        
        in_hypotheses = False
        in_response = False
        
        for line in response_text.split('\n'):
            if "hypothèses" in line.lower() and "générées" in line.lower():
                in_hypotheses = True
                continue
            elif "hypothèses" in line.lower() and "retenues" in line.lower():
                in_hypotheses = True
                continue
            elif "réponse" in line.lower() and "patient" in line.lower():
                in_hypotheses = False
                in_response = True
                continue
            
            if in_hypotheses and line.strip().startswith('-'):
                match = re.search(r'Pathologie\s+\d+:\s*([^-]+)\s*-\s*Probabilité:\s*([^-]+)\s*-\s*Justification:\s*(.+)', line)
                if match:
                    hypotheses.append({
                        "pathology": match.group(1).strip(),
                        "probability": match.group(2).strip(),
                        "justification": match.group(3).strip()
                    })
            elif in_response:
                response += line + "\n"
        
        if not response:
            response = response_text
        
        return {
            "hypotheses": hypotheses,
            "response": response.strip()
        }
