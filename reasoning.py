class ReasoningTechniques:
    
    def __init__(self, client, model_name: str):
        self.client = client
        self.model = model_name
    
    def _generate_content(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.complete(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content
    
    def chain_of_thought(self, prompt: str) -> str:
        cot_prompt = f"""
        {prompt}
        
        Pense étape par étape:
        1. Analyse les données
        2. Identifie les variables
        3. Calcule la solution
        """
        return self._generate_content(cot_prompt)
    
    def tree_of_thoughts(self, prompt: str, num_branches: int = 3) -> list:
        branches = []
        
        for i in range(num_branches):
            branch_prompt = f"""
            {prompt}
            
            Explore une piste de solution différente. Piste {i+1}:
            """
            solution = self._generate_content(branch_prompt)
            branches.append({
                "branch": i + 1,
                "solution": solution
            })
        
        evaluation_prompt = f"""
        Évalue ces {num_branches} pistes de solutions et garde seulement la meilleure:
        
        {chr(10).join([f"Piste {b['branch']}: {b['solution']}" for b in branches])}
        
        Identifie la meilleure piste et explique pourquoi.
        """
        
        evaluation = self._generate_content(evaluation_prompt)
        
        return {
            "branches": branches,
            "evaluation": evaluation,
            "best_branch": evaluation
        }
    
    def react_loop(self, initial_prompt: str, max_iterations: int = 3) -> dict:
        history = []
        
        for i in range(max_iterations):
            thought_prompt = f"""
            {initial_prompt}
            
            Itération {i+1}:
            THOUGHT: [Analyse la situation]
            ACTION: [Décide de l'action]
            OBSERVATION: [Observe le résultat]
            """
            
            response = self._generate_content(thought_prompt)
            history.append({
                "iteration": i + 1,
                "response": response
            })
        
        return {
            "iterations": history,
            "final_response": history[-1]["response"] if history else ""
        }
    
    def self_correction(self, initial_response: str, context: str) -> dict:
        critique_prompt = f"""
        Critique cette réponse et identifie les erreurs:
        
        Réponse initiale:
        {initial_response}
        
        Contexte:
        {context}
        
        Liste les problèmes et erreurs.
        """
        
        critique = self._generate_content(critique_prompt)
        
        correction_prompt = f"""
        Réponse initiale:
        {initial_response}
        
        Critique:
        {critique}
        
        Génère une version corrigée.
        """
        
        corrected = self._generate_content(correction_prompt)
        
        return {
            "initial": initial_response,
            "critique": critique,
            "corrected": corrected
        }
