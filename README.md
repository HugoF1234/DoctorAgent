# Agent de Diagnostic Médical

Application interactive de diagnostic médical utilisant des Agents LLM avec techniques de raisonnement avancées, développée avec Streamlit.

## Description du Projet

Cet agent de diagnostic médical permet aux utilisateurs de décrire leurs symptômes et reçoit en retour :
- Des hypothèses de pathologies probables
- Des questions complémentaires pour affiner le diagnostic
- Des justifications pour chaque hypothèse
- Un classement des pathologies par probabilité

L'agent utilise plusieurs techniques de raisonnement avancées pour garantir une analyse rigoureuse et transparente.

## Techniques de Raisonnement Implémentées

### 1. Chain of Thought (CoT)

**Principe** : Force le modèle à décomposer le problème étape par étape.

**Implémentation** : L'agent analyse les symptômes en suivant une séquence logique :
1. Analyse des symptômes mentionnés
2. Identification des patterns et associations
3. Considération des pathologies possibles
4. Évaluation de la probabilité de chaque pathologie
5. Détermination des informations supplémentaires nécessaires
6. Formulation de la réponse ou des questions

**Avantage** : Rend le raisonnement transparent et structuré.

### 2. Tree of Thoughts (ToT)

**Principe** : Génère plusieurs pistes de solutions, les évalue, et conserve seulement la plus prometteuse (élagage).

**Implémentation** : 
- Génération de 3-5 hypothèses de pathologies différentes
- Évaluation de chaque hypothèse (probabilité, cohérence, plausibilité)
- Élagage : conservation des 2-3 hypothèses les plus prometteuses
- Justification de chaque hypothèse retenue

**Avantage** : Explore plusieurs possibilités avant de se concentrer sur les plus probables.

### 3. ReAct (Reason + Act)

**Principe** : Boucle itérative Pensée → Action (Outil) → Observation → Réponse.

**Implémentation** : 
- **THOUGHT** : Analyse les informations disponibles
- **ACTION** : Décide de la prochaine action (poser une question, générer une hypothèse, demander clarification)
- **OBSERVATION** : Évalue ce qui est observé
- **RESPONSE** : Formule la réponse au patient

**Avantage** : Permet une interaction dynamique et adaptative avec l'utilisateur.

### 4. Self-Correction (Réflexion)

**Principe** : L'agent produit une première réponse, la critique lui-même pour trouver des erreurs, et génère une version finale corrigée.

**Implémentation** :
1. Génération d'une première analyse diagnostique
2. Auto-critique : identification des erreurs potentielles (hallucinations, logique incorrecte, informations manquantes, incohérences)
3. Génération d'une version corrigée et améliorée

**Avantage** : Réduit les erreurs et améliore la qualité du diagnostic.

### 5. Mode Hybride

Combine plusieurs techniques pour une analyse plus robuste :
- CoT pour l'analyse initiale structurée
- ToT pour générer et évaluer plusieurs hypothèses
- Self-Correction pour affiner les résultats

## Installation

### Prérequis

- Python 3.8 ou supérieur
- Une clé API Google Gemini (obtenable sur [Google AI Studio](https://makersuite.google.com/app/apikey))

### Étapes d'installation

1. **Cloner le projet**

```bash
git clone https://github.com/HugoF1234/DoctorAgent.git
cd DoctorAgent
```

2. **Créer l'environnement virtuel**

**Sur macOS/Linux :**
```bash
python3 -m venv venv
```

**Sur Windows :**
```bash
python -m venv venv
```

3. **Activer l'environnement virtuel**

**Sur macOS/Linux :**
```bash
source venv/bin/activate
```

**Sur Windows :**
```bash
venv\Scripts\activate
```

4. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

5. **Configurer la clé API**

La clé API est déjà configurée dans le code. Si vous souhaitez utiliser une autre clé, modifiez la variable `GEMINI_API_KEY` dans le fichier `app.py`.

6. **Lancer l'application**

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

**Note importante** : N'oubliez pas d'activer l'environnement virtuel à chaque fois que vous travaillez sur le projet :
- macOS/Linux : `source venv/bin/activate`
- Windows : `venv\Scripts\activate`

## Utilisation

1. **Choisir la technique de raisonnement** dans la barre latérale (ReAct recommandé pour commencer)
2. **Uploader des documents** (optionnel) : PDF, TXT, MD dans la section Documents de la sidebar
3. **Décrire vos symptômes** dans le champ de chat
4. **Répondre aux questions** de l'agent pour affiner le diagnostic
5. **Consulter les hypothèses** dans la section "État du Diagnostic"
6. **Voir le raisonnement détaillé** en cliquant sur l'accordéon "Processus de Raisonnement"

### Exemple d'utilisation

```
Utilisateur : "J'ai des maux de tête intenses depuis 3 jours, avec des nausées et une sensibilité à la lumière."

Agent : [Analyse avec ReAct]
- THOUGHT: Analyse des symptômes (céphalée, nausées, photophobie)
- ACTION: Générer des hypothèses et poser des questions complémentaires
- OBSERVATION: Pattern suggérant une migraine ou une méningite
- RESPONSE: "Ces symptômes suggèrent plusieurs possibilités. Avez-vous de la fièvre ? Des raideurs dans la nuque ?"
```

## Structure du Projet

```
.
├── app.py                 # Application Streamlit principale
├── agent.py               # Classe DiagnosticAgent avec logique de diagnostic
├── reasoning.py           # Implémentations des techniques de raisonnement
├── document_processor.py  # Traitement des documents uploadés (PDF, TXT)
├── requirements.txt       # Dépendances Python
├── README.md             # Documentation du projet
└── venv/                 # Environnement virtuel (à créer)
```

## Avertissements Importants

**Cet agent est un outil éducatif et de démonstration. Il ne remplace en aucun cas un avis médical professionnel.**

- Les diagnostics générés sont des hypothèses basées sur des patterns
- Toujours consulter un professionnel de santé pour un diagnostic réel
- Ne pas utiliser pour des situations d'urgence médicale

## Fonctionnalités

- Interface interactive de chat
- Extraction automatique des symptômes
- Génération d'hypothèses de pathologies
- Questions complémentaires adaptatives
- Justification des diagnostics
- Classement par probabilité
- Affichage du raisonnement détaillé (dans un accordéon)
- Support de plusieurs techniques de raisonnement
- Upload et traitement de documents (PDF, TXT, MD)
- Historique de conversation
- État du diagnostic en temps réel

## Technologies Utilisées

- **Streamlit** : Interface utilisateur interactive
- **Google Gemini API** : Modèle de langage pour le raisonnement
- **PyPDF2** : Traitement des fichiers PDF
- **Python** : Langage de programmation

Realisé par Matthieu HOUETTE, Victor LESTRADE et Hugo FOUAN