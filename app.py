import streamlit as st
import os
from agent import DiagnosticAgent
from dotenv import load_dotenv
from document_processor import process_multiple_files

load_dotenv()

GEMINI_API_KEY = "AIzaSyAzbzoCR5eV_e3N4TqbV2SexxyJzF3ftfQ"

st.set_page_config(
    page_title="Agent de Diagnostic Médical",
    page_icon="🏥",
    layout="wide"
)

if "agent" not in st.session_state:
    st.session_state.agent = DiagnosticAgent(api_key=GEMINI_API_KEY)
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "diagnosis_state" not in st.session_state:
    st.session_state.diagnosis_state = {
        "symptoms": [],
        "hypotheses": [],
        "questions_asked": [],
        "current_step": "initial"
    }
if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

def main():
    st.title("🏥 Agent de Diagnostic Médical")
    st.markdown("""
    Cet agent utilise des techniques de raisonnement avancées pour vous aider à diagnostiquer 
    des pathologies à partir de vos symptômes.
    
    **Techniques utilisées :**
    - Chain of Thought (CoT) : Analyse étape par étape
    - Tree of Thoughts (ToT) : Exploration de plusieurs hypothèses
    - ReAct : Raisonnement et Action itératifs
    - Self-Correction : Auto-critique et amélioration
    """)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        reasoning_mode = st.selectbox(
            "Technique de raisonnement",
            ["ReAct (Recommandé)", "Chain of Thought", "Tree of Thoughts", "Self-Correction", "Hybride"],
            help="Choisissez la technique de raisonnement à utiliser"
        )
        
        st.divider()
        
        st.subheader("📄 Documents")
        uploaded_files = st.file_uploader(
            "Ajouter des documents (PDF, TXT, MD)",
            type=['pdf', 'txt', 'md'],
            accept_multiple_files=True,
            help="Vous pouvez uploader des rapports médicaux, analyses, etc."
        )
        
        if uploaded_files:
            if st.button("📥 Traiter les documents", use_container_width=True):
                with st.spinner("Traitement des documents..."):
                    processed_docs = process_multiple_files(uploaded_files)
                    st.session_state.uploaded_documents = processed_docs
                    st.success(f"✅ {len(processed_docs)} document(s) traité(s)")
                    st.rerun()
        
        if st.session_state.uploaded_documents:
            st.write(f"**Documents chargés ({len(st.session_state.uploaded_documents)}):**")
            for doc in st.session_state.uploaded_documents:
                st.write(f"📄 {doc['name']}")
            if st.button("🗑️ Supprimer tous les documents", use_container_width=True):
                st.session_state.uploaded_documents = []
                st.rerun()
        
        st.divider()
        
        if st.button("🔄 Nouveau Diagnostic", use_container_width=True):
            st.session_state.conversation = []
            st.session_state.diagnosis_state = {
                "symptoms": [],
                "hypotheses": [],
                "questions_asked": [],
                "current_step": "initial"
            }
            st.session_state.uploaded_documents = []
            st.rerun()
    
    if st.session_state.agent is None:
        st.error("❌ Erreur d'initialisation de l'agent")
        return
    
    st.header("💬 Conversation avec l'agent")
    
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.conversation):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    if "reasoning" in message and message["reasoning"]:
                        reasoning = message["reasoning"]
                        
                        if isinstance(reasoning, dict):
                            if "THOUGHT" in reasoning or "ACTION" in reasoning:
                                expander_title = "🧠 Processus de Raisonnement (ReAct)"
                            elif "steps" in reasoning:
                                expander_title = "🧠 Raisonnement Étape par Étape (CoT)"
                            elif "hypotheses" in reasoning:
                                expander_title = "🧠 Exploration des Hypothèses (ToT)"
                            elif "initial_analysis" in reasoning:
                                expander_title = "🧠 Auto-Correction"
                            else:
                                expander_title = "🧠 Détails du raisonnement"
                        else:
                            expander_title = "🧠 Détails du raisonnement"
                        
                        with st.expander(expander_title, expanded=False):
                            if isinstance(reasoning, dict):
                                if "THOUGHT" in reasoning or "ACTION" in reasoning:
                                    st.markdown("### Processus de Raisonnement (ReAct)")
                                    
                                    if reasoning.get("THOUGHT"):
                                        st.markdown("**💭 PENSÉE (Thought):**")
                                        st.info(reasoning["THOUGHT"])
                                    
                                    if reasoning.get("ACTION"):
                                        st.markdown("**⚡ ACTION:**")
                                        st.warning(reasoning["ACTION"])
                                    
                                    if reasoning.get("OBSERVATION"):
                                        st.markdown("**👁️ OBSERVATION:**")
                                        st.success(reasoning["OBSERVATION"])
                                
                                elif "steps" in reasoning:
                                    st.markdown("### Raisonnement Étape par Étape (CoT)")
                                    for i, step in enumerate(reasoning.get("steps", []), 1):
                                        st.markdown(f"**Étape {i}:** {step}")
                                
                                elif "hypotheses" in reasoning:
                                    st.markdown("### Exploration des Hypothèses (ToT)")
                                    for i, hyp in enumerate(reasoning.get("hypotheses", []), 1):
                                        st.markdown(f"**Hypothèse {i}:** {hyp.get('pathology', 'N/A')} - {hyp.get('probability', 'N/A')}")
                                
                                elif "initial_analysis" in reasoning:
                                    st.markdown("### Auto-Correction")
                                    with st.expander("📝 Analyse Initiale", expanded=False):
                                        st.write(reasoning.get("initial_analysis", ""))
                                    with st.expander("🔍 Critique", expanded=False):
                                        st.write(reasoning.get("critique", ""))
                                    with st.expander("✅ Analyse Corrigée", expanded=False):
                                        st.write(reasoning.get("corrected_analysis", ""))
                                
                                else:
                                    st.json(reasoning)
                            else:
                                st.write(reasoning)
    
    user_input = st.chat_input("Décrivez vos symptômes ou répondez aux questions...")
    
    if user_input:
        st.session_state.conversation.append({
            "role": "user",
            "content": user_input
        })
        
        reasoning_placeholder = st.empty()
        with st.spinner("🤔 L'agent analyse vos symptômes..."):
            try:
                reasoning_technique = {
                    "ReAct (Recommandé)": "react",
                    "Chain of Thought": "cot",
                    "Tree of Thoughts": "tot",
                    "Self-Correction": "self_correction",
                    "Hybride": "hybrid"
                }[reasoning_mode]
                
                with reasoning_placeholder.container():
                    st.info(f"🔄 Utilisation de la technique: **{reasoning_mode}**")
                
                response = st.session_state.agent.process_user_input(
                    user_input,
                    reasoning_technique=reasoning_technique,
                    conversation_history=st.session_state.conversation[:-1],
                    diagnosis_state=st.session_state.diagnosis_state,
                    documents=st.session_state.uploaded_documents if st.session_state.uploaded_documents else None
                )
                
                if "diagnosis_state" in response:
                    st.session_state.diagnosis_state.update(response["diagnosis_state"])
                
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": response["message"],
                    "reasoning": response.get("reasoning_details", {})
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": f"Désolé, une erreur s'est produite : {str(e)}"
                })
                st.rerun()
    
    if st.session_state.diagnosis_state["hypotheses"]:
        st.divider()
        st.header("📊 État du Diagnostic")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Symptômes identifiés")
            for symptom in st.session_state.diagnosis_state["symptoms"]:
                st.write(f"• {symptom}")
        
        with col2:
            st.subheader("Hypothèses explorées")
            for i, hypothesis in enumerate(st.session_state.diagnosis_state["hypotheses"], 1):
                with st.expander(f"Hypothèse {i}: {hypothesis.get('pathology', 'Inconnue')}"):
                    st.write(f"**Probabilité :** {hypothesis.get('probability', 'N/A')}")
                    st.write(f"**Justification :** {hypothesis.get('justification', 'N/A')}")

if __name__ == "__main__":
    main()
