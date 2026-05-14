import streamlit as st
import pandas as pd
import sys
import os

# Add local path for engine & utils
sys.path.append(os.getcwd())
from benchmark_utils import save_logs

st.set_page_config(page_title="SpecDiff Benchmark", layout="wide")

st.title("🚀 Speculative Diffusion Decoding (SpecDiff) Benchmark")
st.markdown("**Framework :** Modèle Cible (GPT-2 XL) | Modèle Draft (MDLM-OWT - No Flash-Attn)")

mode = st.radio("Mode d'utilisation", ["📊 Visualiser les Logs (Local)", "⚙️ Exécuter un nouveau Benchmark (Colab/GPU)"], horizontal=True)

log_path = "/content/antigravity/specdiff_benchmark_logs.csv"

if mode == "📊 Visualiser les Logs (Local)":
    st.info("Ce mode lit les logs générés par Colab. Assure-toi d'avoir placé le fichier CSV au bon endroit, ou upload le ici.")
    
    uploaded_file = st.file_uploader("Uploader specdiff_benchmark_logs.csv", type=["csv"])
    
    if uploaded_file is not None:
        df_logs = pd.read_csv(uploaded_file)
    elif os.path.exists("logs/specdiff_benchmark_logs.csv"):
        df_logs = pd.read_csv("logs/specdiff_benchmark_logs.csv")
    elif os.path.exists(log_path):
        df_logs = pd.read_csv(log_path)
    elif os.path.exists("specdiff_benchmark_logs.csv"):
        df_logs = pd.read_csv("specdiff_benchmark_logs.csv")
    else:
        st.warning(f"Aucun log trouvé. Exécute d'abord le script sur Colab, place le CSV dans le dossier 'logs' ou upload le ici.")
        df_logs = None
        
    if df_logs is not None:
        st.subheader("Historique des Benchmarks")
        st.dataframe(df_logs)
        
        # Simple comparison if we have both methods
        st.subheader("Comparaison de la dernière exécution")
        ar_logs = df_logs[df_logs["method"] == "Standard AR"]
        spec_logs = df_logs[df_logs["method"] == "SpecDiff"]
        
        if not ar_logs.empty and not spec_logs.empty:
            last_ar = ar_logs.iloc[-1]
            last_spec = spec_logs.iloc[-1]
            
            speedup = last_spec["throughput_tok_sec"] / last_ar["throughput_tok_sec"] if last_ar["throughput_tok_sec"] > 0 else 0
            
            # Gérer la rétrocompatibilité si le CSV n'a pas encore la nouvelle colonne
            dec_ar = last_ar.get("decode_throughput_tok_sec", last_ar["throughput_tok_sec"])
            dec_spec = last_spec.get("decode_throughput_tok_sec", last_spec["throughput_tok_sec"])
            decode_speedup = dec_spec / dec_ar if dec_ar > 0 else 0
            
            metrics_data = {
                "Métrique": ["Throughput (tokens/sec)", "Decode Throughput (Vitesse de Croisière)", "TTFT (ms) [Warmed-up]", "ITL (ms)", "Taux d'acceptation (α %)", "Speedup Brut", "Vrai Speedup (Decode)"],
                "Standard AR": [f"{last_ar['throughput_tok_sec']:.2f}", f"{dec_ar:.2f}", f"{last_ar['ttft_ms']:.2f}", f"{last_ar['avg_itl_ms']:.2f}", "N/A", "1.0x", "1.0x"],
                "SpecDiff": [f"{last_spec['throughput_tok_sec']:.2f}", f"{dec_spec:.2f}", f"{last_spec['ttft_ms']:.2f}", f"{last_spec['avg_itl_ms']:.2f}", f"{last_spec['acceptance_rate_percent']:.2f} %", f"{speedup:.2f}x", f"{decode_speedup:.2f}x"]
            }
            st.table(pd.DataFrame(metrics_data))

else:
    # Mode Exécution
    from engine import InferenceEngine
    
    st.sidebar.header("Paramètres de Génération")
    prompt = st.sidebar.text_area("Prompt d'entrée", "The future of artificial intelligence is")
    max_new_tokens = st.sidebar.slider("Tokens à générer", min_value=8, max_value=128, value=32, step=8)

    st.sidebar.header("Hyperparamètres SpecDiff")
    gamma = st.sidebar.slider("Longueur du Draft (γ)", min_value=1, max_value=16, value=4, step=1)
    T = st.sidebar.slider("Étapes de Diffusion (T)", min_value=1, max_value=50, value=3, step=1)

    @st.cache_resource
    def load_engine():
        return InferenceEngine(target_model="togethercomputer/RedPajama-INCITE-Base-3B-v1", draft_model="kuleshov-group/mdlm-no_flashattn-fp32-owt")

    engine = load_engine()

    if st.button("Lancer le Benchmark", type="primary"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Standard Autoregressive (AR)")
            with st.spinner("Génération AR en cours..."):
                text_ar, metrics_ar = engine.standard_autoregressive(prompt, max_new_tokens=max_new_tokens)
                st.success("Terminé!")
                st.text_area("Texte Généré (AR)", text_ar, height=200)
            save_logs("Standard AR", max_new_tokens, 0, 0, metrics_ar)
                
        with col2:
            st.subheader("⚡ Speculative Diffusion Decoding")
            with st.spinner("Génération SpecDiff en cours..."):
                text_spec, metrics_spec = engine.speculative_diffusion_decoding(prompt, max_new_tokens=max_new_tokens, gamma=gamma, T=T)
                st.success("Terminé!")
                st.text_area("Texte Généré (SpecDiff)", text_spec, height=200)
            save_logs("SpecDiff", max_new_tokens, gamma, T, metrics_spec)

        st.success("Logs mis à jour dans le CSV.")
