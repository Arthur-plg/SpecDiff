import os
import sys

# Ajouter mdlm au sys.path s'il existe (géré dans engine.py, mais sécurité supplémentaire)
sys.path.append(os.getcwd())

from engine import InferenceEngine
from benchmark_utils import save_logs

def main():
    print("🚀 Démarrage du Benchmark SpecDiff (Headless Mode)")
    
    prompt = "The future of artificial intelligence is"
    max_new_tokens = 256
    
    # 1. Charger les modèles
    print("\n--- Chargement des modèles ---")
    # TEST DU MODÈLE REDPAJAMA 3B
    engine = InferenceEngine(target_model="togethercomputer/RedPajama-INCITE-Base-3B-v1", draft_model="kuleshov-group/mdlm-no_flashattn-fp32-owt")
    
    # 2. Standard AR
    print(f"\n--- Exécution: Standard AR ({max_new_tokens} tokens) ---")
    text_ar, metrics_ar = engine.standard_autoregressive(prompt, max_new_tokens=max_new_tokens)
    print(f"✅ Terminé. Throughput: {metrics_ar.get_throughput():.2f} tok/s | Decode Throughput: {metrics_ar.get_decode_throughput():.2f} tok/s | TTFT: {metrics_ar.ttft:.2f} ms")
    
    # Sauvegarde des logs pour AR
    save_logs("Standard AR 3B", max_new_tokens, 0, 0, metrics_ar)
    
    # 3. Grid Search sur le modèle 2.7B
    hyperparams_to_test = []
    for t_val in [3, 4, 5]:
        for g_val in [3, 4, 5]:
            hyperparams_to_test.append((g_val, t_val))
            
    for gamma, T in hyperparams_to_test:
        print(f"\n--- Exécution: SpecDiff (γ={gamma}, T={T}) ---")
        text_spec, metrics_spec = engine.speculative_diffusion_decoding(prompt, max_new_tokens=max_new_tokens, gamma=gamma, T=T)
        
        # Compute real Decode Speedup
        decode_speedup = metrics_spec.get_decode_throughput() / metrics_ar.get_decode_throughput() if metrics_ar.get_decode_throughput() > 0 else 0
        
        print(f"✅ Terminé. Throughput: {metrics_spec.get_throughput():.2f} tok/s | Decode Throughput: {metrics_spec.get_decode_throughput():.2f} tok/s | Acceptation: {metrics_spec.get_acceptance_rate()*100:.2f}% | Decode Speedup: {decode_speedup:.2f}x")
        
        # Sauvegarde des logs pour SpecDiff
        log_path = save_logs("SpecDiff 3B", max_new_tokens, gamma, T, metrics_spec)
    
    print(f"\n🎉 Grid Search 3B terminé ! Les logs sont disponibles ici : {log_path}")

if __name__ == "__main__":
    main()
