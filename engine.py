import time
import torch
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM

class EngineMetrics:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.total_tokens = 0
        self.ttft = 0.0
        self.start_time = 0.0
        self.draft_accepted = 0
        self.draft_total = 0
        self.total_time = 0.0

    def get_throughput(self):
        if self.total_time == 0: return 0
        return self.total_tokens / self.total_time

    def get_avg_itl(self):
        # Vrai calcul global de l'ITL: (Temps total sans le TTFT) / (Nombre de tokens générés après le premier)
        if self.total_tokens <= 1: return 0
        return (self.total_time * 1000 - self.ttft) / (self.total_tokens - 1)
        
    def get_decode_throughput(self):
        # "Vitesse de croisière" qui exclut le TTFT (et donc le warmup)
        if self.total_tokens <= 1: return 0
        time_after_first = self.total_time - (self.ttft / 1000.0)
        if time_after_first <= 0: return 0
        return (self.total_tokens - 1) / time_after_first

    def get_acceptance_rate(self):
        if self.draft_total == 0: return 0
        return self.draft_accepted / self.draft_total

class MDLMGenerator:
    """Implémentation mathématique autonome du modèle de diffusion MDLM."""
    def __init__(self, model_name, device):
        self.device = device
        # =========================================================================
        # LE "HACK" ULTIME POUR CONTOURNER FLASH_ATTN DANS HUGGING FACE
        # Même le modèle "no_flashattn" possède un "import flash_attn" écrit
        # en haut de son fichier texte sur le serveur d'Hugging Face.
        # Transformers scanne le texte (AST) et bloque tout. On va désactiver ça !
        # =========================================================================
        import sys
        from unittest.mock import MagicMock
        import transformers.dynamic_module_utils

        # 1. On désactive le scanner de texte d'Hugging Face qui cherche les imports
        transformers.dynamic_module_utils.check_imports = lambda filename: []

        # 2. On leurre l'interpréteur Python quand il exécutera la ligne "import flash_attn"
        if 'flash_attn' not in sys.modules:
            sys.modules['flash_attn'] = MagicMock()
            sys.modules['flash_attn.layers'] = MagicMock()
            sys.modules['flash_attn.layers.rotary'] = MagicMock()
        # =========================================================================

        self.model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.vocab_size = self.model.config.vocab_size # Généralement 50257 ou 50258
        self.model.eval()

        # =========================================================================
        # LE "HACK" ROTARY EMBEDDINGS (Pire oubli des chercheurs !)
        # Même s'ils ont rajouté une option "use_flash_attn=False", ils ont oublié
        # d'écrire un fallback (plan B) pour la fonction RoPE (Rotary Position Embeddings)
        # qui continue d'appeler le kernel C++ de flash_attn.
        # Comme on l'a "Mocké", le kernel renvoie un objet vide (MagicMock), 
        # et le tenseur perd toutes ses dimensions (d'où l'erreur torch.arange !).
        # On remplace donc dynamiquement leur fonction par du pur PyTorch !
        # =========================================================================
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def patched_apply_rotary_pos_emb(qkv, cos, sin):
            # Leurs tensors cos et sin originaux
            cos_half = cos[0, :, 0, 0, :cos.shape[-1]//2]
            sin_half = sin[0, :, 0, 0, :sin.shape[-1]//2]
            # On recrée les dimensions complètes pour PyTorch
            cos_full = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(0).unsqueeze(2)
            sin_full = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(0).unsqueeze(2)
            
            q = qkv[:, :, 0]
            k = qkv[:, :, 1]
            
            q_embed = (q * cos_full) + (rotate_half(q) * sin_full)
            k_embed = (k * cos_full) + (rotate_half(k) * sin_full)
            
            qkv_out = qkv.clone()
            qkv_out[:, :, 0] = q_embed
            qkv_out[:, :, 1] = k_embed
            return qkv_out

        # On injecte notre fonction PyTorch dans leur code !
        module = sys.modules[self.model.__module__]
        module.apply_rotary_pos_emb = patched_apply_rotary_pos_emb
        # =========================================================================

        self.mask_index = 50257 # Index par défaut du MASK pour un tokenizer GPT-2
        self.eps = 1e-3
        
    def noise_total(self, t):
        """LogLinear Noise Schedule"""
        return -torch.log1p(-(1 - self.eps) * t)
        
    def _sample_categorical(self, categorical_probs):
        """Echantillonnage de Gumbel pour tokens catégoriels"""
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)

    def _subs_parameterization(self, logits, xt):
        """Paramétrisation de substitution pour diffusion discrète absorbante"""
        logits[:, :, self.mask_index] += -1000000.0
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = -1000000.0
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _ddpm_caching_update(self, x, t_tensor, dt, p_x0=None):
        """Etape de débruitage avec caching pour SpecDiff"""
        sigma_t = self.noise_total(t_tensor)
        
        t_squeeze = t_tensor.squeeze(-1) if t_tensor.ndim > 1 else t_tensor
        move_chance_t = t_squeeze[:, None, None]
        move_chance_s = (t_squeeze - dt)[:, None, None]
        
        if p_x0 is None:
            # Squeeze sigma_t to be 1D, sinon le réseau calcule de mauvaises dimensions
            sigma_t_sq = sigma_t.squeeze(-1) if sigma_t.ndim > 1 else sigma_t
            out = self.model(x, sigma_t_sq)
            logits = out.logits if hasattr(out, 'logits') else out
            if isinstance(logits, tuple): logits = logits[0]
            log_p_x0 = self._subs_parameterization(logits, x)
            p_x0 = log_p_x0.exp()
            
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = self._sample_categorical(q_xs)
        
        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def generate_draft(self, prompt_ids, gamma, T):
        """Génère un block de gamma tokens avec T étapes de diffusion"""
        bsz = prompt_ids.size(0)
        prompt_len = prompt_ids.size(1)
        
        # Initialisation : [PROMPT] + [MASK]*gamma
        x = torch.cat([
            prompt_ids, 
            torch.full((bsz, gamma), self.mask_index, dtype=torch.long, device=self.device)
        ], dim=1)
        
        timesteps = torch.linspace(1, 1e-5, T + 1, device=self.device)
        dt = (1 - 1e-5) / T
        
        p_x0_cache = None
        
        with torch.no_grad():
            for i in range(T):
                t_tensor = timesteps[i] * torch.ones(bsz, 1, device=self.device)
                p_x0_cache, x_next = self._ddpm_caching_update(x, t_tensor, dt, p_x0=p_x0_cache)
                # Infilling : on force le prompt à rester intact
                x_next[:, :prompt_len] = prompt_ids
                x = x_next
                
        return x[:, prompt_len:]

class InferenceEngine:
    def __init__(self, target_model="gpt2-xl", draft_model="kuleshov-group/mdlm-no_flashattn-fp32-owt", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading Target Model: {target_model} on {device}")
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model)
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model, torch_dtype=torch.float16).to(self.device)
        self.target_model.eval()
        
        print(f"Loading Draft Model MDLM (Standalone Extraction): {draft_model} on {device}")
        self.mdlm_generator = MDLMGenerator(draft_model, device)
        print("MDLM loaded successfully.")
        
        print("Warming up models (CUDA)...")
        dummy_input = self.target_tokenizer.encode("Warmup", return_tensors="pt").to(self.device)
        _ = self.target_model(dummy_input)
        _ = self.mdlm_generator.generate_draft(dummy_input, gamma=4, T=2)
        print("Warmup complete. TTFT will now accurately reflect real generation start time.")

    def standard_autoregressive(self, prompt, max_new_tokens=32):
        metrics = EngineMetrics()
        input_ids = self.target_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        start_time = time.time()
        metrics.start_time = start_time
        
        generated_ids = input_ids
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                outputs = self.target_model(generated_ids)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Troncature pour compatibilité vocabulaire (ex: RedPajama 3B)
                next_token_logits = next_token_logits[:, :self.mdlm_generator.vocab_size]
                
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                if i == 0:
                    metrics.ttft = (time.time() - start_time) * 1000 # ms
                
                metrics.total_tokens += 1

        metrics.total_time = time.time() - start_time
        return self.target_tokenizer.decode(generated_ids[0]), metrics

    def speculative_diffusion_decoding(self, prompt, max_new_tokens=32, gamma=4, T=10):
        metrics = EngineMetrics()
        input_ids = self.target_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        start_time = time.time()
        metrics.start_time = start_time
        
        generated_ids = input_ids
        tokens_generated = 0
        first_token_emitted = False
        
        with torch.no_grad():
            while tokens_generated < max_new_tokens:
                step_start = time.time()
                
                # 1. Draft Generation via Standalone MDLM
                draft_tokens = self.mdlm_generator.generate_draft(generated_ids, gamma, T)
                
                # 2. Parallel Verification via Target Model
                verify_ids = torch.cat([generated_ids, draft_tokens], dim=1)
                outputs = self.target_model(verify_ids)
                
                seq_len = generated_ids.size(1)
                logits = outputs.logits[:, seq_len-1 : seq_len-1+gamma, :]
                
                # Troncature pour compatibilité vocabulaire (ex: RedPajama 3B)
                logits = logits[:, :, :self.mdlm_generator.vocab_size]
                
                # Verification (Algorithm 1) - Greedy
                accepted_count = 0
                target_tokens = torch.argmax(logits, dim=-1)
                
                for i in range(gamma):
                    metrics.draft_total += 1
                    target_token = target_tokens[:, i].unsqueeze(-1)
                    draft_token = draft_tokens[:, i].unsqueeze(-1)
                    
                    if target_token.item() == draft_token.item():
                        generated_ids = torch.cat([generated_ids, target_token], dim=1)
                        accepted_count += 1
                        metrics.draft_accepted += 1
                        tokens_generated += 1
                        
                        if not first_token_emitted:
                            metrics.ttft = (time.time() - start_time) * 1000
                            first_token_emitted = True
                            
                        if tokens_generated >= max_new_tokens:
                            break
                    else:
                        generated_ids = torch.cat([generated_ids, target_token], dim=1)
                        tokens_generated += 1
                        
                        if not first_token_emitted:
                            metrics.ttft = (time.time() - start_time) * 1000
                            first_token_emitted = True
                        break
                
        metrics.total_time = time.time() - start_time
        metrics.total_tokens = tokens_generated
        return self.target_tokenizer.decode(generated_ids[0]), metrics
