import pandas as pd
import os
import datetime

def save_logs(method_name, max_new_tokens, gamma, T, metrics, log_dir="/content/antigravity/"):
    # Ensure directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "specdiff_benchmark_logs.csv")
    
    data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": method_name,
        "max_new_tokens": max_new_tokens,
        "gamma": gamma,
        "T_steps": T,
        "throughput_tok_sec": metrics.get_throughput(),
        "decode_throughput_tok_sec": metrics.get_decode_throughput(),
        "ttft_ms": metrics.ttft,
        "avg_itl_ms": metrics.get_avg_itl(),
        "acceptance_rate_percent": metrics.get_acceptance_rate() * 100,
        "total_time_sec": metrics.total_time
    }
    
    df = pd.DataFrame([data])
    
    if os.path.exists(log_file):
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, mode='w', header=True, index=False)
    
    return log_file
