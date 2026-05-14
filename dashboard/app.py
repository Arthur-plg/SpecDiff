import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

# Set page config for a premium look
st.set_page_config(
    page_title="SpecDiff Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a sleek dark/modern theme
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4251;
    }
    </style>
    """, unsafe_allow_html=True)

def load_all_results(results_dir="results"):
    """Find all CSV files in the results directory and merge them."""
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not files:
        return None
    
    all_df = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Ensure required columns exist
            required = ["method", "throughput_tok_sec", "decode_speedup"]
            if all(col in df.columns for col in required):
                all_df.append(df)
        except Exception as e:
            st.error(f"Error loading {f}: {e}")
            
    return pd.concat(all_df, ignore_index=True) if all_df else None

def main():
    st.title("🚀 SpecDiff Performance Analytics")
    st.markdown("---")

    # Sidebar for filters
    st.sidebar.header("Data Source")
    df = load_all_results()

    if df is None:
        st.warning("No benchmark logs found in the `results/` folder.")
        st.info("Run an experiment first using `run.py` or `grid_search.py` and place the CSV in the results directory.")
        return

    # Cleanup and format data
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", ascending=False)

    # Sidebar Filters
    target_models = st.sidebar.multiselect("Target Models", options=df["target_model"].unique(), default=df["target_model"].unique())
    draft_models = st.sidebar.multiselect("Draft Models", options=df["draft_model"].unique(), default=df["draft_model"].unique())

    filtered_df = df[
        (df["target_model"].isin(target_models)) & 
        (df["draft_model"].isin(draft_models))
    ]

    if filtered_df.empty:
        st.error("No data matches the selected filters.")
        return

    # --- TOP METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    
    # Get best speedup from filtered data
    spec_only = filtered_df[filtered_df["method"] == "SpecDiff"]
    ar_only = filtered_df[filtered_df["method"] == "Standard AR"]

    best_run = spec_only.loc[spec_only["decode_speedup"].idxmax()] if not spec_only.empty else None
    avg_alpha = spec_only["acceptance_rate_percent"].mean() if not spec_only.empty else 0

    with col1:
        st.metric("Total Runs", len(filtered_df))
    with col2:
        if best_run is not None:
            st.metric("Max Decode Speedup", f"{best_run['decode_speedup']:.2f}x", delta_color="normal")
    with col3:
        st.metric("Avg. Acceptance (α)", f"{avg_alpha:.1f}%")
    with col4:
        st.metric("Models Tested", len(target_models))

    st.markdown("### Performance Overview")
    
    # --- PLOT 1: Throughput Comparison ---
    tab1, tab2 = st.tabs(["🚀 Speedup Analysis", "📊 Raw Throughput"])
    
    with tab1:
        if not spec_only.empty:
            fig_speedup = px.scatter(
                spec_only, 
                x="gamma", 
                y="decode_speedup", 
                color="T_steps",
                size="acceptance_rate_percent",
                hover_data=["target_model", "timestamp"],
                title="Decode Speedup vs. Gamma (Bubble size = Acceptance %)",
                labels={"decode_speedup": "Speedup (x)", "gamma": "Draft Length (γ)"},
                template="plotly_dark",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_speedup, use_container_width=True)
        else:
            st.info("Run a SpecDiff benchmark to see speedup analysis.")

    with tab2:
        fig_tp = px.box(
            filtered_df, 
            x="method", 
            y="decode_throughput_tok_sec", 
            color="method",
            points="all",
            title="Throughput Distribution by Method",
            labels={"decode_throughput_tok_sec": "Tokens/sec (Steady State)"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_tp, use_container_width=True)

    # --- PLOT 3: Heatmap for Grid Search ---
    st.markdown("### Grid Search Optimization")
    if not spec_only.empty:
        # Aggregate data for heatmap (average if multiple runs exist for same gamma/T)
        heatmap_data = spec_only.groupby(["gamma", "T_steps"])["decode_speedup"].mean().reset_index()
        pivot_data = heatmap_data.pivot(index="T_steps", columns="gamma", values="decode_speedup")
        
        fig_heat = px.imshow(
            pivot_data,
            labels=dict(x="Gamma (γ)", y="Diffusion Steps (T)", color="Speedup"),
            x=pivot_data.columns,
            y=pivot_data.index,
            text_auto=".2f",
            title="Hyperparameter Optimization (Mean Speedup)",
            template="plotly_dark",
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # --- DATA TABLE ---
    st.markdown("### Detailed Logs")
    st.dataframe(
        filtered_df[[
            "timestamp", "method", "target_model", "gamma", "T_steps", 
            "decode_throughput_tok_sec", "decode_speedup", "acceptance_rate_percent", "ttft_ms"
        ]].style.background_gradient(subset=["decode_speedup"], cmap="Greens"),
        use_container_width=True
    )

if __name__ == "__main__":
    main()
