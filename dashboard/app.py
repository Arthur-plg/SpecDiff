import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Speculative Decoding Analysis",
    page_icon="🚀",
    layout="wide",
)

# --- CUSTOM STYLING (HuggingFace / W&B Inspired) ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Metric Card Styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700;
        color: #1f2937;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Header styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 2rem;
    }
    
    /* Section containers */
    .section-card {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
def load_data(results_dir="results"):
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not files:
        return None
    
    all_df = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Basic validation
            if "method" in df.columns:
                all_df.append(df)
        except Exception:
            continue
            
    if not all_df:
        return None
        
    df = pd.concat(all_df, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp", ascending=False)

# --- SIDEBAR (Filters) ---
def render_sidebar(df):
    with st.sidebar:
        st.title("🔍 Filters")
        st.markdown("---")
        
        # Experiment Selection
        all_targets = sorted(df["target_model"].unique())
        selected_targets = st.multiselect("Target Models", all_targets, default=all_targets)
        
        all_drafts = sorted(df["draft_model"].unique())
        selected_drafts = st.multiselect("Draft Models", all_drafts, default=all_drafts)
        
        st.markdown("---")
        st.markdown("### Display Options")
        chart_theme = st.selectbox("Chart Theme", ["plotly_white", "plotly_dark", "ggplot2"])
        
        st.markdown("---")
        st.caption("v1.0.0 | SpecDiff Framework")
        
        return selected_targets, selected_drafts, chart_theme

# --- MAIN APP ---
def main():
    # 1. Header
    st.markdown('<h1 class="main-title">Speculative Decoding Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Performance comparison: draft model vs target diffusion model</p>', unsafe_allow_html=True)

    df = load_data()
    if df is None:
        st.info("👋 Welcome! Please add your experiment CSV files to the `results/` directory to begin analysis.")
        return

    # 2. Filtering
    targets, drafts, theme = render_sidebar(df)
    filtered_df = df[df["target_model"].isin(targets) & df["draft_model"].isin(drafts)]

    if filtered_df.empty:
        st.warning("No data found for the selected filters.")
        return

    # 3. KPI SUMMARY CARDS
    # Extract key stats
    ar_baseline = filtered_df[filtered_df["method"].str.contains("AR", case=False)]
    spec_runs = filtered_df[filtered_df["method"] == "SpecDiff"]
    
    best_tp = 0
    max_speedup = 0
    avg_alpha = 0
    avg_latency = 0

    if not spec_runs.empty:
        best_spec = spec_runs.loc[spec_runs["decode_throughput_tok_sec"].idxmax()]
        best_tp = best_spec["decode_throughput_tok_sec"]
        max_speedup = spec_runs["decode_speedup"].max()
        avg_alpha = spec_runs["acceptance_rate_percent"].mean()
        avg_latency = spec_runs["avg_itl_ms"].mean()

    # Layout KPI cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Peak Throughput", f"{best_tp:.1f}", "tok/s")
    with m2:
        st.metric("Max Speed-up", f"{max_speedup:.2f}x", "vs baseline")
    with m3:
        st.metric("Avg. Accept Rate", f"{avg_alpha:.1f}%", "of draft")
    with m4:
        st.metric("Avg. ITL Latency", f"{avg_latency:.1f}", "ms/tok")

    st.markdown("---")

    # 4. CHARTS GRID
    col_left, col_right = st.columns(2)

    with col_left:
        # Chart 1: Throughput Comparison (Bar)
        st.markdown("#### 🚀 Throughput by Configuration")
        fig_tp = px.bar(
            filtered_df, 
            x="gamma", 
            y="decode_throughput_tok_sec", 
            color="method",
            barmode="group",
            facet_col="T_steps",
            template=theme,
            color_discrete_sequence=["#636EFA", "#00CC96"]
        )
        fig_tp.update_layout(margin=dict(t=40, b=40, l=20, r=20))
        st.plotly_chart(fig_tp, use_container_width=True)

    with col_right:
        # Chart 2: Speedup vs Acceptance Rate (Scatter)
        st.markdown("#### 🎯 Efficiency Analysis")
        if not spec_runs.empty:
            fig_scatter = px.scatter(
                spec_runs,
                x="acceptance_rate_percent",
                y="decode_speedup",
                size="gamma",
                color="T_steps",
                hover_data=["target_model"],
                template=theme,
                labels={"acceptance_rate_percent": "Accept Rate (%)", "decode_speedup": "Speedup (x)"}
            )
            st.plotly_chart(fig_scatter, width='stretch')
        else:
            st.info("No SpecDiff data for efficiency analysis.")

    # 3. DETAILED ANALYSIS
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🎯 Optimization Grid", "📈 Latency Analysis", "📄 Full Logs"])

    with tab1:
        if not spec_runs.empty:
            st.markdown("#### Finding the optimal (γ, T) combination")
            heatmap_data = spec_runs.groupby(["gamma", "T_steps"])["decode_speedup"].mean().reset_index()
            pivot_data = heatmap_data.pivot(index="T_steps", columns="gamma", values="decode_speedup")
            
            fig_heat = px.imshow(
                pivot_data,
                labels=dict(x="Gamma (γ)", y="Diffusion Steps (T)", color="Speedup"),
                x=pivot_data.columns,
                y=pivot_data.index,
                text_auto=".2f",
                template=theme,
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_heat, width='stretch')
        else:
            st.info("No SpecDiff data found for optimization grid.")

    with tab2:
        fig_itl = px.line(
            df.sort_values("timestamp"), 
            x="timestamp", 
            y="avg_itl_ms", 
            color="method",
            markers=True,
            title="Inter-Token Latency (ITL) over time",
            template=theme
        )
        st.plotly_chart(fig_itl, width='stretch')

    with tab3:
        st.dataframe(
            df[[
                "timestamp", "method", "gamma", "T_steps", 
                "decode_throughput_tok_sec", "decode_speedup", "acceptance_rate_percent", "ttft_ms"
            ]],
            width='stretch',
            hide_index=True
        )

    # 6. FOOTER
    st.markdown("---")
    f1, f2 = st.columns(2)
    with f1:
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with f2:
        st.markdown('<p style="text-align: right; color: gray; font-size: 0.8rem;">SpecDiff Framework Analytics Tool</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
