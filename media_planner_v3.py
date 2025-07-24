#!/usr/bin/env python3
"""
HypeReach Media Planner v3.0 - Clean Implementation
==================================================

Complete rewrite using the new calculation system with proper mathematics
INCLUDES: Sainsbury weighted model option for quality reach analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from reach_calc_v3 import (
    calculate_campaign_kpis, 
    optimize_media_plan,
    load_realistic_media_data,
    load_realistic_audience_data,
    get_media_strategies,
    optimize_media_plan_with_strategy,
    analyze_strategy_performance,
    REACH_CALC_VERSION
)

# =========================================================================
# PAGE CONFIG
# =========================================================================

st.set_page_config(
    page_title="HypeReach Media Planner v3.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)



# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def format_currency(amount):
    """Format currency in Turkish Lira"""
    return f"₺{amount:,.0f}"

def format_number(num):
    """Format large numbers with K/M suffixes"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:.0f}"

def create_kpi_card(title, value, subtitle="", delta=None, delta_color="normal"):
    """Create a KPI card"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.metric(
            label=title,
            value=value,
            delta=delta,
            delta_color=delta_color
        )
        if subtitle:
            st.caption(subtitle)

# =========================================================================
# MAIN APP
# =========================================================================

def main():
    st.set_page_config(page_title=" HypeReach - Media Plan Optimizer", layout="wide")
    
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("P&G_logo.png", width=100)
    with col2:
        st.title("HypeReach - Media Plan Optimizer")
        st.caption(f"Reach Calculator: {REACH_CALC_VERSION}")
    
    # Shared parameters for format-specific plans
    add_shared_parameters_sidebar()
    
    # Navigation tabs
    tab1, tab2 = st.tabs([
        "📺 Long-Format Plan", 
        "📱 Short-Format Plan"
    ])
    
    with tab1:
        long_format_plan_tab()
    
    with tab2:
        short_format_plan_tab()

def add_shared_parameters_sidebar():
    """Add shared parameters to sidebar for all tabs"""
    st.sidebar.header("📋 Campaign Parameters")
    st.sidebar.caption("Parameters for all media planning functions")
    
    # Budget input
    budget = st.sidebar.number_input(
        "Total Budget (₺)",
        min_value=100_000,
        max_value=50_000_000,
        value=5_000_000,
        step=100_000,
        format="%d",
        key="shared_budget"
    )
    
    # Target reach input
    target_reach = st.sidebar.slider(
        "Target Reach (%)",
        min_value=10,
        max_value=80,
        value=45,
        step=5,
        key="shared_target_reach"
    )
    
    # Convergence factor
    convergence_factor = st.sidebar.slider(
        "Convergence Factor",
        min_value=0.75,
        max_value=0.95,
        value=0.85,
        step=0.05,
        key="shared_convergence",
        help="Lower values penalize spreading across many channels"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Dual Strategy Generation")
    st.sidebar.info(
        "📊 **Strategy Comparison:**\n"
        "• **Long-Format**: TV + YouTube LF (Brand building)\n"
        "• **Short-Format**: YouTube + Meta + TikTok SF (Performance)\n"
        "• Both use Sainsbury Weighted Model\n"
        "• Same budget allocated optimally per strategy"
    )
    
    # Generate both plans button
    if st.sidebar.button("🚀 Generate Both Plans", type="primary", key="generate_both"):
        # Store shared parameters
        st.session_state.shared_params = {
            'budget': budget,
            'target_reach': target_reach,
            'convergence_factor': convergence_factor
        }
        
        # Create progress indicators
        progress_container = st.container()
        with progress_container:
            st.info("🚀 **Media Plan Generation Started**")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
        try:
            # Generate long-format plan
            status_text.text("🔄 Optimizing long-format strategy (TV + YouTube LF)...")
            progress_bar.progress(20)
            
            import time
            start_time = time.time()
            
            lf_spend, lf_results = optimize_media_plan_with_strategy(
                target_reach, budget, 'long_format_focused', True, convergence_factor
            )
            
            lf_time = time.time() - start_time
            progress_bar.progress(60)
            status_text.text(f"✅ Long-format completed in {lf_time:.1f}s. Optimizing short-format...")
            
            # Generate short-format plan
            start_time = time.time()
            sf_spend, sf_results = optimize_media_plan_with_strategy(
                target_reach, budget, 'short_format_focused', True, convergence_factor
            )
            
            sf_time = time.time() - start_time
            progress_bar.progress(100)
            status_text.text(f"✅ Both plans completed! (LF: {lf_time:.1f}s, SF: {sf_time:.1f}s)")
            
            if lf_spend and lf_results and sf_spend and sf_results:
                # Store results in session state
                st.session_state.lf_results = {
                    'spend': lf_spend,
                    'results': lf_results
                }
                
                st.session_state.sf_results = {
                    'spend': sf_spend,
                    'results': sf_results
                }
                
                st.success(f"✅ Both media plans generated successfully! (Total time: {lf_time + sf_time:.1f}s)")
                st.info("📺 Check the 'Long-Format Plan' tab for brand-focused strategy")
                st.info("📱 Check the 'Short-Format Plan' tab for performance-focused strategy")
                
            else:
                st.error("❌ Plan generation failed. Please try different parameters.")
                
        except Exception as e:
            st.error(f"❌ Error during plan generation: {str(e)}")
            st.info("💡 Try reducing the target reach or increasing the budget")
    
    # Display current parameters
    if hasattr(st.session_state, 'shared_params'):
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Current Parameters")
        st.sidebar.write(f"**Budget**: {format_currency(st.session_state.shared_params['budget'])}")
        st.sidebar.write(f"**Target Reach**: {st.session_state.shared_params['target_reach']}%")
        st.sidebar.write(f"**Convergence**: {st.session_state.shared_params['convergence_factor']}")
        
        # Show plan status
        lf_status = "✅ Generated" if 'lf_results' in st.session_state else "❌ Not generated"
        sf_status = "✅ Generated" if 'sf_results' in st.session_state else "❌ Not generated"
        
        st.sidebar.write(f"**Long-Format Plan**: {lf_status}")
        st.sidebar.write(f"**Short-Format Plan**: {sf_status}")


def long_format_plan_tab():
    """Long-format focused media plan with comprehensive analysis"""
    st.header("Long-Format Focused Media Plan")
    
    # Check if we have shared parameters
    if not hasattr(st.session_state, 'shared_params'):
        st.info("👈 Please set parameters in the sidebar and click 'Generate Both Plans' to see results.")
        return
    
    # Display results if available
    if 'lf_results' in st.session_state:
        display_comprehensive_plan(
            st.session_state.lf_results['spend'],
            st.session_state.lf_results['results'],
            "Long-Format",
            "📺",
            st.session_state.shared_params['budget'],
            st.session_state.shared_params['target_reach'],
            key_suffix="_lf"
        )
    else:
        st.info("👈 Please generate plans using the sidebar to see long-format results.")

def short_format_plan_tab():
    """Short-format focused media plan with comprehensive analysis"""
    st.header("Short-Format Focused Media Plan")
    
    # Check if we have shared parameters
    if not hasattr(st.session_state, 'shared_params'):
        st.info("👈 Please set parameters in the sidebar and click 'Generate Both Plans' to see results.")
        return
    
    # Display results if available
    if 'sf_results' in st.session_state:
        display_comprehensive_plan(
            st.session_state.sf_results['spend'],
            st.session_state.sf_results['results'],
            "Short-Format",
            "📱",
            st.session_state.shared_params['budget'],
            st.session_state.shared_params['target_reach'],
            key_suffix="_sf"
        )
    else:
        st.info("👈 Please generate plans using the sidebar to see short-format results.")

def display_comprehensive_plan(spend_dict, results, plan_type, icon, budget, target_reach, key_suffix=""):
    """Display comprehensive media plan analysis"""
    
    # Get TAM for absolute calculations
    from reach_calc_v3 import load_realistic_audience_data
    _, TAM = load_realistic_audience_data()
    
    # Section 1: Campaign Performance (Sainsbury Weighted Model)
    st.subheader(f"Campaign Performance - {plan_type} Focus (Sainsbury Weighted Model)")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        target_people = (target_reach / 100) * TAM
        st.metric(
            "Target Reach", 
            f"{target_reach}%", 
            f"{target_people / 1_000_000:.2f}M of {TAM / 1_000_000:.1f}M TAM"
        )
    
    with col2:
        achieved_reach = results.get('net_reach_pct', 0)
        achieved_people = (achieved_reach / 100) * TAM
        delta = f"{achieved_reach - target_reach:+.1f}%"
        delta_color = "normal" if abs(achieved_reach - target_reach) < 2 else "off"
        st.metric(
            "Achieved Reach", 
            f"{achieved_reach:.1f}%", 
            delta=delta, 
            delta_color=delta_color
        )
        st.caption(f"{achieved_people / 1_000_000:.2f}M people")
    
    with col3:
        quality_reach = results.get('quality_reach_pct', results.get('net_reach_pct', 0))
        quality_people = (quality_reach / 100) * TAM
        quality_delta = f"{quality_reach - target_reach:+.1f}%"
        quality_delta_color = "normal" if abs(quality_reach - target_reach) < 2 else "off"
        st.metric(
            "HypeQ Score", 
            f"{quality_reach:.1f}%", 
            delta=quality_delta, 
            delta_color=quality_delta_color
        )
        st.caption(f"{quality_people / 1_000_000:.2f}M people")
    
    with col4:
        st.metric("Total Budget", format_currency(budget))
    
    with col5:
        total_spend = sum(spend_dict.values())
        efficiency = (total_spend / budget) * 100
        st.metric("Budget Efficiency", f"{efficiency:.1f}%")
    
    with col6:
        avg_freq = results.get('avg_frequency', 0)
        st.metric("Avg Frequency", f"{avg_freq:.1f}x")
    
    # Section 2: Media Mix Analysis
    st.subheader("📊 Media Mix Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Budget allocation pie chart
        spend_df = pd.DataFrame({
            'Channel': list(spend_dict.keys()),
            'Spend': list(spend_dict.values()),
            'Percentage': [v/sum(spend_dict.values())*100 for v in spend_dict.values()]
        })
        
        # Filter out zero spend channels
        spend_df = spend_df[spend_df['Spend'] > 0]
        
        fig_pie = px.pie(
            spend_df,
            values='Spend',
            names='Channel',
            title="Budget Allocation",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True, key=f"budget_pie{key_suffix}")
    
    with col2:
        # Channel type breakdown
        lf_spend = sum(spend for channel, spend in spend_dict.items() if channel.endswith('_LF'))
        sf_spend = sum(spend for channel, spend in spend_dict.items() if channel.endswith('_SF'))
        
        type_df = pd.DataFrame({
            'Format': ['Long-Form', 'Short-Form'],
            'Spend': [lf_spend, sf_spend],
            'Percentage': [lf_spend/(lf_spend+sf_spend)*100, sf_spend/(lf_spend+sf_spend)*100]
        })
        
        fig_format = px.bar(
            type_df,
            x='Format',
            y='Spend',
            title="Format Distribution",
            color='Format',
            color_discrete_map={'Long-Form': '#2E86AB', 'Short-Form': '#A23B72'},
            text='Percentage'
        )
        fig_format.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_format, use_container_width=True, key=f"format_bar{key_suffix}")
    
    # Section 3: Platform Performance
    st.subheader("🎯 Platform Performance")
    
    # Create platform performance table
    performance_data = []
    channels = load_realistic_media_data()
    
    for channel, spend in spend_dict.items():
        if spend > 0:
            # Calculate individual channel metrics
            single_channel_spend = {ch: 0 for ch in spend_dict.keys()}
            single_channel_spend[channel] = spend
            
            try:
                channel_results = calculate_campaign_kpis(single_channel_spend, True, 0.85)
                
                # Calculate CPM
                impressions = channel_results.get('total_impressions', 0)
                cpm = (spend / impressions * 1000) if impressions > 0 else 0
                
                performance_data.append({
                    'Channel': channel,
                    'Spend': format_currency(spend),
                    'Reach %': f"{channel_results.get('quality_reach_pct', 0):.1f}%",
                    'Impressions': format_number(impressions),
                    'CPM': f"₺{cpm:.2f}",
                    'Frequency': f"{channel_results.get('avg_frequency', 0):.1f}x"
                })
            except:
                performance_data.append({
                    'Channel': channel,
                    'Spend': format_currency(spend),
                    'Reach %': "N/A",
                    'Impressions': "N/A",
                    'CPM': "N/A",
                    'Frequency': "N/A"
                })
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True)
    
    # Section 4: Detailed Budget Breakdown
    st.subheader("💰 Detailed Budget Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Channel Investment Analysis**")
        
        breakdown_data = []
        total_spend = sum(spend_dict.values())
        
        for channel, spend in spend_dict.items():
            if spend > 0:
                channel_info = channels.get(channel)
                if channel_info:
                    breakdown_data.append({
                        'Channel': channel,
                        'Budget': format_currency(spend),
                        'Percentage': f"{spend/total_spend*100:.1f}%",
                        'CPM': f"₺{channel_info.cpm}",
                        'Max Reach': f"{channel_info.max_reach*100:.1f}%"
                    })
        
        breakdown_df = pd.DataFrame(breakdown_data)
        st.dataframe(breakdown_df, use_container_width=True)
    
    with col2:
        st.write("**Budget Utilization**")
        
        total_allocated = sum(spend_dict.values())
        remaining = budget - total_allocated
        utilization = (total_allocated / budget) * 100
        
        st.metric("Total Allocated", format_currency(total_allocated))
        st.metric("Remaining Budget", format_currency(remaining))
        st.metric("Utilization Rate", f"{utilization:.1f}%")
        
        # Budget efficiency by format
        if lf_spend > 0 and sf_spend > 0:
            st.write("**Format Efficiency**")
            st.metric("Long-Form Investment", format_currency(lf_spend))
            st.metric("Short-Form Investment", format_currency(sf_spend))
            st.metric("LF:SF Ratio", f"{lf_spend/sf_spend:.2f}:1")

# =========================================================================
# APP ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    main() 