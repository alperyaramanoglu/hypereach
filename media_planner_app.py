import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Import functions from our reach calculator
from reach_calc import (
    segments, cpm, aud_prof, w, reach_par, ov, TAM, platform_totals,
    LONG_FORMAT_CONSTRAINTS, SHORT_FORMAT_CONSTRAINTS, CHANNEL_CONSTRAINTS,
    optimize_media_plan_hybrid, test_max_possible_reach_hybrid,
    plan_kpis_hybrid, calculate_incremental_contributions,
    extract_real_campaign_data_from_screenshot, integrate_real_campaign_data,
    calculate_reach_curves_from_data, calculate_enhanced_quality_weights,
    calculate_channel_frequency
)

# Page configuration
st.set_page_config(
    page_title="P&G Media Planning Optimizer",
    page_icon="P&G_logo.png",  # Use P&G logo as favicon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .header-subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    /* Sidebar logo styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    .stSidebar .stImage {
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'real_data_loaded' not in st.session_state:
    st.session_state.real_data_loaded = False

# Load real campaign data
def load_real_campaign_data():
    """Load and process real campaign data"""
    try:
        # Load real data
        real_data = extract_real_campaign_data_from_screenshot()
        
        # Process data
        global segments, cpm, aud_prof, w, reach_par
        new_segments, new_cpm, new_profiles, new_quality_weights, platform_audiences = integrate_real_campaign_data(real_data)
        new_reach_curves = calculate_reach_curves_from_data(real_data)
        
        # Update globals
        segments.update(new_segments)
        cpm.update(new_cpm)
        aud_prof.update(new_profiles)
        w.update(new_quality_weights)
        reach_par.update(new_reach_curves)
        
        # CRITICAL: Update TAM with correct platform audience sizes from demographic segment table
        global TAM, platform_totals
        # Use the current platform totals from Excel file (or fallback values)
        TAM = max(platform_totals.values())  # Use the largest platform as TAM
        
        # Store in session state for display functions
        st.session_state.updated_cpm = new_cpm
        st.session_state.updated_segments = new_segments
        st.session_state.updated_aud_prof = new_profiles
        st.session_state.updated_w = new_quality_weights
        st.session_state.updated_reach_par = new_reach_curves
        st.session_state.updated_TAM = TAM  # Store updated TAM
        
        return True, platform_audiences
    except Exception as e:
        st.error(f"Error loading real campaign data: {e}")
        return False, None

def calculate_longform_reach(spend_dict):
    """Calculate reach specifically for long-format channels"""
    
    # Get updated data from session state or fallback to globals
    current_cpm = getattr(st.session_state, 'updated_cpm', cpm)
    
    # CRITICAL FIX: Ensure all expected channels are present in current_cpm
    # If real data is missing for any channel, use global fallback
    expected_channels = ['TV_LF', 'YT_LF', 'YT_SF', 'Meta_SF', 'TT_SF']
    for channel in expected_channels:
        if channel not in current_cpm:
            current_cpm[channel] = cpm.get(channel, 10)  # Use global or default fallback
    
    current_segments = getattr(st.session_state, 'updated_segments', segments)
    current_aud_prof = getattr(st.session_state, 'updated_aud_prof', aud_prof)
    current_reach_par = getattr(st.session_state, 'updated_reach_par', reach_par)
    
    # CRITICAL FIX: Ensure all expected channels have audience profiles
    # If real data is missing audience profile for any channel, use global fallback
    expected_channels = ['TV_LF', 'YT_LF', 'YT_SF', 'Meta_SF', 'TT_SF']
    for channel in expected_channels:
        if channel not in current_aud_prof:
            current_aud_prof[channel] = aud_prof.get(channel, {})
        elif not current_aud_prof[channel]:  # Empty dict
            current_aud_prof[channel] = aud_prof.get(channel, {})
    
    # CRITICAL FIX: Ensure all expected channels have reach parameters
    for channel in expected_channels:
        if channel not in current_reach_par:
            current_reach_par[channel] = reach_par.get(channel, (0.5, 0.2))
    
    # Identify long-format channels
    lf_channels = {channel: spend for channel, spend in spend_dict.items() 
                   if channel.endswith('_LF') and spend > 0}
    
    if not lf_channels:
        return 0.0
    
    # Calculate reach for each segment
    total_lf_reach = 0
    total_population = sum(current_segments.values())
    
    for segment, population in current_segments.items():
        segment_lf_reach = 0
        
        # Calculate platform-level reaches for long-format only
        platform_reaches = {}
        
        for channel, spend in lf_channels.items():
            if segment in current_aud_prof.get(channel, {}):
                # Calculate GRPs and reach for this channel
                channel_cpm = current_cpm.get(channel, 0)
                if channel_cpm > 0:
                    segment_grp = (spend / channel_cpm) / (population / 1000)
                    max_reach, k = current_reach_par.get(channel, (0.5, 0.2))
                    channel_reach = max_reach * (1 - np.exp(-k * segment_grp))
                    
                    # Weight by audience profile
                    weighted_reach = channel_reach * current_aud_prof[channel][segment]
                    
                    # Group by platform for overlap calculation
                    platform = channel.replace('_LF', '')
                    if platform == 'YT':
                        platform = 'YouTube'
                    elif platform == 'TT':
                        platform = 'TikTok'
                    
                    platform_reaches[platform] = weighted_reach
        
        # Calculate deduplicated reach using GWI overlap factors for this segment
        if platform_reaches:
            # Handle TV separately (no overlap with digital)
            tv_reach = platform_reaches.get('TV', 0)
            
            # Digital platforms with overlap
            yt_reach = platform_reaches.get('YouTube', 0)
            meta_reach = platform_reaches.get('Meta', 0)
            tt_reach = platform_reaches.get('TikTok', 0)
            
            # Apply GWI deduplication
            digital_overlap = 0
            if yt_reach > 0 and meta_reach > 0:
                digital_overlap += ov[('YT','Meta')] * min(yt_reach, meta_reach)
            if yt_reach > 0 and tt_reach > 0:
                digital_overlap += ov[('YT','TT')] * min(yt_reach, tt_reach)
            if meta_reach > 0 and tt_reach > 0:
                digital_overlap += ov[('Meta','TT')] * min(meta_reach, tt_reach)
            
            # Calculate net reach for this segment
            segment_lf_reach = tv_reach + yt_reach + meta_reach + tt_reach - digital_overlap
            segment_lf_reach = max(0, min(1, segment_lf_reach))  # Cap between 0 and 1
        
        # Weight by segment population
        total_lf_reach += segment_lf_reach * population
    
    # Convert to percentage of TAM
    if total_population > 0:
        return (total_lf_reach / total_population) * 100
    else:
        return 0.0

def filter_segments_by_demographics(include_male, include_female, include_18_24, include_25_34, include_35_44, include_45_54, include_55_plus):
    """Filter segments based on demographic selection"""
    
    # Get original segments
    original_segments = segments.copy()
    filtered_segments = {}
    
    # Create flexible age group mapping to handle different naming conventions
    age_selections = {
        '18_24': include_18_24,
        '25_34': include_25_34, 
        '35_44': include_35_44,
        '45_54': include_45_54,
        '55_plus': include_55_plus,
        '55+': include_55_plus,  # Handle alternative naming
        '55plus': include_55_plus
    }
    
    gender_selections = {
        'M': include_male,
        'F': include_female
    }
    
    # Filter segments based on selection
    selected_segments = []
    for segment_name, population in original_segments.items():
        # Parse segment name (e.g., 'F18_24' -> gender='F', age='18_24')
        if len(segment_name) < 2:
            continue
            
        gender = segment_name[0]  # First character (F or M)
        age_group = segment_name[1:]  # Rest of the string
        
        # Check if this segment should be included
        gender_match = gender_selections.get(gender, False)
        age_match = age_selections.get(age_group, False)
        
        if gender_match and age_match:
            filtered_segments[segment_name] = population
            selected_segments.append(segment_name)
    
    return filtered_segments

def update_platform_totals_for_demographics(filtered_segments):
    """Recalculate platform totals based on filtered segments"""
    
    # Use cached Excel data if available
    if hasattr(st.session_state, 'excel_data') and st.session_state.excel_data is not None:
        df = st.session_state.excel_data
        
        # Initialize platform totals
        new_platform_totals = {
            'TV': 0,
            'YouTube': 0,
            'Meta': 0,
            'TikTok': 0
        }
        
        # Sum platform-specific audiences for filtered segments only
        for segment_name in filtered_segments.keys():
            # Convert internal format back to Excel format (F18_24 -> F18-24, F55_plus -> F55+)
            if segment_name.endswith('_plus'):
                # Handle 55+ segments: F55_plus -> F55+
                excel_segment_name = segment_name.replace('_plus', '+')
            else:
                # Handle regular segments: F18_24 -> F18-24
                excel_segment_name = segment_name.replace('_', '-')
            
            # Find the row for this segment
            segment_row = df[df['Segment'] == excel_segment_name]
            
            if not segment_row.empty:
                # Add platform-specific audience sizes
                new_platform_totals['TV'] += int(segment_row['TV'].iloc[0])
                new_platform_totals['YouTube'] += int(segment_row['YouTube'].iloc[0])
                new_platform_totals['Meta'] += int(segment_row['Meta'].iloc[0])
                new_platform_totals['TikTok'] += int(segment_row['TikTok'].iloc[0])

        
        return new_platform_totals
    
    else:
        # Fallback: use proportional allocation based on original platform totals
        total_filtered_population = sum(filtered_segments.values())
        original_total_population = sum(segments.values())
        
        if original_total_population > 0:
            proportion = total_filtered_population / original_total_population
            
            new_platform_totals = {
                'TV': int(platform_totals['TV'] * proportion),
                'YouTube': int(platform_totals['YouTube'] * proportion),
                'Meta': int(platform_totals['Meta'] * proportion),
                'TikTok': int(platform_totals['TikTok'] * proportion)
            }
            
            return new_platform_totals
        else:
            # Final fallback
            return {
                'TV': total_filtered_population,
                'YouTube': total_filtered_population, 
                'Meta': total_filtered_population,
                'TikTok': total_filtered_population
            }

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">HypeReach - Media Planning Optimizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle"><strong>Strategic Media Planning with Real Campaign Data</strong></p>', unsafe_allow_html=True)
    
    # Initialize session state for parameter tracking
    if 'last_optimization_params' not in st.session_state:
        st.session_state.last_optimization_params = None
    
    # Load real data
    if not st.session_state.real_data_loaded:
        with st.spinner("Loading real campaign data..."):
            success, platform_audiences = load_real_campaign_data()
            if success:
                st.session_state.real_data_loaded = True
                st.session_state.platform_audiences = platform_audiences
                
                # Cache raw Excel data for demographic filtering
                try:
                    import pandas as pd
                    import os
                    
                    file_path = 'audience-segments.xlsx'
                    if os.path.exists(file_path):
                        df = pd.read_excel(file_path, sheet_name='Sheet1')
                        st.session_state.excel_data = df
                    else:
                        st.session_state.excel_data = None
                except Exception as e:
                    st.session_state.excel_data = None
                    print(f"Error caching Excel data: {e}")
                
                st.success("✅ Real campaign data loaded successfully!")
            else:
                st.error("❌ Failed to load real campaign data")
                return
    
    # Sidebar - P&G Logo and Input Controls
    # Add P&G logo to sidebar
    try:
        st.sidebar.image("P&G_logo.png", width=100)
    except FileNotFoundError:
        st.sidebar.write("📺")  # Fallback emoji if logo not found
    
    # Add separator in sidebar
    st.sidebar.markdown("---")
    
    st.sidebar.header("Planning Parameters")
    
    # Target Reach Input
    target_reach = st.sidebar.slider(
        "Target Reach (% of TAM)",
        min_value=10,
        max_value=80,
        value=44,
        step=1,
        help="Target reach as percentage of Total Addressable Market"
    )
    
    # Budget Input
    budget = st.sidebar.number_input(
        "Available Budget (₺)",
        min_value=1_000_000,
        max_value=1_000_000_000,
        value=6_000_000,
        step=500_000,
        help="Total media budget available"
    )
    
    # Strategy Selection
    st.sidebar.header("Strategy Options")
    strategy = st.sidebar.selectbox(
        "Media Strategy",
        ["Balanced", "Long-Format Focused", "Short-Format Focused"],
        help="Choose your media strategy approach"
    )
    
    # Demographic Targeting
    st.sidebar.header("Demographic Targeting")
    
    # Gender Selection
    st.sidebar.subheader("Gender")
    include_male = st.sidebar.checkbox("Male", value=True)
    include_female = st.sidebar.checkbox("Female", value=True)
    
    # Age Group Selection
    st.sidebar.subheader("Age Groups")
    include_18_24 = st.sidebar.checkbox("18-24", value=True)
    include_25_34 = st.sidebar.checkbox("25-34", value=True)
    include_35_44 = st.sidebar.checkbox("35-44", value=True)
    include_45_54 = st.sidebar.checkbox("45-54", value=True)
    include_55_plus = st.sidebar.checkbox("55+", value=True)
    
    # Validate selection
    if not any([include_male, include_female]):
        st.sidebar.error("⚠️ Please select at least one gender")
    if not any([include_18_24, include_25_34, include_35_44, include_45_54, include_55_plus]):
        st.sidebar.error("⚠️ Please select at least one age group")
    
    # Real-time demographic filtering for charts
    if any([include_male, include_female]) and any([include_18_24, include_25_34, include_35_44, include_45_54, include_55_plus]):
        # Apply demographic filtering in real-time
        filtered_segments = filter_segments_by_demographics(
            include_male, include_female, include_18_24, include_25_34, 
            include_35_44, include_45_54, include_55_plus
        )
        
        # Update platform totals and TAM
        filtered_platform_totals = update_platform_totals_for_demographics(filtered_segments)
        filtered_TAM = max(filtered_platform_totals.values())
        
        # Store in session state for real-time chart updates
        st.session_state.filtered_segments = filtered_segments
        st.session_state.filtered_platform_totals = filtered_platform_totals
        st.session_state.filtered_TAM = filtered_TAM
        
        # Show real-time demographic info
        st.sidebar.success(f"📊 **Live Preview**: {len(filtered_segments)} segments, TAM: {filtered_TAM:,} people")
        
        # Clear optimization results when demographics change (to prevent confusion)
        demographics_key = f"{include_male}_{include_female}_{include_18_24}_{include_25_34}_{include_35_44}_{include_45_54}_{include_55_plus}"
        if 'last_demographics' not in st.session_state:
            st.session_state.last_demographics = demographics_key
        elif st.session_state.last_demographics != demographics_key:
            st.session_state.optimization_results = None
            st.session_state.last_demographics = demographics_key
            st.sidebar.info("📊 Charts updated! Run optimization to see new results.")
    else:
        # Clear filtered data if invalid selection
        if 'filtered_segments' in st.session_state:
            del st.session_state.filtered_segments
        if 'filtered_platform_totals' in st.session_state:
            del st.session_state.filtered_platform_totals
        if 'filtered_TAM' in st.session_state:
            del st.session_state.filtered_TAM
    
    # Advanced Options
    with st.sidebar.expander("⚙️ Advanced Options"):
        use_sainsbury = st.checkbox("Use Sainsbury+GWI Hybrid", value=True,
                                   help="Use enhanced quality weighting method")
        convergence_factor = st.slider("Convergence Factor", 0.7, 0.9, 0.85, 0.05,
                                     help="Sainsbury convergence parameter")
        
        # Frequency Optimization
        st.subheader("📊 Frequency Optimization")
        enable_frequency_optimization = st.checkbox(
            "Enable Frequency Optimization",
            value=False,
            help="Optimize to reduce excessive frequency and improve reach efficiency"
        )
        
        if enable_frequency_optimization:
            max_frequency = st.slider(
                "Maximum Frequency Cap",
                min_value=3.0,
                max_value=10.0,
                value=6.0,
                step=0.5,
                help="Maximum acceptable frequency per channel (impressions per person)"
            )
            
            frequency_weight = st.slider(
                "Frequency Penalty Weight",
                min_value=1.0,
                max_value=50.0,
                value=10.0,
                step=5.0,
                help="Higher values = stronger penalty for exceeding frequency cap"
            )
        else:
            max_frequency = 6.0
            frequency_weight = 10.0
    
    # Check if parameters have changed and clear cache if needed
    current_params = {
        'target_reach': target_reach,
        'budget': budget,
        'strategy': strategy,
        'use_sainsbury': use_sainsbury,
        'convergence_factor': convergence_factor,
        'demographic_filters': {
            'include_male': include_male,
            'include_female': include_female,
            'include_18_24': include_18_24,
            'include_25_34': include_25_34,
            'include_35_44': include_35_44,
            'include_45_54': include_45_54,
            'include_55_plus': include_55_plus
        },
        'enable_frequency_optimization': enable_frequency_optimization,
        'max_frequency': max_frequency,
        'frequency_weight': frequency_weight
    }
    
    # Auto-clear cache if parameters changed
    if (st.session_state.last_optimization_params is not None and 
        st.session_state.last_optimization_params != current_params and
        'optimization_results' in st.session_state):
        # Parameters changed, clear cached results
        del st.session_state.optimization_results
        st.sidebar.info("🔄 Parameters changed - cache cleared automatically")
    
    # Run Optimization Button
    if st.sidebar.button("🚀 Run Optimization", type="primary"):
        # Validate demographic selection
        if not any([include_male, include_female]) or not any([include_18_24, include_25_34, include_35_44, include_45_54, include_55_plus]):
            st.error("❌ Please select at least one gender and one age group.")
        else:
            demographic_filters = {
                'include_male': include_male,
                'include_female': include_female,
                'include_18_24': include_18_24,
                'include_25_34': include_25_34,
                'include_35_44': include_35_44,
                'include_45_54': include_45_54,
                'include_55_plus': include_55_plus
            }
            
            with st.spinner("Running optimization..."):
                run_optimization(target_reach, budget, strategy, use_sainsbury, convergence_factor, demographic_filters, enable_frequency_optimization, max_frequency, frequency_weight)
    
    # Cache Management
    st.sidebar.header("🔄 Cache Management")
    if st.sidebar.button("🗑️ Clear Results Cache"):
        # Clear all cached optimization results
        if 'optimization_results' in st.session_state:
            del st.session_state.optimization_results
        if 'filtered_segments' in st.session_state:
            del st.session_state.filtered_segments
        if 'filtered_TAM' in st.session_state:
            del st.session_state.filtered_TAM
        if 'filtered_platform_totals' in st.session_state:
            del st.session_state.filtered_platform_totals
        if 'last_optimization_params' in st.session_state:
            del st.session_state.last_optimization_params
        st.success("✅ Cache cleared! Re-run optimization for fresh results.")
        st.rerun()
    
    # Force Fresh Calculation Button
    if st.sidebar.button("🔄 Force Fresh Calculation", type="primary"):
        # Nuclear option - clear everything and force recalculation
        keys_to_clear = [
            'optimization_results', 'filtered_segments', 'filtered_TAM', 
            'filtered_platform_totals', 'last_optimization_params',
            'real_data_loaded', 'platform_audiences', 'excel_data'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Force Python module reload to ensure latest code
        import importlib
        import reach_calc
        importlib.reload(reach_calc)
        
        st.success("🔄 **FORCED REFRESH** - All cache cleared and modules reloaded! Page will reload.")
        st.rerun()
    
    # Debug section
    st.sidebar.header("🔍 Debug Tools")
    if st.sidebar.button("Debug Reach Calculation"):
        debug_reach_issue()
    
    # Live calculation preview
    if st.sidebar.button("🧪 Test Live Calculation"):
        st.sidebar.write("**Testing current parameters...**")
        try:
            # Run a quick test with current parameters
            from reach_calc import optimize_media_plan_hybrid, plan_kpis_hybrid, CHANNEL_CONSTRAINTS, TAM, REACH_CALC_VERSION
            
            st.sidebar.write(f"Current TAM: {TAM:,}")
            st.sidebar.write(f"Reach Calc Version: {REACH_CALC_VERSION}")
            
            test_result = optimize_media_plan_hybrid(
                target_reach, budget, CHANNEL_CONSTRAINTS, use_sainsbury, convergence_factor,
                False, 6.0, 10.0
            )
            
            if test_result[0]:  # optimal_spend exists
                _, test_kpis = plan_kpis_hybrid(test_result[0], use_sainsbury)
                st.sidebar.success(f"✅ **Live Test**: {test_kpis['NetReach_%_of_TAM']:.1f}% reach")
                st.sidebar.write(f"People: {test_kpis['NetReach_persons']:,}")
                
                # Show if this matches what we expect
                if test_kpis['NetReach_%_of_TAM'] >= 99.9:
                    st.sidebar.error("⚠️ Still showing 100%! Module not reloaded.")
                else:
                    st.sidebar.success("✅ Calculation is working correctly!")
            else:
                st.sidebar.error("❌ Test failed")
        except Exception as e:
            st.sidebar.error(f"❌ Test error: {str(e)}")
    
    # Display Results with cache validation
    if st.session_state.optimization_results:
        # Validate that results are not stale/cached
        results = st.session_state.optimization_results
        if results.get('success', False):
            # Check if this is a fresh calculation (has timestamp)
            if 'timestamp' in results:
                display_results()
            else:
                # Old results without timestamp - force recalculation
                st.warning("⚠️ **Stale results detected** - clearing cache and showing welcome screen")
                del st.session_state.optimization_results
                display_welcome_screen()
        else:
            display_welcome_screen()
    else:
        display_welcome_screen()

def run_optimization(target_reach, budget, strategy, use_sainsbury, convergence_factor, demographic_filters, enable_frequency_optimization, max_frequency, frequency_weight):
    """Run the media planning optimization with demographic filtering"""
    
    # FORCE FRESH CALCULATION: Clear any cached results
    if 'optimization_results' in st.session_state:
        del st.session_state.optimization_results
    
    # Store current parameters for change detection
    current_params = {
        'target_reach': target_reach,
        'budget': budget,
        'strategy': strategy,
        'use_sainsbury': use_sainsbury,
        'convergence_factor': convergence_factor,
        'demographic_filters': demographic_filters,
        'enable_frequency_optimization': enable_frequency_optimization,
        'max_frequency': max_frequency,
        'frequency_weight': frequency_weight
    }
    st.session_state.last_optimization_params = current_params
    
    # Apply demographic filtering
    filtered_segments = filter_segments_by_demographics(**demographic_filters)
    
    # Validate that we have segments selected
    if not filtered_segments:
        st.error("❌ No audience segments selected. Please choose at least one demographic group.")
        return
    
    # Update platform totals based on filtered segments
    filtered_platform_totals = update_platform_totals_for_demographics(filtered_segments)
    filtered_TAM = max(filtered_platform_totals.values())
    
    # Display filtered audience info
    st.info(f"🎯 **Filtered Audience**: {len(filtered_segments)} segments, TAM: {filtered_TAM:,} people")
    
    # Temporarily update global segments for optimization
    global segments, TAM, platform_totals
    original_segments = segments.copy()
    original_TAM = TAM
    original_platform_totals = platform_totals.copy()
    
    # Apply filters
    segments.update(filtered_segments)
    segments = {k: v for k, v in segments.items() if k in filtered_segments}  # Remove unselected segments
    TAM = filtered_TAM
    platform_totals = filtered_platform_totals
    
    # Store filtered data in session state
    st.session_state.filtered_segments = filtered_segments
    st.session_state.filtered_TAM = filtered_TAM
    st.session_state.filtered_platform_totals = filtered_platform_totals
    
    try:
        # Select constraints based on strategy
        if strategy == "Long-Format Focused":
            constraints = LONG_FORMAT_CONSTRAINTS
        elif strategy == "Short-Format Focused":
            constraints = SHORT_FORMAT_CONSTRAINTS
        else:
            constraints = CHANNEL_CONSTRAINTS
        
        # Test maximum possible reach
        max_reach, max_spend_plan = test_max_possible_reach_hybrid(
            budget, constraints, use_sainsbury
        )
        
        # Handle None return from max reach test
        if max_reach is None:
            # Fallback: estimate max reach based on budget and typical CPM
            avg_cpm = sum(cpm.values()) / len(cpm) if cpm else 50
            estimated_impressions = (budget / avg_cpm) * 1000
            estimated_reach = min(95, (estimated_impressions / TAM) * 100)  # Cap at 95%
            max_reach = estimated_reach
            st.warning(f"⚠️ Max reach calculation failed, using estimate: {max_reach:.1f}%")
        
        # Run optimization
        optimal_spend, result = optimize_media_plan_hybrid(
            target_reach, budget, constraints, use_sainsbury, convergence_factor,
            enable_frequency_optimization, max_frequency, frequency_weight
        )
        
        if optimal_spend:
            # Calculate performance
            _, kpis = plan_kpis_hybrid(optimal_spend, use_sainsbury)
            
            # Calculate incremental contributions
            incremental_data = calculate_incremental_contributions(optimal_spend, use_sainsbury)
            
            # Store results with timestamp to ensure freshness
            import time
            st.session_state.optimization_results = {
                'target_reach': target_reach,
                'budget': budget,
                'strategy': strategy,
                'max_reach': max_reach,
                'optimal_spend': optimal_spend,
                'kpis': kpis,
                'incremental_data': incremental_data,
                'use_sainsbury': use_sainsbury,
                'demographic_filters': demographic_filters,
                'filtered_segments': filtered_segments,
                'filtered_TAM': filtered_TAM,
                'filtered_platform_totals': filtered_platform_totals,
                'success': True,
                'timestamp': time.time()  # Add timestamp for freshness verification
            }
            
            st.success("✅ Optimization completed successfully!")
        else:
            st.error("❌ Optimization failed. Try adjusting your parameters.")
            st.session_state.optimization_results = {'success': False}
            
    finally:
        # Restore original segments
        segments.clear()
        segments.update(original_segments)
        TAM = original_TAM
        platform_totals = original_platform_totals

def display_welcome_screen():
    """Display welcome screen with data overview"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Campaign Data Overview")
        
        # Check if demographics are being filtered
        is_filtered = hasattr(st.session_state, 'filtered_TAM')
        
        if is_filtered:
            # Show filtered view
            filtered_TAM = st.session_state.filtered_TAM
            original_TAM = getattr(st.session_state, 'updated_TAM', TAM)
            
            st.markdown(f"**Filtered Audience:** {filtered_TAM:,} people")
            st.caption(f"*Original TAM: {original_TAM:,} people*")
            
            # Show which segments are included
            filtered_segments = getattr(st.session_state, 'filtered_segments', {})
            if filtered_segments:
                segment_list = []
                for seg in filtered_segments.keys():
                    # Format segment names for display
                    gender = 'F' if seg.startswith('F') else 'M'
                    age_group = seg[1:].replace('_', '-').replace('plus', '+')
                    segment_list.append(f"{gender}{age_group}")
                st.info(f"📊 **Active Segments**: {', '.join(segment_list)}")
        else:
            # Show full view
            current_TAM = getattr(st.session_state, 'updated_TAM', TAM)
            st.markdown(f"**Total Addressable Market:** {current_TAM:,} people")
            st.caption("*Select demographics from the sidebar to filter audience*")
        
        # Platform audience breakdown
        if hasattr(st.session_state, 'platform_audiences'):
            # Use filtered platform totals if available, otherwise use current platform totals
            current_platform_totals = getattr(st.session_state, 'filtered_platform_totals', platform_totals)
            is_filtered = hasattr(st.session_state, 'filtered_platform_totals')
            
            # Dynamic title based on filtering state
            title = "Platform Audiences" if is_filtered else "Platform Audiences"
            st.subheader(title)
            
            if is_filtered:
                st.caption("*Showing filtered demographic audience sizes*")
            else:
                st.caption("*Platform audiences represent maximum reach potential*")
            
            platform_data = [
                {'Platform': platform, 'Audience': audience}
                for platform, audience in current_platform_totals.items()
            ]
            
            df_platforms = pd.DataFrame(platform_data)
            
            # Create bar chart with dynamic coloring
            colors = ['#2ca02c' if is_filtered else '#1f77b4',  # Green for filtered, blue for normal
                     '#ff7f0e', '#2ca02c', '#d62728']
            
            fig_platforms = px.bar(
                df_platforms, 
                x='Platform', 
                y='Audience',
                title=f"Platform Audience Sizes{' (Live Update)' if is_filtered else ''}",
                color='Platform',
                color_discrete_sequence=colors,
                text='Audience'
            )
            fig_platforms.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            
            # Calculate dynamic y-axis range to prevent text cutoff
            max_audience = df_platforms['Audience'].max()
            y_range_max = max_audience * 1.15  # Add 15% padding above the highest bar
            
            fig_platforms.update_layout(
                showlegend=False,
                margin=dict(t=80, b=40, l=40, r=40),  # Increase top margin
                yaxis=dict(range=[0, y_range_max]),   # Set y-axis range with padding
                height=400  # Set explicit height for consistent display
            )
            
            # Add annotation if filtered
            if is_filtered:
                fig_platforms.add_annotation(
                    text="Live filtering active",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(color="green", size=10)
                )
            
            st.plotly_chart(fig_platforms, use_container_width=True)
        
        # Segment breakdown
        # Use filtered segments if available, otherwise use all segments
        current_segments = getattr(st.session_state, 'filtered_segments', segments)
        is_filtered = hasattr(st.session_state, 'filtered_segments')
        
        # Dynamic title based on filtering state
        title = "Demographic Segments" if is_filtered else "Demographic Segments"
        st.subheader(title)
        
        if is_filtered:
            st.caption("*Showing only selected demographic segments*")
        else:
            st.caption("*All available demographic segments*")
        
        segment_data = []
        for seg, pop in current_segments.items():
            gender = 'Female' if seg.startswith('F') else 'Male'
            age_group = seg[1:].replace('_', '-').replace('plus', '+')  # Format for display
            segment_data.append({
                'Segment': seg,
                'Gender': gender,
                'Age Group': age_group,
                'Population': pop
            })
        
        df_segments = pd.DataFrame(segment_data)
        
        # Create demographic chart with dynamic styling
        chart_title = f"Demographic Breakdown{' (Live Update)' if is_filtered else ''}"
        fig_demo = px.sunburst(
            df_segments,
            path=['Gender', 'Age Group'],
            values='Population',
            title=chart_title,
            color='Population',
            color_continuous_scale='Viridis' if is_filtered else 'Blues'
        )
        
        # Add annotation if filtered
        if is_filtered:
            fig_demo.add_annotation(
                text="Real-time filtering",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(color="green", size=10)
            )
        
        st.plotly_chart(fig_demo, use_container_width=True)
    
    with col2:
        st.header("Getting Started")
        
        st.markdown("""
        **How to use this tool:**
        
        1. **Set your target reach** - Choose what percentage of TAM you want to reach
        
        2. **Enter your budget** - Total media spend available
        
        3. **Select strategy** - Choose focus area:
           - **Balanced**: Standard optimization
           - **Long-Format**: Brand building focus
           - **Short-Format**: Performance focus
        
        4. **Run optimization** - Click the button to get your media plan
        
        **Features:**
        - Real campaign data integration
        - Cross-platform reach deduplication
        - Quality-weighted optimization
        - Incremental contribution analysis
        - Interactive visualizations
        """)
        
        st.info("👈 Configure your parameters in the sidebar and click 'Run Optimization' to get started!")

def display_results():
    """Display optimization results"""
    
    results = st.session_state.optimization_results
    
    if not results.get('success', False):
        st.error("No valid optimization results to display.")
        return
    
    # Results Header
    st.header("Optimization Results")
    
    # Show when results were calculated for transparency
    if 'timestamp' in results:
        import time
        calc_time = time.strftime('%H:%M:%S', time.localtime(results['timestamp']))
        st.caption(f"🕐 **Results calculated at**: {calc_time} (fresh calculation)")
    else:
        st.caption("⚠️ **Results may be cached** - use 'Clear Results Cache' for fresh calculation")
    
    # Show demographic filter information if applied
    if 'demographic_filters' in results:
        filters = results['demographic_filters']
        
        # Build filter summary
        genders = []
        if filters.get('include_male'): genders.append('Male')
        if filters.get('include_female'): genders.append('Female')
        
        ages = []
        if filters.get('include_18_24'): ages.append('18-24')
        if filters.get('include_25_34'): ages.append('25-34')
        if filters.get('include_35_44'): ages.append('35-44')
        if filters.get('include_45_54'): ages.append('45-54')
        if filters.get('include_55_plus'): ages.append('55+')
        
        filter_summary = f"**🎯 Target Demographics**: {', '.join(genders)} | {', '.join(ages)}"
        st.info(filter_summary)
    
    # Key Metrics - Updated to 5 columns to include Long-form %
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Use filtered TAM if available, otherwise use updated TAM
        current_TAM = results.get('filtered_TAM', getattr(st.session_state, 'updated_TAM', TAM))
        target_people = (results['target_reach'] / 100) * current_TAM
        st.metric(
            "Target Reach",
            f"{results['target_reach']}%",
            f"{target_people / 1_000_000:.2f}M of {current_TAM / 1_000_000:.1f}M people"
        )
    
    with col2:
        actual_reach = results['kpis']['NetReach_%_of_TAM']
        
        # DEBUG: Force display of exact value to prevent any caching issues
        st.caption(f"🔍 **Debug**: Raw KPI value = {actual_reach}")
        
        delta = actual_reach - results['target_reach']
        current_TAM = results.get('filtered_TAM', getattr(st.session_state, 'updated_TAM', TAM))
        achieved_people = (actual_reach / 100) * current_TAM
        target_people = (results['target_reach'] / 100) * current_TAM
        
        # Calculate percentage points difference (achieved % - target %)
        percentage_points_difference = actual_reach - results['target_reach']
        
        # The delta string MUST start with the signed number for the color/arrow to work.
        delta_str = f"{percentage_points_difference:+.1f}% ({achieved_people / 1_000_000:.2f}M vs {target_people / 1_000_000:.2f}M Target)"
        
        # Force exact display - no rounding issues
        display_value = f"{actual_reach:.1f}%"
        if actual_reach >= 99.9:
            display_value = f"⚠️ {actual_reach:.1f}% (Check calculation!)"
        
        st.metric(
            "Achieved Reach",
            display_value,
            delta=delta_str
        )
    
    with col3:
        total_spend = sum(results['optimal_spend'].values())
        budget_used = total_spend / results['budget']
        st.metric(
            "Budget Used",
            f"₺{total_spend:,.0f}",
            f"{budget_used:.1%} of budget"
        )
    
    with col4:
        if results['use_sainsbury']:
            quality_people = results['kpis']['SainsburyQuality_persons']
        else:
            quality_people = results['kpis']['QualityReach_persons']
        
        # Calculate as percentage of reached audience instead of TAM
        reached_people = results['kpis']['NetReach_persons']
        quality_reach_pct = (quality_people / reached_people * 100) if reached_people > 0 else 0
        
        st.metric(
            "HypeQ Score",
            f"{quality_reach_pct:.1f}%",
            f"{quality_people:,} people"
        )
    
    with col5:
        # Calculate Long-form % as percentage of people reached by long-format channels
        total_spend = sum(results['optimal_spend'].values())
        longform_spend = sum(spend for channel, spend in results['optimal_spend'].items() 
                           if channel.endswith('_LF'))
        
        longform_spend_pct = (longform_spend / total_spend * 100) if total_spend > 0 else 0
        
        # Calculate number of people reached by long-format channels
        longform_reach_pct = calculate_longform_reach(results['optimal_spend'])
        
        # Get TAM for absolute people calculation (use proper TAM, not segment sum)
        current_platform_totals = getattr(st.session_state, 'filtered_platform_totals', platform_totals)
        tam = max(current_platform_totals.values()) if current_platform_totals else 66_100_000
        
        # Calculate absolute number of people reached by long-format channels
        # This should be a subset of total campaign reach, not exceed it
        total_campaign_reach_persons = results['kpis']['NetReach_persons']
        longform_people_reached = int((longform_reach_pct / 100) * total_campaign_reach_persons)
        
        st.metric(
            "Long-form %",
            f"{longform_spend_pct:.1f}%",
            f"{longform_people_reached:,} people"
        )
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Media Mix", "📈 Performance", "💡 Insights"])
    
    with tab1:
        display_media_mix(results)
    
    with tab2:
        display_performance_analysis(results)
    
    with tab3:
        display_insights(results)
        
        # Add frequency analysis if available
        if 'optimal_spend' in results:
            st.subheader("📊 Frequency Analysis")
            
            # Calculate frequencies for each channel
            frequency_data = []
            current_segments = results.get('filtered_segments', getattr(st.session_state, 'updated_segments', segments))
            
            for channel, spend in results['optimal_spend'].items():
                if spend > 0:
                    frequency = calculate_channel_frequency(spend, channel, current_segments)
                    
                    # Determine status
                    if frequency <= 4.0:
                        status = "🟢 Optimal"
                        status_color = "green"
                    elif frequency <= 6.0:
                        status = "🟡 Acceptable"
                        status_color = "orange"
                    else:
                        status = "🔴 High"
                        status_color = "red"
                    
                    frequency_data.append({
                        'Channel': channel,
                        'Spend': f"₺{spend:,.0f}",
                        'Frequency': f"{frequency:.1f}",
                        'Status': status,
                        'Recommendation': get_frequency_recommendation(frequency)
                    })
            
            if frequency_data:
                df_frequency = pd.DataFrame(frequency_data)
                st.dataframe(df_frequency, use_container_width=True, hide_index=True)
                
                # Frequency insights
                high_freq_channels = [d for d in frequency_data if d['Status'] == "🔴 High"]
                if high_freq_channels:
                    st.warning(f"⚠️ **High Frequency Alert**: {len(high_freq_channels)} channel(s) have frequency > 6.0. Consider enabling frequency optimization to improve reach efficiency.")
                else:
                    st.success("✅ **Frequency Status**: All channels have acceptable frequency levels.")

def get_frequency_recommendation(frequency):
    """Get recommendation based on frequency level."""
    if frequency <= 3.0:
        return "Consider increasing spend for better impact"
    elif frequency <= 4.0:
        return "Optimal frequency range"
    elif frequency <= 6.0:
        return "Acceptable but monitor closely"
    elif frequency <= 8.0:
        return "Consider reducing spend or enabling frequency optimization"
    else:
        return "High frequency - enable frequency optimization"

def display_media_mix(results):
    """Display media mix visualization"""
    
    # Get updated data from session state or fallback to globals
    current_cpm = getattr(st.session_state, 'updated_cpm', cpm)
    
    # CRITICAL FIX: Ensure all expected channels are present in current_cpm
    # If real data is missing for any channel, use global fallback
    expected_channels = ['TV_LF', 'YT_LF', 'YT_SF', 'Meta_SF', 'TT_SF']
    for channel in expected_channels:
        if channel not in current_cpm:
            current_cpm[channel] = cpm.get(channel, 10)  # Use global or default fallback
            print(f"🔧 FALLBACK: {channel} not in real data, using global CPM: ₺{current_cpm[channel]}")
    
    # Debug: Check if YT_SF CPM is correct
    if 'YT_SF' in current_cpm:
        print(f"🔍 DEBUG: YT_SF CPM = ₺{current_cpm['YT_SF']} (source: {'session_state' if hasattr(st.session_state, 'updated_cpm') else 'global'})")
    else:
        print(f"🔍 DEBUG: YT_SF not found in current_cpm. Available channels: {list(current_cpm.keys())}")
    
    # Use filtered segments if available from optimization results
    current_segments = results.get('filtered_segments', getattr(st.session_state, 'updated_segments', segments))
    current_aud_prof = getattr(st.session_state, 'updated_aud_prof', aud_prof)
    current_reach_par = getattr(st.session_state, 'updated_reach_par', reach_par)
    
    # CRITICAL FIX: Ensure all expected channels have audience profiles
    # If real data is missing audience profile for any channel, use global fallback
    expected_channels = ['TV_LF', 'YT_LF', 'YT_SF', 'Meta_SF', 'TT_SF']
    for channel in expected_channels:
        if channel not in current_aud_prof:
            current_aud_prof[channel] = aud_prof.get(channel, {})
        elif not current_aud_prof[channel]:  # Empty dict
            current_aud_prof[channel] = aud_prof.get(channel, {})
    
    # CRITICAL FIX: Ensure all expected channels have reach parameters
    for channel in expected_channels:
        if channel not in current_reach_par:
            current_reach_par[channel] = reach_par.get(channel, (0.5, 0.2))
    
    # Debug YT_SF specifically
    print(f"🔍 DEBUG: YT_SF audience profile keys: {list(current_aud_prof.get('YT_SF', {}).keys())}")
    print(f"🔍 DEBUG: Available segments: {list(current_segments.keys())}")
    
    spend_data = []
    for channel, spend in results['optimal_spend'].items():
        if spend > 0:
            format_type = "Long-Format" if channel.endswith('_LF') else "Short-Format"
            platform = channel.replace('_LF', '').replace('_SF', '')
            
            # Calculate additional metrics using updated data
            channel_cpm = current_cpm.get(channel, 0)  # Get CPM for this channel
            impressions = (spend / channel_cpm) * 1000 if channel_cpm > 0 else 0  # Calculate impressions
            
            # Calculate frequency using the same method as optimization (consistent with backend)
            frequency = calculate_channel_frequency(spend, channel, segments)
            
            # Calculate absolute reach per channel (gross reach)
            # This shows the individual reach for each channel without deduplication
            if frequency > 0:
                absolute_reach = int(impressions / frequency)
            else:
                absolute_reach = 0
            
            # Calculate reach percentage based on TAM
            current_platform_totals = getattr(st.session_state, 'filtered_platform_totals', platform_totals)
            tam_size = max(current_platform_totals.values()) if current_platform_totals else 66_100_000
            
            if tam_size > 0 and absolute_reach > 0:
                channel_reach_pct = (absolute_reach / tam_size) * 100
            else:
                channel_reach_pct = 0
            
            spend_data.append({
                'Channel': channel,
                'Platform': platform,
                'Format': format_type,
                'Spend': spend,
                'Percentage': spend / sum(results['optimal_spend'].values()) * 100,
                'CPM': channel_cpm,
                'Impressions': impressions,
                'Reach_Pct': channel_reach_pct,
                'Absolute_Reach': absolute_reach,
                'Frequency': frequency
            })
    
    df_spend = pd.DataFrame(spend_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of spend by channel
        fig_pie = px.pie(
            df_spend,
            values='Spend',
            names='Channel',
            title="Budget Allocation by Channel"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart by platform and format
        fig_bar = px.bar(
            df_spend,
            x='Channel',
            y='Spend',
            color='Format',
            title="Spend by Channel and Format",
            color_discrete_map={
                'Long-Format': '#1f77b4',
                'Short-Format': '#ff7f0e'
            }
        )
        fig_bar.update_xaxes(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Detailed breakdown table with additional metrics
    st.subheader("Detailed Budget Breakdown")
    
    display_df = df_spend.copy()
    display_df['Spend'] = display_df['Spend'].apply(lambda x: f"₺{x:,.0f}")
    display_df['Percentage'] = display_df['Percentage'].apply(lambda x: f"{x:.1f}%")
    display_df['CPM'] = display_df['CPM'].apply(lambda x: f"₺{x:.0f}")
    display_df['Impressions'] = display_df['Impressions'].apply(lambda x: f"{x:,.0f}")
    display_df['Frequency'] = display_df['Frequency'].apply(lambda x: f"{x:.1f}")
    display_df['Reach_Pct'] = display_df['Reach_Pct'].apply(lambda x: f"{x:.1f}%")
    display_df['Absolute_Reach'] = display_df['Absolute_Reach'].apply(lambda x: f"{x:,.0f}")
    
    # Reorder columns for better display
    display_df = display_df[['Channel', 'Platform', 'Format', 'Spend', 'Percentage', 'CPM', 'Impressions', 'Frequency', 'Reach_Pct', 'Absolute_Reach']]
    
    # Rename columns for display
    display_df.columns = ['Channel', 'Platform', 'Format', 'Spend', 'Percentage', 'CPM', 'Impressions', 'Frequency', 'Reach %', 'Absolute Reach']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def display_performance_analysis(results):
    """Display performance analysis"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Reach Performance")
        
        # Reach metrics
        reach_data = {
            'Metric': ['Target Reach', 'Achieved Reach', 'Max Possible', 'HypeQ Score'],
            'Value': [
                results['target_reach'],
                results['kpis']['NetReach_%_of_TAM'],
                results['max_reach'],
                results['kpis']['SainsburyQuality_%_TAM'] if results['use_sainsbury'] else results['kpis']['QualityReach_%_TAM']
            ]
        }
        
        df_reach = pd.DataFrame(reach_data)
        
        fig_reach = px.bar(
            df_reach,
            x='Metric',
            y='Value',
            title="Reach Performance Metrics (%)",
            color='Metric'
        )
        st.plotly_chart(fig_reach, use_container_width=True)
    
    with col2:
        st.subheader("Efficiency Analysis")
        
        # Calculate efficiency metrics
        total_spend = sum(results['optimal_spend'].values())
        reach_efficiency = results['kpis']['NetReach_%_of_TAM'] / (total_spend / 1_000_000)
        quality_efficiency = (results['kpis']['SainsburyQuality_%_TAM'] if results['use_sainsbury'] else results['kpis']['QualityReach_%_TAM']) / (total_spend / 1_000_000)
        
        efficiency_data = {
            'Metric': ['Reach per ₺1M', 'HypeQ per ₺1M', 'Budget Utilization'],
            'Value': [
                reach_efficiency,
                quality_efficiency,
                (total_spend / results['budget']) * 100
            ]
        }
        
        df_efficiency = pd.DataFrame(efficiency_data)
        
        fig_efficiency = px.bar(
            df_efficiency,
            x='Metric',
            y='Value',
            title="Efficiency Metrics",
            color='Metric'
        )
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Performance summary
    st.subheader("Performance Summary")
    
    performance_summary = f"""
    **🎯 Reach Performance:**
    - Target: {results['target_reach']}% of TAM
    - Achieved: {results['kpis']['NetReach_%_of_TAM']:.1f}% of TAM ({results['kpis']['NetReach_persons']:,} people)
    - Gap: {results['kpis']['NetReach_%_of_TAM'] - results['target_reach']:+.1f} percentage points
    
    **💰 Budget Efficiency:**
    - Budget Used: ₺{sum(results['optimal_spend'].values()):,.0f} ({sum(results['optimal_spend'].values())/results['budget']:.1%})
    - Reach per ₺1M: {reach_efficiency:.1f}%
    - HypeQ per ₺1M: {quality_efficiency:.1f}%
    
    **⭐ HypeQ Performance:**
    - HypeQ Score: {results['kpis']['SainsburyQuality_%_TAM'] if results['use_sainsbury'] else results['kpis']['QualityReach_%_TAM']:.1f}% of TAM
    - HypeQ People: {results['kpis']['SainsburyQuality_persons'] if results['use_sainsbury'] else results['kpis']['QualityReach_persons']:,}
    """
    
    st.markdown(performance_summary)

def display_incremental_analysis(results):
    """Display incremental contribution analysis"""
    
    incremental_data = results['incremental_data']
    
    if not incremental_data:
        st.warning("No incremental data available.")
        return
    
    # Prepare data for visualization
    inc_data = []
    for channel, data in incremental_data.items():
        if data['spend'] > 0:
            inc_data.append({
                'Channel': channel,
                'Spend': data['spend'],
                'Incremental_Reach': data['incremental_reach_pp'],
                'Incremental_Quality': data['incremental_quality_pp'],
                'Reach_Efficiency': data['incremental_reach_per_1m'],
                'Quality_Efficiency': data['incremental_quality_per_1m'],
                'Format': 'Long-Format' if channel.endswith('_LF') else 'Short-Format'
            })
    
    df_inc = pd.DataFrame(inc_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Incremental reach chart
        fig_inc_reach = px.bar(
            df_inc,
            x='Channel',
            y='Incremental_Reach',
            color='Format',
            title="Incremental Reach Contribution (pp)",
            color_discrete_map={
                'Long-Format': '#1f77b4',
                'Short-Format': '#ff7f0e'
            }
        )
        fig_inc_reach.update_xaxes(tickangle=45)
        st.plotly_chart(fig_inc_reach, use_container_width=True)
    
    with col2:
        # Reach efficiency chart
        fig_efficiency = px.scatter(
            df_inc,
            x='Spend',
            y='Reach_Efficiency',
            size='Incremental_Reach',
            color='Format',
            hover_name='Channel',
            title="Reach Efficiency vs Spend",
            labels={'Reach_Efficiency': 'Reach per ₺1M (%)', 'Spend': 'Spend (₺)'}
        )
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Incremental analysis table
    st.subheader("Incremental Contribution Details")
    
    display_inc_df = df_inc.copy()
    display_inc_df['Spend'] = display_inc_df['Spend'].apply(lambda x: f"₺{x:,.0f}")
    display_inc_df['Incremental_Reach'] = display_inc_df['Incremental_Reach'].apply(lambda x: f"{x:.2f}pp")
    display_inc_df['Incremental_Quality'] = display_inc_df['Incremental_Quality'].apply(lambda x: f"{x:.2f}pp")
    display_inc_df['Reach_Efficiency'] = display_inc_df['Reach_Efficiency'].apply(lambda x: f"{x:.1f}%")
    display_inc_df['Quality_Efficiency'] = display_inc_df['Quality_Efficiency'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(display_inc_df, use_container_width=True, hide_index=True)
    
    # Key insights
    st.subheader("🔍 Key Insights")
    
    # Find best performers
    best_reach = max(incremental_data.items(), key=lambda x: x[1]['incremental_reach_pp'])
    best_quality = max(incremental_data.items(), key=lambda x: x[1]['incremental_quality_pp'])
    most_efficient = max(incremental_data.items(), key=lambda x: x[1]['incremental_reach_per_1m'])
    
    insights = f"""
    **🎯 Top Performers:**
    - **Highest Reach Contributor:** {best_reach[0]} (+{best_reach[1]['incremental_reach_pp']:.2f}pp)
    - **Highest HypeQ Contributor:** {best_quality[0]} (+{best_quality[1]['incremental_quality_pp']:.2f}pp)
    - **Most Efficient:** {most_efficient[0]} ({most_efficient[1]['incremental_reach_per_1m']:.1f}% per ₺1M)
    
    **📊 Understanding Incremental Analysis:**
    - Values show reach/HypeQ LOST when removing each channel
    - Higher values = more critical channels for your campaign
    - Zero values may indicate mathematical artifacts from HypeQ weighting
    """
    
    st.markdown(insights)

def display_insights(results):
    """Display strategic insights and recommendations"""
    
    st.subheader("💡 Strategic Insights")
    
    # Calculate key metrics for insights
    total_spend = sum(results['optimal_spend'].values())
    lf_spend = sum(spend for channel, spend in results['optimal_spend'].items() if channel.endswith('_LF'))
    sf_spend = sum(spend for channel, spend in results['optimal_spend'].items() if channel.endswith('_SF'))
    
    lf_share = lf_spend / total_spend if total_spend > 0 else 0
    sf_share = sf_spend / total_spend if total_spend > 0 else 0
    
    actual_reach = results['kpis']['NetReach_%_of_TAM']
    target_reach = results['target_reach']
    
    # Generate insights based on results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Campaign Performance")
        
        if abs(actual_reach - target_reach) < 1:
            st.success(f"✅ **Target Achieved**: Reached {actual_reach:.1f}% (within 1% of target)")
        elif actual_reach > target_reach:
            st.info(f"📈 **Target Exceeded**: Reached {actual_reach:.1f}% (+{actual_reach-target_reach:.1f}pp above target)")
        else:
            st.warning(f"⚠️ **Target Missed**: Reached {actual_reach:.1f}% (-{target_reach-actual_reach:.1f}pp below target)")
        
        budget_efficiency = (total_spend / results['budget'])
        if budget_efficiency < 0.8:
            st.info(f"💰 **Budget Efficient**: Used only {budget_efficiency:.1%} of budget")
        elif budget_efficiency > 0.95:
            st.warning(f"💸 **Budget Intensive**: Used {budget_efficiency:.1%} of budget")
        else:
            st.success(f"💲 **Budget Optimized**: Used {budget_efficiency:.1%} of budget")
    
    with col2:
        st.markdown("### 📊 Media Mix Strategy")
        
        if lf_share > 0.6:
            st.info(f"📺 **Long-Format Focused**: {lf_share:.1%} long-format allocation")
            st.markdown("*Best for brand building and storytelling*")
        elif sf_share > 0.6:
            st.info(f"📱 **Short-Format Focused**: {sf_share:.1%} short-format allocation")
            st.markdown("*Best for performance and reach maximization*")
        else:
            st.success(f"⚖️ **Balanced Strategy**: {lf_share:.1%} long-format, {sf_share:.1%} short-format")
            st.markdown("*Optimal balance of reach and HypeQ*")
    
    # Recommendations
    st.subheader("🎯 Strategic Recommendations")
    
    recommendations = []
    
    # Budget recommendations
    if budget_efficiency < 0.7:
        recommendations.append("💰 **Budget Optimization**: Consider reducing budget or increasing target reach to improve efficiency")
    elif budget_efficiency > 0.95:
        recommendations.append("💸 **Budget Constraint**: Consider increasing budget for better reach potential")
    
    # Format recommendations
    if lf_share > 0.7:
        recommendations.append("📺 **Long-Format Strategy**: Excellent for brand building. Consider adding some short-format for reach amplification")
    elif sf_share > 0.7:
        recommendations.append("📱 **Short-Format Strategy**: Great for performance. Consider adding long-format for quality depth")
    
    # Quality recommendations
    quality_reach = results['kpis']['SainsburyQuality_%_TAM'] if results['use_sainsbury'] else results['kpis']['QualityReach_%_TAM']
    if quality_reach / actual_reach > 0.5:
        recommendations.append("⭐ **High HypeQ**: Strong engagement quality ratio. Great for brand campaigns")
    elif quality_reach / actual_reach < 0.2:
        recommendations.append("🔍 **HypeQ Opportunity**: Consider increasing long-format allocation for better engagement")
    
    # Platform recommendations
    incremental_data = results['incremental_data']
    if incremental_data:
        best_channel = max(incremental_data.items(), key=lambda x: x[1]['incremental_reach_pp'])
        recommendations.append(f"🎯 **Channel Focus**: {best_channel[0]} shows highest incremental value - consider increasing allocation")
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Action items
    st.subheader("✅ Next Steps")
    
    action_items = [
        "**Review incremental analysis** to identify optimization opportunities",
        "**Test different target reach levels** to find optimal budget efficiency",
        "**Compare long-format vs short-format strategies** for your specific campaign goals",
        "**Monitor actual campaign performance** against these projections",
        "**Adjust allocations** based on real-time performance data"
    ]
    
    for item in action_items:
        st.markdown(f"• {item}")

# =========  DEBUG FUNCTIONS  ============================

def debug_reach_issue():
    """Debug function to test reach calculation issue"""
    st.subheader("🔍 Debug: Reach Calculation Analysis")
    
    # Test with different spend allocations directly
    test_cases = [
        {"TV_LF": 1000000, "YT_LF": 1000000, "YT_SF": 1000000, "Meta_SF": 1000000, "TT_SF": 1000000},
        {"TV_LF": 5000000, "YT_LF": 500000, "YT_SF": 500000, "Meta_SF": 0, "TT_SF": 0},
        {"TV_LF": 0, "YT_LF": 0, "YT_SF": 2000000, "Meta_SF": 2000000, "TT_SF": 2000000},
        {"TV_LF": 3000000, "YT_LF": 3000000, "YT_SF": 0, "Meta_SF": 0, "TT_SF": 0},
    ]
    
    st.write("Testing different spend allocations...")
    
    for i, spend_dict in enumerate(test_cases, 1):
        st.write(f"**Test Case {i}:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Spend allocation:")
            for channel, spend in spend_dict.items():
                st.write(f"  {channel}: ₺{spend:,}")
        
        with col2:
            try:
                # Calculate reach directly
                from reach_calc import plan_kpis_hybrid
                _, kpis = plan_kpis_hybrid(spend_dict, use_sainsbury=True)
                st.write("Results:")
                st.write(f"  Net Reach: {kpis['NetReach_%_of_TAM']}% of TAM")
                st.write(f"  Quality Reach: {kpis['SainsburyQuality_%_TAM']}% of TAM")
            except Exception as e:
                st.error(f"Error calculating reach: {e}")
        
        st.write("---")
    
    # Test extreme allocations
    st.subheader("Extreme Test Cases")
    
    extreme_cases = [
        ("All TV", {"TV_LF": 6000000, "YT_LF": 0, "YT_SF": 0, "Meta_SF": 0, "TT_SF": 0}),
        ("All Meta", {"TV_LF": 0, "YT_LF": 0, "YT_SF": 0, "Meta_SF": 6000000, "TT_SF": 0}),
        ("All YouTube SF", {"TV_LF": 0, "YT_LF": 0, "YT_SF": 6000000, "Meta_SF": 0, "TT_SF": 0}),
        ("Equal Split", {"TV_LF": 1200000, "YT_LF": 1200000, "YT_SF": 1200000, "Meta_SF": 1200000, "TT_SF": 1200000}),
    ]
    
    for case_name, spend_dict in extreme_cases:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**{case_name}:**")
            for channel, spend in spend_dict.items():
                if spend > 0:
                    st.write(f"  {channel}: ₺{spend:,}")
        
        with col2:
            try:
                from reach_calc import plan_kpis_hybrid
                _, kpis = plan_kpis_hybrid(spend_dict, use_sainsbury=True)
                st.write(f"Reach: {kpis['NetReach_%_of_TAM']}% of TAM")
                st.write(f"Quality: {kpis['SainsburyQuality_%_TAM']}% of TAM")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Data validation
    st.subheader("Data Validation")
    try:
        from reach_calc import TAM, segments, cpm
        st.write(f"TAM: {TAM:,}")
        st.write(f"Segments: {len(segments)} ({sum(segments.values()):,} total)")
        st.write(f"CPM data: {cpm}")
    except Exception as e:
        st.error(f"Error accessing data: {e}")

if __name__ == "__main__":
    main() 