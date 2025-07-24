# ------------------------------------------------------------
#  Sainsbury-Weighted Reach Calculator with Real Campaign Data
#  © 2025  Powered by 5 months of real campaign data
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import pprint
from scipy.optimize import minimize, Bounds
from scipy.optimize import differential_evolution
import os

# =========  AUDIENCE DATA LOADING  ===========================

def load_audience_segments_from_excel(file_path='audience-segments.xlsx'):
    """
    Load audience segments from Excel file instead of hardcoded values.
    
    Expected Excel structure:
    - Column A: Segment (F18-24, F25-34, ..., M55+)
    - Column B: TV (audience size for each segment)
    - Column C: YouTube (audience size for each segment)
    - Column D: Meta (audience size for each segment)
    - Column E: TikTok (audience size for each segment)
    
    Returns:
        tuple: (segments_dict, platform_totals, tam_value)
    """
    try:
        if not os.path.exists(file_path):
            print(f"⚠️  Excel file not found: {file_path}")
            print("   Using fallback hardcoded values...")
            return load_fallback_audience_data()
        
        # Read the Excel file
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        
        # Validate required columns
        required_columns = ['Segment', 'TV', 'YouTube', 'Meta', 'TikTok']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"⚠️  Missing columns in Excel file: {missing_columns}")
            print("   Using fallback hardcoded values...")
            return load_fallback_audience_data()
        
        # Convert segment names to match internal format (F18-24 -> F18_24, F55+ -> F55_plus)
        segments_dict = {}
        for _, row in df.iterrows():
            segment_name = str(row['Segment']).replace('-', '_').replace('+', '_plus')
            segments_dict[segment_name] = int(row['TV'])  # Use TV as the baseline for segment population
        
        # Calculate platform totals
        platform_totals = {
            'TV': int(df['TV'].sum()),
            'YouTube': int(df['YouTube'].sum()),
            'Meta': int(df['Meta'].sum()),
            'TikTok': int(df['TikTok'].sum())
        }
        
        # TAM is the largest platform audience
        tam_value = max(platform_totals.values())
        largest_platform = max(platform_totals, key=platform_totals.get)
        
        print(f"✅ Loaded audience data from Excel file: {file_path}")
        print(f"   📊 TAM: {tam_value:,} people (largest platform: {largest_platform})")
        print(f"   🎯 Segments: {len(segments_dict)}")
        print(f"   📺 Platform totals: {platform_totals}")
        
        return segments_dict, platform_totals, tam_value
        
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        print("   Using fallback hardcoded values...")
        return load_fallback_audience_data()

def load_fallback_audience_data():
    """
    Fallback function with hardcoded values if Excel file is not available.
    """
    fallback_segments = {
        # Female segments
        "F18_24": 3_740_000,
        "F25_34": 5_760_000,
        "F35_44": 5_730_000, 
        "F45_54": 4_540_000,
        "F55_plus": 4_010_000,
        
        # Male segments  
        "M18_24": 4_050_000,
        "M25_34": 6_090_000,
        "M35_44": 5_870_000,
        "M45_54": 4_930_000,
        "M55_plus": 4_420_000,
    }
    
    fallback_platform_totals = {
        'TV': 48_392_000,
        'YouTube': 49_140_000,
        'Meta': 66_100_000,
        'TikTok': 39_262_000
    }
    
    fallback_tam = max(fallback_platform_totals.values())
    
    print("📝 Using fallback hardcoded audience data")
    
    return fallback_segments, fallback_platform_totals, fallback_tam

# Load audience data on import
segments, platform_totals, TAM = load_audience_segments_from_excel()

# Keep original segment populations - do NOT scale to TAM
# TAM represents the theoretical maximum reach (largest platform audience)
# Segments represent the actual demographic breakdown used for reach calculations
# The sum of segments may be less than TAM, which is correct - not all TAM is reachable
segment_total = sum(segments.values())
print(f"📊 SEGMENTS: {segment_total:,} people across {len(segments)} demographics")
print(f"🎯 TAM: {TAM:,} people (maximum theoretical reach)")
print(f"📺 Platform totals: {platform_totals}")
print(f"💡 Note: Segments sum to {segment_total:,} (TV baseline), TAM is {TAM:,} (Meta max)")

# =========  INPUT TABLES  ===================================

# 1) Population by segment is now loaded from Excel above

# 2) Cost curves  (all CPM in ₺ for simplicity)
cpm = {
    "TV_LF":   80,    # TV only has long format
    "YT_LF":   20,
    "YT_SF":   15,    # Same audience pool as YT_LF
    "Meta_SF":  8,    # Meta only has short format
    "TT_SF":    6,    # TikTok only has short format
}

# 3) Reach-curve parameters  (Max,  k)   Reach = Max · (1 − e^(−k·GRP))
# Adjusted to more realistic values to prevent impossible reach calculations
reach_par = {
    "TV_LF":   (0.65, 0.08),  # TV only has long format - more realistic max reach
    "YT_LF":   (0.50, 0.15),  # YouTube long format - conservative reach
    "YT_SF":   (0.45, 0.18),  # YouTube short format - lower than long format
    "Meta_SF": (0.50, 0.18),  # Meta only has short format - realistic reach
    "TT_SF":   (0.40, 0.20),  # TikTok only has short format - most conservative
}

# 4) Audience-profile matrix  (share of each segment in the format's reach)
# NOTE: Long and short formats on same platform have IDENTICAL profiles (same audience pool)
# UPDATED: More realistic audience profiles for 50% reach capability
aud_prof = {
    "TV_LF": {  # TV only has long format - broad reach across all demographics
        "F18_24": 0.35, "F25_34": 0.45, "F35_44": 0.55, "F45_54": 0.65, "F55_plus": 0.75,
        "M18_24": 0.30, "M25_34": 0.40, "M35_44": 0.50, "M45_54": 0.60, "M55_plus": 0.70
    },
    # YouTube: Same audience pool for both formats - strong digital reach
    "YT_LF": {
        "F18_24": 0.65, "F25_34": 0.70, "F35_44": 0.55, "F45_54": 0.45, "F55_plus": 0.30,
        "M18_24": 0.70, "M25_34": 0.75, "M35_44": 0.60, "M45_54": 0.50, "M55_plus": 0.35
    },
    "YT_SF": {  # IDENTICAL to YT_LF - same audience pool
        "F18_24": 0.65, "F25_34": 0.70, "F35_44": 0.55, "F45_54": 0.45, "F55_plus": 0.30,
        "M18_24": 0.70, "M25_34": 0.75, "M35_44": 0.60, "M45_54": 0.50, "M55_plus": 0.35
    },
    # Meta: Short format only - strong social media reach
    "Meta_SF": {
        "F18_24": 0.75, "F25_34": 0.80, "F35_44": 0.70, "F45_54": 0.55, "F55_plus": 0.40,
        "M18_24": 0.65, "M25_34": 0.75, "M35_44": 0.65, "M45_54": 0.50, "M55_plus": 0.35
    },
    # TikTok: Short format only - youth-focused but expanding
    "TT_SF": {
        "F18_24": 0.85, "F25_34": 0.75, "F35_44": 0.60, "F45_54": 0.40, "F55_plus": 0.25,
        "M18_24": 0.80, "M25_34": 0.70, "M35_44": 0.55, "M45_54": 0.35, "M55_plus": 0.20
    },
}

# 5) Enhanced quality weights system (TV_LF is baseline 1.00)
# Designed to properly reward long-format content quality
w = {
    "TV_LF":   1.00,  # Premium baseline - will be updated with VAT data
    "YT_LF":   0.75,  # Will be calculated with long-format bonus
    "YT_SF":   0.35,  # Will be calculated with short-format penalty
    "Meta_SF": 0.25,  # Meta short format only
    "TT_SF":   0.20,  # TikTok short format only
}

# Format-specific quality multipliers for engagement depth
QUALITY_MULTIPLIERS = {
    'TV_LF': 1.0,      # Premium baseline (storytelling, brand impact)
    'YT_LF': 0.85,     # Strong long-format (detailed content, tutorials)
    'YT_SF': 0.40,     # Limited short-format (quick engagement)
    'Meta_SF': 0.35,   # Basic short-format (impulse interaction)
    'TT_SF': 0.45,     # Slightly better short-format (viral, trendy)
}

def calculate_enhanced_quality_weights(vat_data):
    """
    Enhanced quality weights calculation that properly rewards long-format content.
    
    Args:
        vat_data: Dict with format keys and average watch time values
        Example: {'TV_LF': 16, 'YT_LF': 5.5, 'YT_SF': 2.5, ...}
    
    Returns:
        Dict with enhanced quality weights prioritizing long-format
    """
    if 'TV_LF' not in vat_data:
        print("Warning: No TV_LF VAT data found, using enhanced defaults")
        return w
    
    tv_vat = vat_data['TV_LF']
    quality_weights = {}
    
    for format_code, vat_seconds in vat_data.items():
        # Base VAT ratio
        vat_ratio = vat_seconds / tv_vat
        
        # Get format-specific quality multiplier
        base_multiplier = QUALITY_MULTIPLIERS.get(format_code, 0.5)
        
        if format_code.endswith('_LF'):
            # LONG-FORMAT BONUS: Progressive scaling + engagement depth bonus
            engagement_bonus = 1.0 + (vat_ratio * 0.3)  # Reward higher VAT
            storytelling_bonus = 1.15  # Inherent long-format quality bonus
            quality_weights[format_code] = min(1.0, base_multiplier * engagement_bonus * storytelling_bonus)
        else:
            # SHORT-FORMAT PENALTY: Acknowledges limited engagement depth
            attention_penalty = 0.6  # 40% penalty for short attention spans
            depth_penalty = max(0.3, vat_ratio)  # Lower floor for very short VAT
            quality_weights[format_code] = max(0.1, base_multiplier * attention_penalty * depth_penalty)
    
    return quality_weights

# Keep original function for backward compatibility
def calculate_quality_weights_from_vat(vat_data):
    """Legacy function - redirects to enhanced version."""
    return calculate_enhanced_quality_weights(vat_data)

# 6) GWI audience-overlap factors  (symmetrised, digital only)
# Increased overlap factors to prevent impossible reach calculations
ov = {('YT','Meta'):0.85,   # Higher overlap between YT and Meta
      ('YT','TT')  :0.65,   # Higher overlap between YT and TT
      ('Meta','TT'):0.70}   # Higher overlap between Meta and TT

# =========  PLANNING INPUTS  =================================

# Budget and target constraints
TOTAL_BUDGET = 6_000_000  # Total budget in ₺
TARGET_REACH_PCT = 44     # Target reach as % of TAM
MIN_CHANNEL_SPEND = 50_000  # Minimum spend per channel if used

# Channel strategy constraints (updated for full budget utilization)
CHANNEL_CONSTRAINTS = {
    'TV_LF': {'min_pct': 0.10, 'max_pct': 0.70},    # TV can take majority if efficient
    'YT_LF': {'min_pct': 0.05, 'max_pct': 0.40},    # YouTube LF increased limit
    'YT_SF': {'min_pct': 0.05, 'max_pct': 0.40},    # YouTube SF increased limit
    'Meta_SF': {'min_pct': 0.05, 'max_pct': 0.50},  # Meta SF increased for efficiency
    'TT_SF': {'min_pct': 0.00, 'max_pct': 0.35},    # TikTok SF increased limit
}

# Strategic media plan alternatives (optimized for reach while maintaining long-format focus)
LONG_FORMAT_CONSTRAINTS = {
    'TV_LF': {'min_pct': 0.25, 'max_pct': 0.70},    # Higher TV allocation for brand building and reach
    'YT_LF': {'min_pct': 0.08, 'max_pct': 0.35},    # YouTube LF for storytelling
    'YT_SF': {'min_pct': 0.05, 'max_pct': 0.40},    # YouTube SF allowed for reach amplification
    'Meta_SF': {'min_pct': 0.08, 'max_pct': 0.45},  # Meta SF for engagement and reach
    'TT_SF': {'min_pct': 0.03, 'max_pct': 0.30},    # TikTok SF for quality and efficiency
}

SHORT_FORMAT_CONSTRAINTS = {
    'TV_LF': {'min_pct': 0.10, 'max_pct': 0.35},    # Limited TV for reach efficiency
    'YT_LF': {'min_pct': 0.00, 'max_pct': 0.20},    # Limited YouTube LF
    'YT_SF': {'min_pct': 0.15, 'max_pct': 0.45},    # Higher YouTube SF for frequency
    'Meta_SF': {'min_pct': 0.15, 'max_pct': 0.50},  # Higher Meta SF for frequency
    'TT_SF': {'min_pct': 0.10, 'max_pct': 0.40},    # Higher TikTok SF for cost efficiency
}



# =========  CORE FUNCTIONS  =================================

# Version check for debugging
REACH_CALC_VERSION = "v2.0_fixed"  # Updated version with reach capping fixes

def grp(spend_lira, cpm_lira, pop):
    """GRPs delivered to *one* segment."""
    return (spend_lira / cpm_lira) / (pop / 1_000)

def reach_single(max_r, k, grp_val):
    """Exponential reach curve."""
    return max_r * (1 - np.exp(-k * grp_val))

def platform_reach_segment(R_fmt_seg, platform):
    """Combine long + short formats inside the same platform."""
    if platform == 'YT':
        return 1 - (1 - R_fmt_seg['YT_LF']) * (1 - R_fmt_seg['YT_SF'])
    if platform == 'Meta':
        return R_fmt_seg['Meta_SF']  # Only short format for Meta
    if platform == 'TT':
        return R_fmt_seg['TT_SF']  # Only short format for TikTok
    if platform == 'TV':
        return R_fmt_seg['TV_LF']
    raise ValueError("unknown platform")

def net_reach_segment(R_platform, ov_factors):
    """Deduplicate YT, Meta, TT using inclusion–exclusion with GWI overlaps."""
    R_TV   = R_platform['TV']
    R_YT   = R_platform['YT']
    R_Meta = R_platform['Meta']
    R_TT   = R_platform['TT']

    # pairwise intersections (digital only)
    I_YM = ov_factors[('YT','Meta')] * min(R_YT, R_Meta)
    I_YT = ov_factors[('YT','TT')]   * min(R_YT, R_TT)
    I_MT = ov_factors[('Meta','TT')] * min(R_Meta, R_TT)

    # TV assumed independent
    net_reach = R_TV + R_YT + R_Meta + R_TT - I_YM - I_YT - I_MT
    
    # Cap at 100% to prevent impossible reach values
    return min(1.0, net_reach)

def enhanced_quality_reach_segment(R_fmt_seg, weights):
    """
    Enhanced quality reach calculation prioritizing long-format engagement depth.
    
    This method:
    1. Separates long-format and short-format contributions
    2. Prioritizes engagement depth over volume
    3. Applies explicit quality hierarchies
    4. Ensures long-format content gets proper quality recognition
    
    Args:
        R_fmt_seg: Reach by format for this segment
        weights: Enhanced quality weights from VAT data
        
    Returns:
        Quality reach that properly rewards long-format content
    """
    
    # Separate formats by type
    lf_formats = {fmt: reach for fmt, reach in R_fmt_seg.items() if fmt.endswith('_LF')}
    sf_formats = {fmt: reach for fmt, reach in R_fmt_seg.items() if fmt.endswith('_SF')}
    
    # STEP 1: Calculate Long-Format Quality Foundation
    # Long-format provides the quality foundation with premium weighting
    lf_quality_reach = 0
    lf_quality_universe = 1.0
    
    # Sort long-format by quality (TV > YT)
    lf_quality_order = ['TV_LF', 'YT_LF']
    
    for fmt in lf_quality_order:
        if fmt in lf_formats and lf_formats[fmt] > 0:
            # Long-format gets full quality contribution with depth bonus
            format_quality_weight = weights[fmt] * QUALITY_MULTIPLIERS[fmt]
            
            # Apply engagement depth bonus for long-format
            engagement_depth_bonus = 1.2  # 20% bonus for storytelling capability
            
            # Calculate this format's quality contribution
            format_quality_reach = (format_quality_weight * lf_formats[fmt] * 
                                  lf_quality_universe * engagement_depth_bonus)
            
            lf_quality_reach += format_quality_reach
            
            # Reduce available universe for next long-format (diminishing returns)
            lf_quality_universe *= (1 - weights[fmt] * lf_formats[fmt] * 0.7)
            lf_quality_universe = max(0.2, lf_quality_universe)  # Maintain quality floor
    
    # STEP 2: Calculate Short-Format Quality Supplement  
    # Short-format can only supplement, not replace long-format quality
    sf_quality_reach = 0
    sf_available_universe = max(0.3, 1.0 - lf_quality_reach)  # SF limited to remaining universe
    
    # Sort short-format by efficiency (but with quality penalties)
    sf_efficiency_order = ['YT_SF', 'TT_SF', 'Meta_SF']
    
    for fmt in sf_efficiency_order:
        if fmt in sf_formats and sf_formats[fmt] > 0:
            # Short-format gets penalized quality contribution
            format_quality_weight = weights[fmt] * QUALITY_MULTIPLIERS[fmt]
            
            # Apply attention span penalty for short-format
            attention_penalty = 0.6  # 40% penalty for limited engagement depth
            
            # Calculate supplemental quality contribution
            format_contribution = (format_quality_weight * sf_formats[fmt] * 
                                 sf_available_universe * attention_penalty)
            
            sf_quality_reach += format_contribution
            
            # Reduce available universe with stronger diminishing returns
            sf_available_universe *= (1 - weights[fmt] * sf_formats[fmt] * 0.9)
            sf_available_universe = max(0.1, sf_available_universe)
    
    # STEP 3: Combine with Long-Format Priority
    total_quality_reach = lf_quality_reach + sf_quality_reach
    
    # Apply final quality hierarchy enforcement
    if lf_quality_reach > 0:
        # If long-format is present, it should dominate quality calculation
        lf_dominance_factor = min(1.0, lf_quality_reach / (lf_quality_reach + sf_quality_reach + 0.001))
        quality_hierarchy_bonus = 1.0 + (lf_dominance_factor * 0.1)  # Up to 10% bonus for LF dominance
        total_quality_reach *= quality_hierarchy_bonus
    
    return min(1.0, total_quality_reach)

# Legacy function for backward compatibility
def quality_reach_segment(R_fmt_seg, weights):
    """Legacy quality reach - redirects to enhanced version."""
    return enhanced_quality_reach_segment(R_fmt_seg, weights)

# =========  ENHANCED SAINSBURY + GWI HYBRID FUNCTIONS  =================

def enhanced_sainsbury_quality_reach_segment(R_fmt_seg, weights, convergence_factor=0.75):
    """
    Enhanced Sainsbury approach using our new long-format priority system.
    
    Args:
        R_fmt_seg: Reach by format for this segment
        weights: Enhanced quality weights from VAT data
        convergence_factor: Convergence factor (adjusted for new system)
    
    Returns:
        Quality reach using enhanced Sainsbury + long-format priority approach
    """
    
    # Use our new enhanced quality reach calculation as the foundation
    base_quality_reach = enhanced_quality_reach_segment(R_fmt_seg, weights)
    
    # Apply Sainsbury-style convergence effects for additional sophistication
    total_formats = len([fmt for fmt, reach in R_fmt_seg.items() if reach > 0])
    
    if total_formats > 1:
        # Multi-format convergence penalty (reduces quality when spreading across many formats)
        convergence_penalty = convergence_factor ** (total_formats - 1)
        
        # But give bonus for long-format concentration
        lf_formats = len([fmt for fmt, reach in R_fmt_seg.items() if fmt.endswith('_LF') and reach > 0])
        sf_formats = len([fmt for fmt, reach in R_fmt_seg.items() if fmt.endswith('_SF') and reach > 0])
        
        if lf_formats > sf_formats:
            # Bonus for focusing on long-format quality
            lf_focus_bonus = 1.0 + (lf_formats / (lf_formats + sf_formats)) * 0.15
            convergence_penalty *= lf_focus_bonus
        
        base_quality_reach *= convergence_penalty
    
    return min(1.0, base_quality_reach)

# Legacy function for backward compatibility  
def sainsbury_weighted_reach_segment(R_fmt_seg, weights, convergence_factor=0.85):
    """Legacy Sainsbury function - redirects to enhanced version."""
    return enhanced_sainsbury_quality_reach_segment(R_fmt_seg, weights, convergence_factor)

def hybrid_sainsbury_gwi_reach_segment(R_fmt_seg, weights, ov_factors, convergence_factor=0.85):
    """
    HYBRID APPROACH: Sainsbury weighted quality + GWI cross-platform deduplication
    
    This combines:
    1. Sainsbury's sophisticated quality weighting within platforms
    2. GWI's proven overlap factors between platforms
    3. Your real VAT data for quality weights
    
    Args:
        R_fmt_seg: Reach by format for this segment
        weights: Quality weights from VAT data
        ov_factors: GWI overlap factors
        convergence_factor: Sainsbury convergence parameter
    
    Returns:
        tuple: (net_reach, sainsbury_quality_reach)
    """
    
    # Step 1: Calculate platform-level reach (same as before)
    R_platform = {
        'TV'  : platform_reach_segment(R_fmt_seg, 'TV'),
        'YT'  : platform_reach_segment(R_fmt_seg, 'YT'),
        'Meta': platform_reach_segment(R_fmt_seg, 'Meta'),
        'TT'  : platform_reach_segment(R_fmt_seg, 'TT'),
    }
    
    # Step 2: GWI-based cross-platform deduplication (proven method)
    R_TV   = R_platform['TV']
    R_YT   = R_platform['YT']
    R_Meta = R_platform['Meta']
    R_TT   = R_platform['TT']

    # GWI pairwise intersections (digital only)
    I_YM = ov_factors[('YT','Meta')] * min(R_YT, R_Meta)
    I_YT = ov_factors[('YT','TT')]   * min(R_YT, R_TT)
    I_MT = ov_factors[('Meta','TT')] * min(R_Meta, R_TT)

    # Net reach using GWI inclusion-exclusion
    net_reach = R_TV + R_YT + R_Meta + R_TT - I_YM - I_YT - I_MT
    
    # Cap at 100% to prevent impossible reach values
    net_reach = min(1.0, net_reach)
    
    # Step 3: Enhanced Sainsbury quality reach (long-format priority method)
    sainsbury_quality = enhanced_sainsbury_quality_reach_segment(R_fmt_seg, weights, convergence_factor)
    
    return net_reach, sainsbury_quality

# =========  ENHANCED MAIN CALCULATION WITH HYBRID OPTION  ==============

def plan_kpis_hybrid(spend_dict, use_sainsbury=True, convergence_factor=0.85):
    """
    Enhanced KPI calculation with Sainsbury + GWI hybrid option.
    
    Args:
        spend_dict: Media spend allocation
        use_sainsbury: If True, use Sainsbury+GWI hybrid; if False, use original method
        convergence_factor: Sainsbury convergence parameter (0.8-0.9)
    
    Returns:
        Enhanced KPIs with both reach methodologies
    """
    seg_rows = []
    
    for seg, pop in segments.items():
        # --- reach per *format* inside this segment -----------------------
        R_fmt_seg = {}
        for fmt in spend_dict:
            # FIX: Audience profile should be applied to population, not spend
            # aud_prof[fmt][seg] represents the % of this segment reachable by this format
            addressable_pop = pop * aud_prof[fmt][seg]
            g = grp(spend_dict[fmt],
                    cpm[fmt],
                    addressable_pop)
            R_fmt_seg[fmt] = reach_single(*reach_par[fmt], g)

        if use_sainsbury:
            # HYBRID METHOD: Sainsbury + GWI
            R_net, Q_sainsbury = hybrid_sainsbury_gwi_reach_segment(
                R_fmt_seg, w, ov, convergence_factor
            )
            # Also calculate original for comparison
        R_platform = {
            'TV'  : platform_reach_segment(R_fmt_seg, 'TV'),
            'YT'  : platform_reach_segment(R_fmt_seg, 'YT'),
            'Meta': platform_reach_segment(R_fmt_seg, 'Meta'),
            'TT'  : platform_reach_segment(R_fmt_seg, 'TT'),
        }
            Q_original = quality_reach_segment(R_fmt_seg, w)
        else:
            # ORIGINAL METHOD
            R_platform = {
                'TV'  : platform_reach_segment(R_fmt_seg, 'TV'),
                'YT'  : platform_reach_segment(R_fmt_seg, 'YT'),
                'Meta': platform_reach_segment(R_fmt_seg, 'Meta'),
                'TT'  : platform_reach_segment(R_fmt_seg, 'TT'),
            }
        R_net = net_reach_segment(R_platform, ov)
            Q_original = quality_reach_segment(R_fmt_seg, w)
            Q_sainsbury = Q_original  # Same as original

        seg_rows.append({
            'Segment'              : seg,
            'Pop'                  : pop,
            'NetReach_%'           : R_net * 100,
            'QualityReach_%'       : Q_original * 100,
            'SainsburyQuality_%'   : Q_sainsbury * 100,
            'NetReach_pers'        : R_net * pop,
            'QualityReach_pers'    : Q_original * pop,
            'SainsburyQuality_pers': Q_sainsbury * pop,
        })

    # ---- roll up ---------------------------------------------------------
    net_pers       = sum(r['NetReach_pers']        for r in seg_rows)
    qual_pers      = sum(r['QualityReach_pers']    for r in seg_rows)
    sainsbury_pers = sum(r['SainsburyQuality_pers']for r in seg_rows)

    # No capping needed - reach calculation should produce realistic values
    # with properly scaled segment populations

    total_kpis = {
        'NetReach_persons'        : round(net_pers),
        'NetReach_%_of_TAM'       : round(net_pers / TAM * 100, 1),
        'QualityReach_persons'    : round(qual_pers),
        'QualityReach_%_TAM'      : round(qual_pers / TAM * 100, 1),
        'SainsburyQuality_persons': round(sainsbury_pers),
        'SainsburyQuality_%_TAM'  : round(sainsbury_pers / TAM * 100, 1),
    }
    return seg_rows, total_kpis

# =========  OPTIMIZATION FUNCTIONS  ============================

def spend_array_to_dict(spend_array):
    """Convert optimization array to spend dictionary."""
    channels = ["TV_LF", "YT_LF", "YT_SF", "Meta_SF", "TT_SF"]
    return dict(zip(channels, spend_array))

def objective_function_hybrid(spend_array, target_reach_pct, budget, use_sainsbury=True):
    """Enhanced objective function using Sainsbury + GWI hybrid with budget utilization incentive."""
    spend_dict = spend_array_to_dict(spend_array)
    
    # Calculate current reach using hybrid method
    _, kpis = plan_kpis_hybrid(spend_dict, use_sainsbury=use_sainsbury)
    current_reach_pct = kpis['NetReach_%_of_TAM']
    
    # Calculate total spend
    total_spend = sum(spend_array)
    
    # Primary penalty: reach gap
    reach_penalty = abs(current_reach_pct - target_reach_pct) * 100
    
    # Budget constraint penalty (for exceeding budget)
    budget_penalty = max(0, total_spend - budget) * 0.01
    
    # Budget utilization incentive (encourage using full budget)
    budget_utilization = total_spend / budget
    if budget_utilization < 0.95:  # If using less than 95% of budget
        underutilization_penalty = (0.95 - budget_utilization) * 25  # Gentle penalty
    else:
        underutilization_penalty = 0
    
    # If we're far from target, prioritize reach over budget efficiency
    reach_gap = abs(current_reach_pct - target_reach_pct)
    if reach_gap > 10:  # If more than 10pp away from target
        reach_priority_multiplier = 1.5  # Increase reach penalty weight
    else:
        reach_priority_multiplier = 1.0
    
    return (reach_penalty * reach_priority_multiplier) + budget_penalty + underutilization_penalty

# Keep original function for backward compatibility and comparison
def plan_kpis(spend_dict):
    """Original KPI calculation method (GWI + simple quality weighting)."""
    return plan_kpis_hybrid(spend_dict, use_sainsbury=False)

def objective_function(spend_array, target_reach_pct, budget):
    """Original objective function for backward compatibility."""
    return objective_function_hybrid(spend_array, target_reach_pct, budget, use_sainsbury=False)

# =========  ENHANCED OPTIMIZATION WITH HYBRID METHOD  ==================

def optimize_media_plan_hybrid(target_reach_pct, budget, constraints=None, use_sainsbury=True, 
                              convergence_factor=0.85, enable_frequency_optimization=False, 
                              max_frequency=6.0, frequency_weight=25.0):
    """
    Enhanced media optimization with Sainsbury + GWI hybrid option and optional frequency optimization.
    
    Args:
        target_reach_pct: Target reach as % of TAM
        budget: Total budget available
        constraints: Dict of channel constraints (min/max % of budget)
        use_sainsbury: If True, use Sainsbury+GWI hybrid; if False, use original
        convergence_factor: Sainsbury convergence parameter (0.8-0.9)
        enable_frequency_optimization: If True, apply frequency constraints
        max_frequency: Maximum acceptable frequency per channel
        frequency_weight: Weight for frequency penalty in optimization
    
    Returns:
        dict: Optimized spend allocation
    """
    if constraints is None:
        constraints = CHANNEL_CONSTRAINTS
    
    # Use frequency optimization if enabled
    if enable_frequency_optimization:
        return optimize_media_plan_with_frequency(
            target_reach_pct, budget, constraints, use_sainsbury, 
            max_frequency, frequency_weight, convergence_factor
        )
    
    # Set up bounds
    bounds = create_optimization_bounds(budget, constraints)
    
    # Optimize using differential evolution with improved settings for complex problems
    result = differential_evolution(
        objective_function_hybrid,
        bounds,
        args=(target_reach_pct, budget, use_sainsbury),
        seed=42,
        maxiter=700,  # Increased for better reach optimization
        popsize=35,   # Increased for better exploration
        atol=1e-4,    # Tighter tolerance for better solutions
        tol=1e-4,     # Tighter tolerance for better solutions
        polish=True,  # Added local optimization polish step
        disp=False    # Suppress verbose output
    )
    
    if result.success:
        optimal_spend = spend_array_to_dict(result.x)
        
        # Double-check budget constraint
        total_spend = sum(optimal_spend.values())
        if total_spend > budget * 1.01:  # Allow 1% tolerance
            print(f"Warning: Budget exceeded by {total_spend - budget:,.0f}")
            # Scale down proportionally
            scale_factor = budget / total_spend
            optimal_spend = {k: v * scale_factor for k, v in optimal_spend.items()}
        
        return optimal_spend, result
    else:
        print(f"Optimization failed: {result.message}")
        # Try with even more relaxed settings if first attempt fails
        print("Attempting optimization with relaxed constraints...")
        
        result_relaxed = differential_evolution(
            objective_function_hybrid,
            bounds,
            args=(target_reach_pct, budget, use_sainsbury),
            seed=123,     # Different seed
            maxiter=300,  # Moderate iterations
            popsize=20,   # Standard population
            atol=1e-2,    # Very relaxed tolerance
            tol=1e-2,     # Very relaxed tolerance
            polish=True,
            disp=False
        )
        
        if result_relaxed.success:
            optimal_spend = spend_array_to_dict(result_relaxed.x)
            total_spend = sum(optimal_spend.values())
            if total_spend > budget * 1.01:
                scale_factor = budget / total_spend
                optimal_spend = {k: v * scale_factor for k, v in optimal_spend.items()}
            print("✅ Optimization succeeded with relaxed settings")
            return optimal_spend, result_relaxed
        else:
            print(f"Both optimization attempts failed: {result_relaxed.message}")
            return None, result

def test_max_possible_reach_hybrid(budget, constraints=None, use_sainsbury=True):
    """Test maximum reach with hybrid method."""
    if constraints is None:
        constraints = CHANNEL_CONSTRAINTS
    
    bounds = create_optimization_bounds(budget, constraints)
    
    def maximize_reach_objective(spend_array):
        """Objective to maximize reach (minimize negative reach)."""
        try:
            spend_dict = spend_array_to_dict(spend_array)
            _, kpis = plan_kpis_hybrid(spend_dict, use_sainsbury=use_sainsbury)
            
            # Calculate total spend
            total_spend = sum(spend_array)
            
            # Heavy penalty for exceeding budget
            if total_spend > budget:
                return 1000 + (total_spend - budget) * 0.01
            
            return -kpis['NetReach_%_of_TAM']  # Negative because we minimize
        except Exception as e:
            # Return a large penalty if calculation fails
            return 1000
    
    try:
        result = differential_evolution(
            maximize_reach_objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15
        )
        
        if result.success:
            max_spend = spend_array_to_dict(result.x)
            total_spend = sum(max_spend.values())
            if total_spend <= budget * 1.01:  # Allow small tolerance
                _, max_kpis = plan_kpis_hybrid(max_spend, use_sainsbury=use_sainsbury)
                return max_kpis['NetReach_%_of_TAM'], max_spend
        
        # If optimization didn't succeed, try a simpler fallback
        # Use equal allocation as fallback
        channels = ["TV_LF", "YT_LF", "YT_SF", "Meta_SF", "TT_SF"]
        equal_spend = budget / len(channels)
        fallback_spend = {channel: equal_spend for channel in channels}
        
        _, fallback_kpis = plan_kpis_hybrid(fallback_spend, use_sainsbury=use_sainsbury)
        return fallback_kpis['NetReach_%_of_TAM'], fallback_spend
        
    except Exception as e:
        print(f"Max reach calculation failed: {e}")
        return None, None

def test_max_possible_reach(budget, constraints=None):
    """Test what's the maximum reach achievable with given budget."""
    if constraints is None:
        constraints = CHANNEL_CONSTRAINTS
    
    bounds = create_optimization_bounds(budget, constraints)
    
    def maximize_reach_objective(spend_array):
        """Objective to maximize reach (minimize negative reach)."""
        spend_dict = spend_array_to_dict(spend_array)
        _, kpis = plan_kpis(spend_dict)
        
        # Calculate total spend
        total_spend = sum(spend_array)
        
        # Heavy penalty for exceeding budget
        if total_spend > budget:
            return 1000 + (total_spend - budget) * 0.01
        
        return -kpis['NetReach_%_of_TAM']  # Negative because we minimize
    
    result = differential_evolution(
        maximize_reach_objective,
        bounds,
        seed=42,
        maxiter=100,
        popsize=15
    )
    
    if result.success:
        max_spend = spend_array_to_dict(result.x)
        total_spend = sum(max_spend.values())
        if total_spend <= budget * 1.01:  # Allow small tolerance
            _, max_kpis = plan_kpis(max_spend)
            return max_kpis['NetReach_%_of_TAM'], max_spend
    
    return None, None

def create_optimization_bounds(budget, constraints):
    """Create bounds for each channel based on constraints."""
    channels = ["TV_LF", "YT_LF", "YT_SF", "Meta_SF", "TT_SF"]
    bounds = []
    
    for channel in channels:
        if channel in constraints:
            min_spend = constraints[channel]['min_pct'] * budget
            max_spend = constraints[channel]['max_pct'] * budget
        else:
            min_spend = 0
            max_spend = budget
        
        bounds.append((min_spend, max_spend))
    
    return bounds

def optimize_media_plan(target_reach_pct, budget, constraints=None):
    """
    Find optimal media allocation to hit target reach within budget.
    
    Args:
        target_reach_pct: Target reach as % of TAM
        budget: Total budget available
        constraints: Dict of channel constraints (min/max % of budget)
    
    Returns:
        dict: Optimized spend allocation
    """
    if constraints is None:
        constraints = CHANNEL_CONSTRAINTS
    
    # Set up bounds
    bounds = create_optimization_bounds(budget, constraints)
    
    # Optimize using differential evolution
    result = differential_evolution(
        objective_function,
        bounds,
        args=(target_reach_pct, budget),
        seed=42,
        maxiter=200,
        popsize=20,
        atol=1e-4,
        tol=1e-4
    )
    
    if result.success:
        optimal_spend = spend_array_to_dict(result.x)
        
        # Double-check budget constraint
        total_spend = sum(optimal_spend.values())
        if total_spend > budget * 1.01:  # Allow 1% tolerance
            print(f"Warning: Budget exceeded by {total_spend - budget:,.0f}")
            # Scale down proportionally
            scale_factor = budget / total_spend
            optimal_spend = {k: v * scale_factor for k, v in optimal_spend.items()}
        
        return optimal_spend, result
    else:
        print(f"Optimization failed: {result.message}")
        return None, result

# =========  PLANNING WORKFLOW  ================================

def run_media_planning():
    """Main planning workflow."""
    print("="*60)
    print("MEDIA PLANNING OPTIMIZATION")
    print("="*60)
    print(f"Target: {TARGET_REACH_PCT}% of TAM ({TARGET_REACH_PCT/100*TAM:,.0f} people)")
    print(f"Budget: ₺{TOTAL_BUDGET:,}")
    print()
    
    # First, test maximum possible reach
    print("--- FEASIBILITY CHECK ---")
    max_reach, max_spend_plan = test_max_possible_reach(TOTAL_BUDGET)
    if max_reach:
        print(f"Maximum possible reach with budget: {max_reach:.1f}% of TAM")
        if max_reach < TARGET_REACH_PCT:
            print(f"⚠️  WARNING: Target {TARGET_REACH_PCT}% may not be achievable with ₺{TOTAL_BUDGET:,}")
            print(f"   Consider increasing budget or reducing target to ~{max_reach:.0f}%")
        else:
            print(f"✅ Target {TARGET_REACH_PCT}% appears achievable")
    print()
    
    # Run optimization
    optimal_spend, result = optimize_media_plan(TARGET_REACH_PCT, TOTAL_BUDGET)
    
    if optimal_spend:
        print("--- OPTIMIZED MEDIA PLAN ---")
        total_optimized = sum(optimal_spend.values())
        for channel, spend in optimal_spend.items():
            pct = spend / total_optimized * 100
            print(f"{channel:>10}: ₺{spend:>10,.0f} ({pct:>5.1f}%)")
        print(f"{'TOTAL':>10}: ₺{total_optimized:>10,.0f}")
        print()
        
        # Calculate performance of optimized plan
        seg_detail, kpis = plan_kpis(optimal_spend)
        
        print("--- OPTIMIZED PLAN PERFORMANCE ---")
        print(f"Net Reach: {kpis['NetReach_%_of_TAM']}% of TAM ({kpis['NetReach_persons']:,} people)")
        print(f"Quality Reach: {kpis['QualityReach_%_TAM']}% of TAM ({kpis['QualityReach_persons']:,} people)")
        print(f"Budget Used: ₺{total_optimized:,.0f} ({total_optimized/TOTAL_BUDGET:.1%})")
        
        # Performance vs target
        reach_gap = TARGET_REACH_PCT - kpis['NetReach_%_of_TAM']
        if abs(reach_gap) < 1:
            print(f"Target achieved (within 1% tolerance)")
        elif reach_gap > 0:
            print(f"Target missed by {reach_gap:.1f} percentage points")
        else:
            print(f"Target exceeded by {abs(reach_gap):.1f} percentage points")
        
        lf_spend = optimal_spend['TV_LF'] + optimal_spend['YT_LF']
        print(f"Long-form share: {lf_spend / total_optimized:.1%}")
        print()
        
        return optimal_spend, kpis
    else:
        print("Optimization failed - using fallback approach")
        return None, None

# =========  ENHANCED PLANNING WORKFLOW  ================================

def run_media_planning_comparison():
    """Enhanced planning workflow comparing original vs Sainsbury+GWI hybrid."""
    print("="*60)
    print("ENHANCED MEDIA PLANNING: SAINSBURY + GWI HYBRID")
    print("="*60)
    print(f"Target: {TARGET_REACH_PCT}% of TAM ({TARGET_REACH_PCT/100*TAM:,.0f} people)")
    print(f"Budget: ₺{TOTAL_BUDGET:,}")
    print()
    
    # Test both methods
    print("--- FEASIBILITY CHECK (BOTH METHODS) ---")
    
    # Original method
    max_reach_original, _ = test_max_possible_reach(TOTAL_BUDGET)
    print(f"Original Method Max Reach: {max_reach_original:.1f}% of TAM")
    
    # Sainsbury + GWI hybrid
    max_reach_hybrid, _ = test_max_possible_reach_hybrid(TOTAL_BUDGET, use_sainsbury=True)
    print(f"Sainsbury+GWI Hybrid Max Reach: {max_reach_hybrid:.1f}% of TAM")
    
    print(f"Hybrid Improvement: +{max_reach_hybrid - max_reach_original:.1f} percentage points")
    print()
    
    # Optimize with hybrid method
    print("--- OPTIMIZED MEDIA PLAN (SAINSBURY + GWI HYBRID) ---")
    optimal_spend, result = optimize_media_plan_hybrid(TARGET_REACH_PCT, TOTAL_BUDGET, use_sainsbury=True)
    
    if optimal_spend:
        total_optimized = sum(optimal_spend.values())
        for channel, spend in optimal_spend.items():
            pct = spend / total_optimized * 100
            print(f"{channel:>10}: ₺{spend:>10,.0f} ({pct:>5.1f}%)")
        print(f"{'TOTAL':>10}: ₺{total_optimized:>10,.0f}")
        print()
        
        # Calculate performance with both methods
        seg_detail_hybrid, kpis_hybrid = plan_kpis_hybrid(optimal_spend, use_sainsbury=True)
        seg_detail_original, kpis_original = plan_kpis_hybrid(optimal_spend, use_sainsbury=False)
        
        print("--- PERFORMANCE COMPARISON ---")
        print(f"{'Metric':<25} {'Original':<15} {'Sainsbury+GWI':<15} {'Improvement':<12}")
        print("-" * 70)
        
        # Net Reach
        original_net = kpis_original['NetReach_%_of_TAM']
        hybrid_net = kpis_hybrid['NetReach_%_of_TAM']
        net_improvement = hybrid_net - original_net
        print(f"{'Net Reach':<25} {original_net:<15.1f}% {hybrid_net:<15.1f}% {net_improvement:+.1f}pp")
        
        # Quality Reach
        original_qual = kpis_original['QualityReach_%_TAM']
        hybrid_qual = kpis_hybrid['SainsburyQuality_%_TAM']
        qual_improvement = hybrid_qual - original_qual
        print(f"{'Quality Reach':<25} {original_qual:<15.1f}% {hybrid_qual:<15.1f}% {qual_improvement:+.1f}pp")
        
        # Budget efficiency
        original_efficiency = original_net / (total_optimized / 1_000_000)
        hybrid_efficiency = hybrid_net / (total_optimized / 1_000_000)
        print(f"{'Reach per ₺1M':<25} {original_efficiency:<15.1f}% {hybrid_efficiency:<15.1f}% {hybrid_efficiency-original_efficiency:+.1f}%")
        
        print()
        print("--- SAINSBURY+GWI HYBRID INSIGHTS ---")
        print(f"Net Reach: {hybrid_net}% of TAM ({kpis_hybrid['NetReach_persons']:,} people)")
        print(f"Sainsbury Quality: {hybrid_qual}% of TAM ({kpis_hybrid['SainsburyQuality_persons']:,} people)")
        print(f"Budget Used: ₺{total_optimized:,.0f} ({total_optimized/TOTAL_BUDGET:.1%})")
        
        # Performance vs target
        reach_gap = TARGET_REACH_PCT - hybrid_net
        if abs(reach_gap) < 1:
            print(f"Target achieved (within 1% tolerance)")
        elif reach_gap > 0:
            print(f"Target missed by {reach_gap:.1f} percentage points")
        else:
            print(f"Target exceeded by {abs(reach_gap):.1f} percentage points")
        
        lf_spend = optimal_spend['TV_LF'] + optimal_spend['YT_LF']
        print(f"Long-form share: {lf_spend / total_optimized:.1%}")
        print()
        
        return optimal_spend, kpis_hybrid
    else:
        print("Hybrid optimization failed - using original approach")
        return run_media_planning()

# =========  STRATEGIC MEDIA PLAN ALTERNATIVES  ==================

def run_strategic_media_planning():
    """
    Run dual optimization to provide Long-Format and Short-Format focused media plans.
    
    Returns:
        tuple: (long_format_plan, short_format_plan, comparative_analysis)
    """
    print("="*70)
    print("🎯 STRATEGIC MEDIA PLAN ALTERNATIVES")
    print("="*70)
    print(f"Target: {TARGET_REACH_PCT}% of TAM ({TARGET_REACH_PCT/100*TAM:,.0f} people)")
    print(f"Budget: ₺{TOTAL_BUDGET:,}")
    print()
    
    # Test feasibility for both approaches
    print("--- STRATEGIC FEASIBILITY ANALYSIS ---")
    
    # Long-Format Focused
    max_reach_lf, _ = test_max_possible_reach_hybrid(TOTAL_BUDGET, LONG_FORMAT_CONSTRAINTS, use_sainsbury=True)
    print(f"Long-Format Focused Max Reach: {max_reach_lf:.1f}% of TAM")
    
    # Short-Format Focused  
    max_reach_sf, _ = test_max_possible_reach_hybrid(TOTAL_BUDGET, SHORT_FORMAT_CONSTRAINTS, use_sainsbury=True)
    print(f"Short-Format Focused Max Reach: {max_reach_sf:.1f}% of TAM")
    
    if max_reach_lf < TARGET_REACH_PCT and max_reach_sf < TARGET_REACH_PCT:
        print(f"⚠️  WARNING: Target {TARGET_REACH_PCT}% not achievable with either strategy")
        print(f"   Consider increasing budget or reducing target to ~{max(max_reach_lf, max_reach_sf):.0f}%")
    elif max_reach_lf >= TARGET_REACH_PCT and max_reach_sf >= TARGET_REACH_PCT:
        print(f"✅ Target {TARGET_REACH_PCT}% achievable with both strategies")
    else:
        better_strategy = "Short-Format" if max_reach_sf > max_reach_lf else "Long-Format"
        print(f"✅ Target {TARGET_REACH_PCT}% achievable with {better_strategy} strategy")
    print()
    
    # Optimize both strategies
    print("--- OPTIMIZING STRATEGIC ALTERNATIVES ---")
    
    # Long-Format Focused Plan
    print("Optimizing Long-Format Focused Plan...")
    lf_spend, lf_result = optimize_media_plan_hybrid(
        TARGET_REACH_PCT, TOTAL_BUDGET, LONG_FORMAT_CONSTRAINTS, use_sainsbury=True
    )
    
    # Short-Format Focused Plan
    print("Optimizing Short-Format Focused Plan...")
    sf_spend, sf_result = optimize_media_plan_hybrid(
        TARGET_REACH_PCT, TOTAL_BUDGET, SHORT_FORMAT_CONSTRAINTS, use_sainsbury=True
    )
    
    if lf_spend and sf_spend:
        # Calculate performance for both plans
        _, lf_kpis = plan_kpis_hybrid(lf_spend, use_sainsbury=True)
        _, sf_kpis = plan_kpis_hybrid(sf_spend, use_sainsbury=True)
        
        print()
        print("--- STRATEGIC MEDIA PLAN COMPARISON ---")
        print()
        
        # Long-Format Plan Details
        print("📺 LONG-FORMAT FOCUSED PLAN (Brand Building & Storytelling)")
        print("-" * 60)
        lf_total = sum(lf_spend.values())
        lf_long_total = lf_spend['TV_LF'] + lf_spend['YT_LF']
        lf_short_total = lf_spend['YT_SF'] + lf_spend['Meta_SF'] + lf_spend['TT_SF']
        
        for channel, spend in lf_spend.items():
            pct = spend / lf_total * 100
            format_type = "LF" if channel.endswith('_LF') else "SF"
            print(f"{channel:>10}: ₺{spend:>10,.0f} ({pct:>5.1f}%) [{format_type}]")
        print(f"{'TOTAL':>10}: ₺{lf_total:>10,.0f}")
        print(f"Long-Format Share: {lf_long_total/lf_total:.1%} | Short-Format Share: {lf_short_total/lf_total:.1%}")
        print()
        
        # Long-Format Incremental Analysis
        display_incremental_analysis(lf_spend, "Long-Format Focused", use_sainsbury=True)
        
        # Short-Format Plan Details  
        print("📱 SHORT-FORMAT FOCUSED PLAN (Reach Maximization & Efficiency)")
        print("-" * 60)
        sf_total = sum(sf_spend.values())
        sf_long_total = sf_spend['TV_LF'] + sf_spend['YT_LF']
        sf_short_total = sf_spend['YT_SF'] + sf_spend['Meta_SF'] + sf_spend['TT_SF']
        
        for channel, spend in sf_spend.items():
            pct = spend / sf_total * 100
            format_type = "LF" if channel.endswith('_LF') else "SF"
            print(f"{channel:>10}: ₺{spend:>10,.0f} ({pct:>5.1f}%) [{format_type}]")
        print(f"{'TOTAL':>10}: ₺{sf_total:>10,.0f}")
        print(f"Long-Format Share: {sf_long_total/sf_total:.1%} | Short-Format Share: {sf_short_total/sf_total:.1%}")
        print()
        
        # Short-Format Incremental Analysis
        display_incremental_analysis(sf_spend, "Short-Format Focused", use_sainsbury=True)
        
        # Performance Comparison
        print("--- PERFORMANCE COMPARISON ---")
        print(f"{'Metric':<25} {'Long-Format':<15} {'Short-Format':<15} {'Advantage':<12}")
        print("-" * 70)
        
        # Net Reach
        lf_net = lf_kpis['NetReach_%_of_TAM']
        sf_net = sf_kpis['NetReach_%_of_TAM']
        net_advantage = "Short-Format" if sf_net > lf_net else "Long-Format" if lf_net > sf_net else "Equal"
        print(f"{'Net Reach':<25} {lf_net:<15.1f}% {sf_net:<15.1f}% {net_advantage:<12}")
        
        # Quality Reach
        lf_qual = lf_kpis['SainsburyQuality_%_TAM']
        sf_qual = sf_kpis['SainsburyQuality_%_TAM']
        qual_advantage = "Long-Format" if lf_qual > sf_qual else "Short-Format" if sf_qual > lf_qual else "Equal"
        print(f"{'Quality Reach':<25} {lf_qual:<15.1f}% {sf_qual:<15.1f}% {qual_advantage:<12}")
        
        # Budget Efficiency (Reach per ₺1M)
        lf_efficiency = lf_net / (lf_total / 1_000_000)
        sf_efficiency = sf_net / (sf_total / 1_000_000)
        eff_advantage = "Short-Format" if sf_efficiency > lf_efficiency else "Long-Format" if lf_efficiency > sf_efficiency else "Equal"
        print(f"{'Reach per ₺1M':<25} {lf_efficiency:<15.1f}% {sf_efficiency:<15.1f}% {eff_advantage:<12}")
        
        # Quality Efficiency (Quality per ₺1M)
        lf_qual_eff = lf_qual / (lf_total / 1_000_000)
        sf_qual_eff = sf_qual / (sf_total / 1_000_000)
        qual_eff_advantage = "Long-Format" if lf_qual_eff > sf_qual_eff else "Short-Format" if sf_qual_eff > lf_qual_eff else "Equal"
        print(f"{'Quality per ₺1M':<25} {lf_qual_eff:<15.1f}% {sf_qual_eff:<15.1f}% {qual_eff_advantage:<12}")
        
        print()
        
        # Strategic Insights
        print("--- STRATEGIC INSIGHTS ---")
        print()
        print("🎯 LONG-FORMAT FOCUSED PLAN - Best for:")
        print("   • Brand building and storytelling campaigns")
        print("   • Premium product launches")
        print("   • Building brand awareness and consideration")
        print("   • Maximizing engagement quality and watch time")
        print("   • Campaigns requiring detailed product demonstration")
        print()
        print("🚀 SHORT-FORMAT FOCUSED PLAN - Best for:")
        print("   • Performance marketing and direct response")
        print("   • Maximizing reach and frequency")
        print("   • Cost-efficient awareness campaigns")
        print("   • Targeting younger demographics")
        print("   • Quick activation and impulse purchase campaigns")
        print()
        
        # Recommendation
        if lf_qual > sf_qual * 1.2:  # Long-format has significantly higher quality
            print("💡 RECOMMENDATION: Long-Format plan for brand building campaigns")
        elif sf_net > lf_net * 1.1:  # Short-format has significantly higher reach
            print("💡 RECOMMENDATION: Short-Format plan for reach maximization")
        else:
            print("💡 RECOMMENDATION: Both strategies viable - choose based on campaign objectives")
        
        return {
            'long_format': {'spend': lf_spend, 'kpis': lf_kpis},
            'short_format': {'spend': sf_spend, 'kpis': sf_kpis},
            'comparison': {
                'reach_advantage': net_advantage,
                'quality_advantage': qual_advantage,
                'efficiency_advantage': eff_advantage
            }
        }
    else:
        print("❌ Strategic optimization failed")
        return None

# =========  INCREMENTAL REACH ANALYSIS  ======================

def calculate_incremental_contributions(spend_dict, use_sainsbury=True):
    """
    Calculate incremental reach and quality reach contribution of each channel.
    
    Args:
        spend_dict: Media spend allocation
        use_sainsbury: Whether to use Sainsbury+GWI hybrid method
        
    Returns:
        Dict with incremental contributions for each channel
        
    Note: Incremental reach represents the reach lost when removing a channel
    while keeping all other channels unchanged (not reallocating budget).
    """
    
    # Calculate baseline performance with all channels
    _, baseline_kpis = plan_kpis_hybrid(spend_dict, use_sainsbury=use_sainsbury)
    baseline_reach = baseline_kpis['NetReach_%_of_TAM']
    baseline_quality = baseline_kpis['SainsburyQuality_%_TAM'] if use_sainsbury else baseline_kpis['QualityReach_%_TAM']
    
    incremental_contributions = {}
    
    for channel in spend_dict.keys():
        if spend_dict[channel] > 0:  # Only analyze channels with spend
            
            # Create scenario without this channel
            scenario_spend = spend_dict.copy()
            scenario_spend[channel] = 0
            
            # Calculate performance without this channel
            _, scenario_kpis = plan_kpis_hybrid(scenario_spend, use_sainsbury=use_sainsbury)
            scenario_reach = scenario_kpis['NetReach_%_of_TAM']
            scenario_quality = scenario_kpis['SainsburyQuality_%_TAM'] if use_sainsbury else scenario_kpis['QualityReach_%_TAM']
            
            # Calculate incremental contributions
            incremental_reach = baseline_reach - scenario_reach
            incremental_quality = baseline_quality - scenario_quality
            
            # VALIDATION: Incremental reach should never be negative in media planning
            # If we get negative values, it indicates an artifact from complex quality weighting
            if incremental_reach < -0.001:  # Allow tiny numerical errors
                print(f"⚠️  Warning: Negative incremental reach detected for {channel} ({incremental_reach:.3f}pp)")
                print(f"   This suggests a mathematical artifact from quality weighting interactions.")
                print(f"   Setting incremental reach to 0 for realistic media planning.")
                incremental_reach = 0
            
            if incremental_quality < -0.001:  # Allow tiny numerical errors
                print(f"⚠️  Warning: Negative incremental quality detected for {channel} ({incremental_quality:.3f}pp)")
                print(f"   Setting incremental quality to 0 for realistic media planning.")
                incremental_quality = 0
            
            # Ensure non-negative values for calculations
            incremental_reach = max(0, incremental_reach)
            incremental_quality = max(0, incremental_quality)
            
            # Calculate efficiency metrics
            channel_spend = spend_dict[channel]
            incremental_reach_per_million = incremental_reach / (channel_spend / 1_000_000) if channel_spend > 0 else 0
            incremental_quality_per_million = incremental_quality / (channel_spend / 1_000_000) if channel_spend > 0 else 0
            
            incremental_contributions[channel] = {
                'spend': channel_spend,
                'spend_pct': channel_spend / sum(spend_dict.values()) * 100,
                'incremental_reach_pp': incremental_reach,
                'incremental_quality_pp': incremental_quality,
                'incremental_reach_per_1m': incremental_reach_per_million,
                'incremental_quality_per_1m': incremental_quality_per_million,
                'reach_efficiency_rank': 0,  # Will be filled later
                'quality_efficiency_rank': 0  # Will be filled later
            }
    
    # Calculate efficiency rankings
    channels_by_reach_eff = sorted(incremental_contributions.items(), 
                                  key=lambda x: x[1]['incremental_reach_per_1m'], reverse=True)
    channels_by_quality_eff = sorted(incremental_contributions.items(), 
                                    key=lambda x: x[1]['incremental_quality_per_1m'], reverse=True)
    
    for rank, (channel, _) in enumerate(channels_by_reach_eff, 1):
        incremental_contributions[channel]['reach_efficiency_rank'] = rank
        
    for rank, (channel, _) in enumerate(channels_by_quality_eff, 1):
        incremental_contributions[channel]['quality_efficiency_rank'] = rank
    
    return incremental_contributions

def display_incremental_analysis(spend_dict, plan_name, use_sainsbury=True):
    """
    Display incremental contribution analysis for a media plan.
    
    Args:
        spend_dict: Media spend allocation
        plan_name: Name of the plan (e.g., "Long-Format Focused")
        use_sainsbury: Whether to use Sainsbury+GWI method
    """
    
    incremental_data = calculate_incremental_contributions(spend_dict, use_sainsbury)
    
    if not incremental_data:
        return
    
    print(f"--- INCREMENTAL CONTRIBUTION ANALYSIS: {plan_name.upper()} ---")
    print("📊 This analysis shows the reach lost when removing each channel individually")
    print("   (without reallocating budget to other channels)")
    print()
    print(f"{'Channel':<10} {'Spend':<12} {'%':<6} {'Inc.Reach':<10} {'Inc.Quality':<12} {'Reach/₺1M':<10} {'Qual/₺1M':<10} {'R.Rank':<7} {'Q.Rank':<7}")
    print("-" * 88)
    
    # Sort by incremental reach contribution (highest first)
    sorted_channels = sorted(incremental_data.items(), 
                           key=lambda x: x[1]['incremental_reach_pp'], reverse=True)
    
    total_incremental_reach = 0
    total_incremental_quality = 0
    
    for channel, data in sorted_channels:
        if data['spend'] > 0:
            format_type = "[LF]" if channel.endswith('_LF') else "[SF]"
            
            print(f"{channel:<10} ₺{data['spend']:>9,.0f} {data['spend_pct']:>5.1f}% "
                  f"{data['incremental_reach_pp']:>9.2f}pp {data['incremental_quality_pp']:>11.2f}pp "
                  f"{data['incremental_reach_per_1m']:>9.1f}% {data['incremental_quality_per_1m']:>9.1f}% "
                  f"{data['reach_efficiency_rank']:>6} {data['quality_efficiency_rank']:>6} {format_type}")
            
            total_incremental_reach += data['incremental_reach_pp']
            total_incremental_quality += data['incremental_quality_pp']
    
    print("-" * 88)
    print(f"{'TOTAL':<10} {'':>12} {'':>6} {total_incremental_reach:>9.2f}pp {total_incremental_quality:>11.2f}pp")
    print()
    
    # Summary insights
    best_reach_channel = max(incremental_data.items(), key=lambda x: x[1]['incremental_reach_pp'])
    best_quality_channel = max(incremental_data.items(), key=lambda x: x[1]['incremental_quality_pp'])
    most_efficient_reach = max(incremental_data.items(), key=lambda x: x[1]['incremental_reach_per_1m'])
    most_efficient_quality = max(incremental_data.items(), key=lambda x: x[1]['incremental_quality_per_1m'])
    
    print(f"🎯 KEY INSIGHTS:")
    print(f"   • Highest Reach Contributor: {best_reach_channel[0]} (+{best_reach_channel[1]['incremental_reach_pp']:.2f}pp)")
    print(f"   • Highest Quality Contributor: {best_quality_channel[0]} (+{best_quality_channel[1]['incremental_quality_pp']:.2f}pp)")
    print(f"   • Most Efficient Reach: {most_efficient_reach[0]} ({most_efficient_reach[1]['incremental_reach_per_1m']:.1f}% per ₺1M)")
    print(f"   • Most Efficient Quality: {most_efficient_quality[0]} ({most_efficient_quality[1]['incremental_quality_per_1m']:.1f}% per ₺1M)")
    print()
    print("📝 NOTE: Incremental values represent reach/quality LOST when removing each channel")
    print("   Higher values = more important channels. Zero values may indicate quality weighting artifacts.")
    print()

# =========  DATA INTEGRATION UTILITIES  ======================

def integrate_real_campaign_data(data_dict):
    """
    Integrate real campaign data from spreadsheet into the script.
    
    Expected data_dict structure (based on user's screenshot):
    {
        'segments': ['F18-24', 'F25-34', 'F35-44', 'F45-54', 'F55+', 
                    'M18-24', 'M25-34', 'M35-44', 'M45-54', 'M55+'],
        'platforms': {
            'TV_Long': {
                'size': [...],      # Audience size per segment
                'cost': [...],      # Total cost per segment  
                'cpm': [...],       # CPM per segment
                'impressions': [...], # Impressions per segment
                'frequency': [...],  # Frequency per segment
                'vat': [...],       # Video Average Watch Time per segment
            },
            'YouTube_Long': {...},   # Same audience pool as YouTube_Short
            'YouTube_Short': {...},  # Same audience pool as YouTube_Long  
            'Meta_Long': {...},      # Same audience pool as Meta_Short
            'Meta_Short': {...},     # Same audience pool as Meta_Long
            'TikTok_Long': {...},    # Same audience pool as TikTok_Short
            'TikTok_Short': {...},   # Same audience pool as TikTok_Long
        }
    }
    """
    
    # Update segment populations from your data
    global segments, cpm, aud_prof, w
    
    segments_updated = {}
    cpm_updated = {}
    aud_prof_updated = {}
    vat_data = {}
    
    # Format mapping
    format_map = {
        'TV_Long': 'TV_LF',
        'YouTube_Long': 'YT_LF', 
        'YouTube_Short': 'YT_SF',
        'Meta_Long': 'Meta_SF',  # Meta only has short format now
        'Meta_Short': 'Meta_SF',
        'TikTok_Long': 'TT_SF',  # TikTok only has short format now
        'TikTok_Short': 'TT_SF'
    }
    
    # CORRECTED TAM LOGIC: Calculate platform totals first
    platform_totals = {}
    for platform, data in data_dict['platforms'].items():
        if 'size' in data:
            platform_base = platform.split('_')[0]  # 'YouTube', 'Meta', 'TikTok', 'TV'
            if platform_base not in platform_totals:
                platform_totals[platform_base] = sum(data['size'])
    
    # TAM should be the largest platform audience (maximum addressable market)
    tam_value = max(platform_totals.values()) if platform_totals else 50_000_000
    largest_platform = max(platform_totals, key=platform_totals.get) if platform_totals else 'Meta'
    
    print(f"🎯 TAM Logic: Using {largest_platform} audience as TAM = {tam_value:,} people")
    print(f"   Platform totals: {platform_totals}")
    
    # Use the largest platform's segment distribution as the market structure
    largest_platform_data = None
    for platform, data in data_dict['platforms'].items():
        if platform.startswith(largest_platform) and 'size' in data:
            largest_platform_data = data['size']
            break
    
    # Create segment populations based on largest platform's structure
    if largest_platform_data:
        for i, segment in enumerate(data_dict['segments']):
            if i < len(largest_platform_data):
                segments_updated[segment] = largest_platform_data[i]
            else:
                segments_updated[segment] = 1_000_000  # Fallback
    else:
        # Fallback to equal distribution
        segment_size = tam_value // len(data_dict['segments'])
        for segment in data_dict['segments']:
            segments_updated[segment] = segment_size
    
    # Extract platform-specific data
    platform_audiences = {}  # Store unique audience per platform
    
    for platform, data in data_dict['platforms'].items():
        if platform in format_map:
            format_code = format_map[platform]
            
            # Calculate average CPM across segments (with error handling)
            if 'cpm' in data and len(data['cpm']) > 0:
                valid_cpms = [x for x in data['cpm'] if x > 0]
                if valid_cpms:
                    avg_cpm = sum(valid_cpms) / len(valid_cpms)
                    cpm_updated[format_code] = avg_cpm
            
            # Extract VAT data for quality weights (with error handling)
            if 'vat' in data and len(data['vat']) > 0:
                valid_vats = [x for x in data['vat'] if x > 0]
                if valid_vats:
                    avg_vat = sum(valid_vats) / len(valid_vats)
                    vat_data[format_code] = avg_vat
            
            # Extract platform audience (avoiding double counting)
            platform_base = platform.split('_')[0]  # 'YouTube', 'Meta', 'TikTok', 'TV'
            if platform_base not in platform_audiences:
                platform_audiences[platform_base] = data['size'].copy()
    
    # Calculate audience profiles from impression distribution
    for platform, data in data_dict['platforms'].items():
        if platform in format_map:
            format_code = format_map[platform]
            
            if 'impressions' in data and len(data['impressions']) > 0:
                valid_impressions = [x for x in data['impressions'] if x > 0]
                total_impressions = sum(valid_impressions)
                
                if total_impressions > 0:
                    aud_prof_updated[format_code] = {}
                    for i, segment in enumerate(data_dict['segments']):
                        if i < len(data['impressions']) and data['impressions'][i] > 0:
                            aud_prof_updated[format_code][segment] = data['impressions'][i] / total_impressions
                        else:
                            aud_prof_updated[format_code][segment] = 0.1  # Small fallback value
    
    # Ensure long and short formats on same platform have identical audience profiles
    # (since they target the same audience pool)
    for platform_base in ['YouTube', 'Meta', 'TikTok']:
        lf_key = f"{platform_base.upper().replace('YOUTUBE', 'YT').replace('TIKTOK', 'TT')}_LF"
        sf_key = f"{platform_base.upper().replace('YOUTUBE', 'YT').replace('TIKTOK', 'TT')}_SF"
        
        if lf_key in aud_prof_updated and sf_key in aud_prof_updated:
            # Use the long format profile for both (they should be identical anyway)
            aud_prof_updated[sf_key] = aud_prof_updated[lf_key].copy()
    
    # Calculate enhanced quality weights from VAT data
    quality_weights_updated = {}
    if vat_data:
        quality_weights_updated = calculate_enhanced_quality_weights(vat_data)
    
    return segments_updated, cpm_updated, aud_prof_updated, quality_weights_updated, platform_audiences

def calculate_reach_curves_from_data(data_dict):
    """
    Estimate reach curve parameters from actual campaign performance.
    Uses GRP vs Reach relationship to fit exponential curves.
    """
    reach_curves = {}
    
    format_map = {
        'TV_Long': 'TV_LF',
        'YouTube_Long': 'YT_LF', 
        'YouTube_Short': 'YT_SF',
        'Meta_Long': 'Meta_SF',  # Meta only has short format now
        'Meta_Short': 'Meta_SF',
        'TikTok_Long': 'TT_SF',  # TikTok only has short format now
        'TikTok_Short': 'TT_SF'
    }
    
    for platform, data in data_dict['platforms'].items():
        if platform in format_map:
            format_code = format_map[platform]
            
            # Calculate average metrics across segments
            if 'frequency' in data and 'impressions' in data and 'size' in data:
                avg_frequency = sum(data['frequency']) / len(data['frequency']) if data['frequency'] else 1.5
                total_impressions = sum(data['impressions'])
                total_population = sum(data['size'])
                
                # Estimate GRPs and Reach
                if total_population > 0:
                    grps = (total_impressions / 1000) / (total_population / 1000)  # Simplified GRP calc
                    estimated_reach = min(0.95, total_impressions / (avg_frequency * total_population))
                    
                    # Fit exponential curve: R = Max * (1 - e^(-k*GRP))
                    # Estimate Max reach (platform ceiling)
                    if 'TV' in platform:
                        max_reach = 0.90
                    elif 'YouTube' in platform:
                        max_reach = 0.75 if 'Long' in platform else 0.70
                    elif 'TikTok' in platform:
                        max_reach = 0.55
                    else:  # Meta
                        max_reach = 0.70
                    
                    # Estimate k parameter
                    if estimated_reach > 0 and grps > 0:
                        # Solve for k: R = Max * (1 - e^(-k*GRP))
                        # k = -ln(1 - R/Max) / GRP
                        reach_ratio = min(0.99, estimated_reach / max_reach)
                        if reach_ratio > 0:
                            k = -np.log(1 - reach_ratio) / grps
                            reach_curves[format_code] = (max_reach, max(0.1, min(0.5, k)))
                        else:
                            reach_curves[format_code] = (max_reach, 0.2)  # Default
                    else:
                        reach_curves[format_code] = (max_reach, 0.2)  # Default
                else:
                    # Fallback to defaults
                    reach_curves[format_code] = reach_par.get(format_code, (0.5, 0.2))
            else:
                # Fallback to defaults if data missing
                reach_curves[format_code] = reach_par.get(format_code, (0.5, 0.2))
    
    return reach_curves

def extract_real_campaign_data_from_screenshot():
    """
    Extract real campaign data based on the user's complete data table.
    This contains 5 months of actual campaign performance data.
    """
    
    real_campaign_data = {
        'segments': ['F18_24', 'F25_34', 'F35_44', 'F45_54', 'F55_plus', 
                    'M18_24', 'M25_34', 'M35_44', 'M45_54', 'M55_plus'],
        'platforms': {
            # TV Long Format - COMPLETE REAL DATA
            'TV_Long': {
                'size': [4890500, 4939000, 4994000, 3926000, 5446500,
                        4890500, 4939000, 4994000, 3926000, 5446500],
                'cost': [400000.00, 400000.00, 400000.00, 400000.00, 400000.00,
                        400000.00, 400000.00, 400000.00, 400000.00, 400000.00],
                'cpm': [80, 80, 80, 80, 80,
                       80, 80, 80, 80, 80],
                'impressions': [815083, 823167, 832333, 654333, 907750,
                               815083, 823167, 832333, 654333, 907750],
                'frequency': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'vat': [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
            },
            
            # YouTube Long Format - COMPLETE REAL DATA
            'YouTube_Long': {
                'size': [3740000, 5760000, 5730000, 4540000, 4010000, 
                        4050000, 6090000, 5870000, 4930000, 4420000],
                'cost': [198982.09, 248644.18, 411934.50, 540.97, 393451.15,
                        215350.84, 165828.54, 93578.75, 81958.06, 77813.15],
                'cpm': [58, 58, 58, 58, 58,
                       58, 58, 58, 58, 58],
                'impressions': [2195402.00, 8089746.00, 7640018.00, 2604568.00, 1514668.00,
                               27639794.00, 20871808.00, 11831313.00, 10695488.00, 9823118.00],
                'frequency': [1.26, 1.24, 1.26, 1.27, 1.30,
                             1.26, 1.27, 1.28, 1.25, 1.29],
                'vat': [6.83, 4.97, 5.55, 4.89, 6.21,
                       4.23, 5.06, 4.60, 3.70, 5.36]
            },
            # YouTube Short Format - EXCLUDED
            
            # Meta Long Format - COMPLETE REAL DATA
            'Meta_Long': {
                'size': [6800000, 8600000, 5900000, 4000000, 3800000,
                        7900000, 11500000, 7400000, 5200000, 5000000],
                'cost': [164505.50, 115791.40, 75475.81, 61056.89, 59260.67,
                        215350.84, 165828.54, 93578.75, 81958.06, 77813.15],
                'cpm': [46.29, 53.17, 50.47, 51.23, 52.58,
                       49.05, 54.33, 44.51, 37.71, 43.60],
                'impressions': [21612145.00, 13386324.00, 8193293.00, 6590605.00, 5994265.00,
                               27639794.00, 20871808.00, 11831313.00, 10695488.00, 9823118.00],
                'frequency': [1.26, 1.24, 1.26, 1.27, 1.30,
                             1.26, 1.27, 1.28, 1.25, 1.29],
                'vat': [6.83, 4.97, 5.55, 4.89, 6.21,
                       4.23, 5.06, 4.60, 3.70, 5.36]
            },
            
            # Meta Short Format - COMPLETE REAL DATA
            'Meta_Short': {
                'size': [6800000, 8600000, 5900000, 4000000, 3800000,  # SAME as Meta_Long
                        7900000, 11500000, 7400000, 5200000, 5000000],
                'cost': [157008.42, 180766.10, 203921.17, 113775.55, 212692.07,
                        202572.01, 319972.72, 274788.45, 134284.54, 295668.11],
                'cpm': [23.80, 29.76, 27.85, 29.85, 41.31,
                       29.90, 33.01, 25.22, 30.81, 34.87],
                'impressions': [17429833.00, 15248519.00, 13698444.00, 8141000.00, 10554076.00,
                               22584839.00, 27821758.00, 20626222.00, 10418823.00, 14427717.00],
                'frequency': [1.29, 1.35, 1.35, 1.24, 1.30,
                             1.31, 1.37, 1.34, 1.26, 1.36],
                'vat': [2.13, 2.56, 2.87, 2.76, 3.21,
                       2.06, 2.34, 2.35, 2.53, 2.66]
            },
            
            # TikTok Long Format - COMPLETE REAL DATA  
            'TikTok_Long': {
                'size': [5600000, 6800000, 1820000, 983000, 939000,
                        5900000, 10500000, 3400000, 1920000, 1400000],
                'cost': [729381.16, 509654.58, 193663.65, 185398.76, 79596.90,
                        732334.64, 1804282.45, 415113.37, 184691.55, 112107.61],
                'cpm': [19.27, 19.52, 24.95, 16.12, 14.85,
                       16.10, 15.95, 18.78, 16.59, 14.85],
                'impressions': [69169491, 45903934, 17453470, 9671511, 10726175,
                               61555768, 65240401, 34518189, 19527884, 13711639],
                'frequency': [1.63, 1.61, 1.65, 2.17, 1.69,
                             1.62, 1.61, 1.61, 1.66, 1.68],
                'vat': [2.57, 2.71, 2.77, 3.23, 2.93,
                       2.18, 2.16, 2.44, 3.36, 2.93]
            },
            
            # TikTok Short Format - COMPLETE REAL DATA
            'TikTok_Short': {
                'size': [5600000, 6800000, 1820000, 983000, 939000,  # SAME as TikTok_Long
                        5900000, 10500000, 3400000, 1920000, 1400000],
                'cost': [150826.64, 114132.78, 400213.55, 147216.34, 299528.22,
                        1147194.10, 1126135.42, 167496.23, 488194.94, 134565.70],
                'cpm': [10.54, 11.64, 11.42, 11.62, 10.97,
                       11.09, 11.04, 12.83, 12.98, 11.83],
                'impressions': [13764333, 9464003, 3567114, 2919568, 2767473,
                               10575425, 10659315, 5513638, 4254940, 3315070],
                'frequency': [2.39, 2.34, 2.50, 2.17, 2.06,
                             2.30, 2.06, 2.19, 2.08, 1.76],
                'vat': [3.59, 3.65, 3.75, 4.73, 4.07,
                       3.39, 3.47, 3.84, 4.25, 4.01]
            }
        }
    }
    
    return real_campaign_data

def apply_real_data_and_run_hybrid_comparison():
    """Apply the real campaign data and run strategic media planning alternatives."""
    print("="*70)
    print("🚀 STRATEGIC MEDIA PLANNING WITH REAL CAMPAIGN DATA")
    print("="*70)
    print()
    
    # Extract real data
    real_data = extract_real_campaign_data_from_screenshot()
    
    # Apply data integration
    try:
        global segments, cpm, aud_prof, w, reach_par
        
        new_segments, new_cpm, new_profiles, new_quality_weights, platform_audiences = integrate_real_campaign_data(real_data)
        new_reach_curves = calculate_reach_curves_from_data(real_data)
        
        # Update global variables
        segments.update(new_segments)
        cpm.update(new_cpm)
        aud_prof.update(new_profiles)
        w.update(new_quality_weights)
        reach_par.update(new_reach_curves)
        
        # CRITICAL: Update TAM with correct platform audience sizes from demographic segment table
        global TAM, platform_totals
        # Use the current platform totals from Excel file (or fallback values)
        TAM = max(platform_totals.values())  # Use the largest platform as TAM
        
        print("✅ Real campaign data successfully integrated!")
        print(f"📊 Total Addressable Market: {TAM:,} people")
        print(f"🎯 Segments: {len(segments)} demographic groups")
        print(f"📺 Platforms: {len(set([p.split('_')[0] for p in real_data['platforms'].keys()]))} platforms")
        print(f"🎬 Formats: {len(real_data['platforms'])} ad formats")
        print()
        
        # Show platform audiences
        print("--- PLATFORM AUDIENCE SIZES ---")
        for platform, total in platform_totals.items():
            print(f"{platform:>10}: {total:>12,} people")
        print()
        
        # Run strategic media planning with real data
        run_strategic_media_planning()
        
    except Exception as e:
        print(f"❌ Error applying real data: {e}")
        print("Please check your real campaign data and try again.")

def apply_real_data_and_run_optimization():
    """Original function maintained for backward compatibility."""
    print("="*70)
    print("🚀 APPLYING REAL CAMPAIGN DATA FROM SCREENSHOT")
    print("="*70)
    print()
    
    # Extract real data
    real_data = extract_real_campaign_data_from_screenshot()
    
    # Apply data integration
    try:
        global segments, cpm, aud_prof, w, reach_par
        
        new_segments, new_cpm, new_profiles, new_quality_weights, platform_audiences = integrate_real_campaign_data(real_data)
        new_reach_curves = calculate_reach_curves_from_data(real_data)
        
        # Update global variables
        segments.update(new_segments)
        cpm.update(new_cpm)
        aud_prof.update(new_profiles)
        w.update(new_quality_weights)
        reach_par.update(new_reach_curves)
        
        # CRITICAL: Update TAM with correct platform audience sizes from demographic segment table
        global TAM, platform_totals
        # Use the current platform totals from Excel file (or fallback values)
        TAM = max(platform_totals.values())  # Use the largest platform as TAM
        
        print("✅ Real campaign data successfully integrated!")
        print(f"📊 Total Addressable Market: {TAM:,} people")
        print(f"🎯 Segments: {len(segments)} demographic groups")
        print(f"📺 Platforms: {len(set([p.split('_')[0] for p in real_data['platforms'].keys()]))} platforms")
        print(f"🎬 Formats: {len(real_data['platforms'])} ad formats")
        print()
        
        # Show platform audiences
        print("--- PLATFORM AUDIENCE SIZES ---")
        for platform, total in platform_totals.items():
            print(f"{platform:>10}: {total:>12,} people")
        print()
        
        # Run optimization with real data
        run_media_planning()
        
    except Exception as e:
        print(f"❌ Error applying real data: {e}")
        print("Please check your real campaign data and try again.")

# =========  RUN ANALYSIS  ====================================

if __name__ == "__main__":
    print("="*70)
    print("🎯 STRATEGIC MEDIA PLANNING - REAL CAMPAIGN DATA VERSION")
    print("="*70)
    print()
    
    # Run real data extraction with strategic media plan alternatives
    apply_real_data_and_run_hybrid_comparison()
    
# =========  FREQUENCY OPTIMIZATION  ================================

def calculate_channel_frequency(spend, channel, segment_populations):
    """
    Calculate average frequency for a channel across all segments.
    
    Frequency = Total Impressions / Total People Reached
    
    Args:
        spend: Media spend for the channel
        channel: Channel identifier (e.g., 'TV_LF', 'YT_SF')
        segment_populations: Dictionary of segment populations
    
    Returns:
        float: Average frequency across all segments
    """
    if spend <= 0:
        return 0
    
    channel_cpm = cpm.get(channel, 10)
    if channel_cpm <= 0:
        return 0
    
    # Total impressions delivered by this spend
    total_impressions = (spend / channel_cpm) * 1000
    
    # Calculate total people reached across all segments
    total_reach_people = 0
    
    for segment, population in segment_populations.items():
        if segment in aud_prof.get(channel, {}):
            # Get audience profile weight for this segment
            segment_weight = aud_prof[channel][segment]
            
            # Calculate addressable population for this segment
            segment_addressable_pop = segment_weight * population
            
            if segment_addressable_pop > 0:
                # Calculate GRP for this segment
                # GRP = (Impressions / 1000) / (Population / 1000) = Impressions / Population
                segment_impressions = total_impressions * segment_weight  # Proportional impressions
                segment_grp = segment_impressions / segment_addressable_pop
                
                # Calculate reach percentage using reach curve
                max_reach, k = reach_par.get(channel, (0.5, 0.2))
                segment_reach_pct = max_reach * (1 - np.exp(-k * segment_grp))
                
                # Convert to people reached
                segment_reach_people = segment_reach_pct * segment_addressable_pop
                total_reach_people += segment_reach_people
    
    # Calculate frequency as impressions / people reached
    if total_reach_people > 0:
        return total_impressions / total_reach_people
    else:
        return 0

def calculate_frequency_penalty(spend_dict, max_frequency=6.0, penalty_weight=10.0):
    """
    Calculate penalty for channels exceeding maximum frequency.
    
    Args:
        spend_dict: Media spend allocation
        max_frequency: Maximum acceptable frequency (default: 6.0)
        penalty_weight: Weight for frequency penalty (default: 10.0)
    
    Returns:
        float: Frequency penalty score
    """
    total_penalty = 0
    
    for channel, spend in spend_dict.items():
        if spend > 0:
            frequency = calculate_channel_frequency(spend, channel, segments)
            
            if frequency > max_frequency:
                # Much more aggressive penalty that strongly discourages high frequencies
                excess_frequency = frequency - max_frequency
                
                # Exponential penalty that gets very expensive for high frequencies
                # Base penalty scales with excess frequency squared
                base_penalty = penalty_weight * 2.0 * (excess_frequency ** 2)
                
                # Additional exponential penalty for extreme frequencies
                if excess_frequency > 5.0:
                    extreme_penalty = penalty_weight * 10.0 * ((excess_frequency - 5.0) ** 3)
                    base_penalty += extreme_penalty
                
                total_penalty += base_penalty
    
    return total_penalty

def objective_function_with_frequency(spend_array, target_reach_pct, budget, use_sainsbury=True, 
                                    max_frequency=6.0, frequency_weight=10.0):
    """
    Enhanced objective function with frequency optimization.
    
    Args:
        spend_array: Array of spend values
        target_reach_pct: Target reach percentage
        budget: Total budget
        use_sainsbury: Whether to use Sainsbury method
        max_frequency: Maximum acceptable frequency
        frequency_weight: Weight for frequency penalty
    
    Returns:
        float: Objective function value (lower is better)
    """
    spend_dict = spend_array_to_dict(spend_array)
    
    # Calculate current reach using hybrid method
    _, kpis = plan_kpis_hybrid(spend_dict, use_sainsbury=use_sainsbury)
    current_reach_pct = kpis['NetReach_%_of_TAM']
    
    # Calculate total spend
    total_spend = sum(spend_array)
    
    # Primary penalty: reach gap
    reach_gap = abs(current_reach_pct - target_reach_pct)
    reach_penalty = reach_gap * 100
    
    # Budget constraint penalty (for exceeding budget)
    budget_penalty = max(0, total_spend - budget) * 0.01
    
    # Budget utilization incentive (encourage using full budget)
    budget_utilization = total_spend / budget
    if budget_utilization < 0.95:  # If using less than 95% of budget
        under_utilization_penalty = (0.95 - budget_utilization) * 30
    else:
        under_utilization_penalty = 0
    
    # Adaptive frequency penalty - reduce when far from reach target but maintain some control
    if reach_gap > 25:  # If more than 25pp away from target
        frequency_weight_adjusted = frequency_weight * 0.3  # Reduce but still maintain control
    elif reach_gap > 20:  # If more than 20pp away from target
        frequency_weight_adjusted = frequency_weight * 0.5  # Moderately reduce frequency penalty
    elif reach_gap > 15:  # If more than 15pp away from target
        frequency_weight_adjusted = frequency_weight * 0.7  # Slightly reduce frequency penalty
    elif reach_gap > 10:  # If more than 10pp away from target
        frequency_weight_adjusted = frequency_weight * 0.8  # Minimal reduction
    elif reach_gap > 5:   # If more than 5pp away from target
        frequency_weight_adjusted = frequency_weight * 0.9  # Very minimal reduction
    else:
        frequency_weight_adjusted = frequency_weight  # Normal frequency penalty
    
    frequency_penalty = calculate_frequency_penalty(spend_dict, max_frequency, frequency_weight_adjusted)
    
    # Prioritize reach when far from target
    if reach_gap > 15:
        reach_priority_multiplier = 2.0  # Strong reach priority
    elif reach_gap > 10:
        reach_priority_multiplier = 1.5  # Moderate reach priority
    else:
        reach_priority_multiplier = 1.0  # Normal priority
    
    return (reach_penalty * reach_priority_multiplier) + budget_penalty + under_utilization_penalty + frequency_penalty

def optimize_media_plan_with_frequency(target_reach_pct, budget, constraints=None, use_sainsbury=True, 
                                     max_frequency=6.0, frequency_weight=25.0, convergence_factor=0.85):
    """
    Enhanced media optimization with frequency constraints.
    
    Args:
        target_reach_pct: Target reach as % of TAM
        budget: Total budget available
        constraints: Dict of channel constraints (min/max % of budget)
        use_sainsbury: If True, use Sainsbury+GWI hybrid; if False, use original
        max_frequency: Maximum acceptable frequency per channel
        frequency_weight: Weight for frequency penalty in optimization
        convergence_factor: Sainsbury convergence parameter (0.8-0.9)
    
    Returns:
        dict: Optimized spend allocation with frequency constraints
    """
    if constraints is None:
        constraints = CHANNEL_CONSTRAINTS
    
    # Set up bounds
    bounds = create_optimization_bounds(budget, constraints)
    
    print(f"🎯 Optimizing with frequency cap: {max_frequency:.1f}")
    print(f"⚖️  Frequency penalty weight: {frequency_weight}")
    
    # Optimize using differential evolution with frequency constraints
    result = differential_evolution(
        objective_function_with_frequency,
        bounds,
        args=(target_reach_pct, budget, use_sainsbury, max_frequency, frequency_weight),
        seed=42,
        maxiter=800,  # Increased iterations for better reach optimization
        popsize=40,   # Increased population for better exploration
        atol=1e-4,    # Tighter tolerance for better solutions
        tol=1e-4,     # Tighter tolerance for better solutions
        polish=True,  
        disp=False    
    )
    
    if result.success:
        optimal_spend = spend_array_to_dict(result.x)
        
        # Double-check budget constraint
        total_spend = sum(optimal_spend.values())
        if total_spend > budget * 1.01:  # Allow 1% tolerance
            print(f"Warning: Budget exceeded by {total_spend - budget:,.0f}")
            # Scale down proportionally
            scale_factor = budget / total_spend
            optimal_spend = {k: v * scale_factor for k, v in optimal_spend.items()}
        
        # Display frequency results
        print("\n📊 FREQUENCY OPTIMIZATION RESULTS:")
        print(f"{'Channel':<10} {'Spend':<12} {'Frequency':<10} {'Status':<10}")
        print("-" * 45)
        
        for channel, spend in optimal_spend.items():
            if spend > 0:
                frequency = calculate_channel_frequency(spend, channel, segments)
                status = "✅ OK" if frequency <= max_frequency else "⚠️ HIGH"
                print(f"{channel:<10} ₺{spend:<11,.0f} {frequency:<10.1f} {status}")
        
        return optimal_spend, result
    else:
        print(f"Frequency optimization failed: {result.message}")
        print("Attempting with relaxed frequency constraints...")
        
        # Try with relaxed frequency constraints
        result_relaxed = differential_evolution(
            objective_function_with_frequency,
            bounds,
            args=(target_reach_pct, budget, use_sainsbury, max_frequency * 1.5, frequency_weight * 0.5),
            seed=123,     
            maxiter=400,  
            popsize=25,   
            atol=1e-2,    
            tol=1e-2,     
            polish=True,
            disp=False
        )
        
        if result_relaxed.success:
            optimal_spend = spend_array_to_dict(result_relaxed.x)
            total_spend = sum(optimal_spend.values())
            if total_spend > budget * 1.01:
                scale_factor = budget / total_spend
                optimal_spend = {k: v * scale_factor for k, v in optimal_spend.items()}
            print("✅ Frequency optimization succeeded with relaxed constraints")
            return optimal_spend, result_relaxed
        else:
            print(f"Both frequency optimization attempts failed: {result_relaxed.message}")
            return None, result

# =========  DEBUG FUNCTIONS  ============================

def debug_reach_calculation():
    """Debug function to understand why reach is stuck at 15.2%."""
    print("🔍 DEBUG: Reach Calculation Analysis")
    print("=" * 50)
    
    # Test with different spend allocations
    test_cases = [
        {"TV_LF": 1000000, "YT_LF": 1000000, "YT_SF": 1000000, "Meta_SF": 1000000, "TT_SF": 1000000},
        {"TV_LF": 5000000, "YT_LF": 500000, "YT_SF": 500000, "Meta_SF": 0, "TT_SF": 0},
        {"TV_LF": 0, "YT_LF": 0, "YT_SF": 2000000, "Meta_SF": 2000000, "TT_SF": 2000000},
        {"TV_LF": 3000000, "YT_LF": 3000000, "YT_SF": 0, "Meta_SF": 0, "TT_SF": 0},
    ]
    
    for i, spend_dict in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        for channel, spend in spend_dict.items():
            print(f"{channel}: ₺{spend:,}")
        
        # Calculate reach
        _, kpis = plan_kpis_hybrid(spend_dict, use_sainsbury=True)
        print(f"Net Reach: {kpis['NetReach_%_of_TAM']}% of TAM")
        print(f"Quality Reach: {kpis['SainsburyQuality_%_TAM']}% of TAM")
        
        # Show segment-level details for first case
        if i == 1:
            seg_rows, _ = plan_kpis_hybrid(spend_dict, use_sainsbury=True)
            print("\nSegment-level breakdown:")
            for row in seg_rows[:3]:  # Show first 3 segments
                print(f"  {row['Segment']}: {row['NetReach_%']:.1f}% reach")
    
    # Test with extreme allocations
    print(f"\n--- Extreme Test: All budget to TV ---")
    extreme_tv = {"TV_LF": 6000000, "YT_LF": 0, "YT_SF": 0, "Meta_SF": 0, "TT_SF": 0}
    _, kpis_tv = plan_kpis_hybrid(extreme_tv, use_sainsbury=True)
    print(f"TV Only: {kpis_tv['NetReach_%_of_TAM']}% of TAM")
    
    print(f"\n--- Extreme Test: All budget to Meta ---")
    extreme_meta = {"TV_LF": 0, "YT_LF": 0, "YT_SF": 0, "Meta_SF": 6000000, "TT_SF": 0}
    _, kpis_meta = plan_kpis_hybrid(extreme_meta, use_sainsbury=True)
    print(f"Meta Only: {kpis_meta['NetReach_%_of_TAM']}% of TAM")
    
    # Check if segments and TAM are correct
    print(f"\n--- Data Check ---")
    print(f"TAM: {TAM:,}")
    print(f"Segments total: {sum(segments.values()):,}")
    print(f"Segments: {len(segments)}")
    print(f"CPM data: {cpm}")
    
    return kpis_tv, kpis_meta

# =========  PLANNING WORKFLOW  ================================

