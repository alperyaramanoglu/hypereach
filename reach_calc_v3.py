#!/usr/bin/env python3
"""
HypeReach Media Planner v3.0 - Reach Calculation Engine
======================================================

Clean implementation with proper mathematical foundations
INCLUDES: Sainsbury weighted model for quality reach analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# Version tracking
REACH_CALC_VERSION = "v3.0_with_strategies"

# =========================================================================
# CORE DATA STRUCTURES
# =========================================================================

class MediaChannel:
    """Represents a single media channel with its properties"""
    def __init__(self, name: str, platform: str, format_type: str, 
                 cpm: float, max_reach: float, reach_curve_k: float):
        self.name = name
        self.platform = platform
        self.format_type = format_type  # 'LF' or 'SF'
        self.cpm = cpm
        self.max_reach = max_reach  # Maximum possible reach (0-1)
        self.reach_curve_k = reach_curve_k  # Reach curve steepness

class AudienceSegment:
    """Represents a demographic segment"""
    def __init__(self, name: str, population: int, channel_affinities: Dict[str, float]):
        self.name = name
        self.population = population
        self.channel_affinities = channel_affinities  # % of segment reachable by each channel

# =========================================================================
# REALISTIC MEDIA DATA
# =========================================================================

def load_realistic_media_data():
    """Load realistic media channel data based on Turkish market"""
    
    channels = {
        'TV_LF': MediaChannel(
            name='TV_LF',
            platform='TV',
            format_type='LF',
            cpm=57,  # ₺57 CPM
            max_reach=0.85,  # 85% max reach (more realistic ceiling)
            reach_curve_k=2.0  # Gradual curve - prioritizes reach over frequency
        ),
        'YT_LF': MediaChannel(
            name='YT_LF', 
            platform='YouTube',
            format_type='LF',
            cpm=60,  # ₺60 CPM
            max_reach=0.75,  # 75% max reach (more realistic ceiling)
            reach_curve_k=1.8  # Gradual curve - prioritizes reach over frequency
        ),
        'YT_SF': MediaChannel(
            name='YT_SF',
            platform='YouTube', 
            format_type='SF',
            cpm=25,  # ₺25 CPM
            max_reach=0.70,  # 70% max reach (realistic for short-form)
            reach_curve_k=1.5  # Moderate curve - good for scaling
        ),
        'Meta_SF': MediaChannel(
            name='Meta_SF',
            platform='Meta',
            format_type='SF', 
            cpm=12,  # ₺12 CPM
            max_reach=0.85,  # 85% max reach (high for social media)
            reach_curve_k=1.3  # Moderate curve - good for scaling
        ),
        'TT_SF': MediaChannel(
            name='TT_SF',
            platform='TikTok',
            format_type='SF',
            cpm=10,  # ₺10 CPM
            max_reach=0.65,  # 65% max reach (realistic for TikTok)
            reach_curve_k=1.0  # Moderate curve - good for scaling
        )
    }
    
    return channels

def load_realistic_audience_data():
    """Load realistic audience segment data from audience-segments.xlsx"""
    
    import pandas as pd
    
    # Load audience data from Excel file
    df = pd.read_excel('audience-segments.xlsx')
    
    # Calculate total addressable market (max reach across all platforms)
    TAM = df[['TV', 'YouTube', 'Meta', 'TikTok']].max(axis=1).sum()
    
    # Create segments based on actual data
    segments = {}
    
    for _, row in df.iterrows():
        segment_name = row['Segment'].replace('-', '_').replace('+', '_plus')
        
        # Get actual addressable populations per channel from Excel
        tv_audience = row['TV']
        youtube_audience = row['YouTube']
        meta_audience = row['Meta']
        tiktok_audience = row['TikTok']
        
        # Use the maximum audience as the segment population
        segment_population = max(tv_audience, youtube_audience, meta_audience, tiktok_audience)
        
        # Calculate channel affinities as ratios of addressable audience to segment population
        segments[segment_name] = AudienceSegment(
            name=segment_name,
            population=segment_population,
            channel_affinities={
                'TV_LF': tv_audience / segment_population,
                'YT_LF': youtube_audience / segment_population,  # Full YouTube audience for LF
                'YT_SF': youtube_audience / segment_population,  # Full YouTube audience for SF (same audience)
                'Meta_SF': meta_audience / segment_population,
                'TT_SF': tiktok_audience / segment_population
            }
        )
    
    return segments, TAM

# =========================================================================
# SAINSBURY WEIGHTED MODEL COMPONENTS
# =========================================================================

def load_sainsbury_quality_weights():
    """Load quality weights for Sainsbury model based on VAT data"""
    
    # Quality weights based on engagement and attention metrics
    # These simulate VAT (Video Attention Time) data
    quality_weights = {
        'TV_LF': 0.98,     # Highest quality - full attention, lean-back
        'YT_LF': 0.92,     # Very high quality - chosen content, engaged viewing
        'YT_SF': 0.58,     # Medium quality - quick consumption, but engaging
        'Meta_SF': 0.62,   # Medium quality - social context, but shorter attention
        'TT_SF': 0.52      # Lower quality - very short attention spans
    }
    
    return quality_weights

def load_gwi_overlap_factors():
    """Load GWI overlap factors for cross-platform deduplication"""
    
    # GWI-based overlap factors (realistic audience overlap between platforms)
    overlap_factors = {
        ('YT', 'Meta'): 0.85,   # 85% overlap between YouTube and Meta users
        ('YT', 'TT'): 0.65,     # 65% overlap between YouTube and TikTok users  
        ('Meta', 'TT'): 0.70    # 70% overlap between Meta and TikTok users
    }
    
    return overlap_factors

def sainsbury_quality_reach_segment(channel_reach_dict: Dict[str, float], 
                                  quality_weights: Dict[str, float],
                                  convergence_factor: float = 0.85) -> float:
    """
    Calculate Sainsbury weighted quality reach for a segment
    
    Args:
        channel_reach_dict: Reach by channel for this segment
        quality_weights: Quality weights for each channel
        convergence_factor: Convergence factor (0.8-0.9)
    
    Returns:
        Quality reach using Sainsbury approach
    """
    
    # Separate long-form and short-form channels
    lf_channels = {k: v for k, v in channel_reach_dict.items() if k.endswith('_LF')}
    sf_channels = {k: v for k, v in channel_reach_dict.items() if k.endswith('_SF')}
    
    # Calculate long-form quality reach (gets priority)
    lf_quality_reach = 0
    lf_quality_universe = 1.0
    
    # Long-form quality calculation with engagement depth bonus
    lf_quality_order = ['TV_LF', 'YT_LF']  # Ordered by quality priority
    
    for channel in lf_quality_order:
        if channel in lf_channels and lf_channels[channel] > 0:
            # Long-format gets full quality contribution with depth bonus
            format_quality_weight = quality_weights[channel]
            engagement_depth_bonus = 1.2  # 20% bonus for storytelling capability
            
            # Calculate this channel's quality contribution
            format_quality_reach = (format_quality_weight * lf_channels[channel] * 
                                  lf_quality_universe * engagement_depth_bonus)
            
            lf_quality_reach += format_quality_reach
            
            # Reduce available universe for next long-format (diminishing returns)
            lf_quality_universe *= (1 - quality_weights[channel] * lf_channels[channel] * 0.7)
            lf_quality_universe = max(0.2, lf_quality_universe)  # Maintain quality floor
    
    # Calculate short-form quality supplement
    sf_quality_reach = 0
    sf_available_universe = max(0.3, 1.0 - lf_quality_reach)  # SF limited to remaining universe
    
    # Sort short-format by efficiency (but with quality penalties)
    sf_efficiency_order = ['YT_SF', 'Meta_SF', 'TT_SF']
    
    for channel in sf_efficiency_order:
        if channel in sf_channels and sf_channels[channel] > 0:
            # Short-format gets penalized quality contribution
            format_quality_weight = quality_weights[channel]
            attention_penalty = 0.6  # 40% penalty for limited engagement depth
            
            # Calculate supplemental quality contribution
            format_contribution = (format_quality_weight * sf_channels[channel] * 
                                 sf_available_universe * attention_penalty)
            
            sf_quality_reach += format_contribution
            
            # Reduce available universe with stronger diminishing returns
            sf_available_universe *= (1 - quality_weights[channel] * sf_channels[channel] * 0.9)
            sf_available_universe = max(0.1, sf_available_universe)
    
    # Combine with long-format priority
    total_quality_reach = lf_quality_reach + sf_quality_reach
    
    # Apply final quality hierarchy enforcement
    if lf_quality_reach > 0:
        # If long-format is present, it should dominate quality calculation
        lf_dominance_factor = min(1.0, lf_quality_reach / (lf_quality_reach + sf_quality_reach + 0.001))
        quality_hierarchy_bonus = 1.0 + (lf_dominance_factor * 0.1)  # Up to 10% bonus for LF dominance
        total_quality_reach *= quality_hierarchy_bonus
    
    # Apply Sainsbury-style convergence effects
    total_channels = len([ch for ch, reach in channel_reach_dict.items() if reach > 0])
    
    if total_channels > 1:
        # Multi-channel convergence penalty (reduces quality when spreading across many channels)
        convergence_penalty = convergence_factor ** (total_channels - 1)
        
        # But give bonus for long-format concentration
        lf_count = len([ch for ch, reach in lf_channels.items() if reach > 0])
        sf_count = len([ch for ch, reach in sf_channels.items() if reach > 0])
        
        if lf_count > sf_count:
            # Bonus for focusing on long-format quality
            lf_focus_bonus = 1.0 + (lf_count / (lf_count + sf_count)) * 0.15
            convergence_penalty *= lf_focus_bonus
        
        total_quality_reach *= convergence_penalty
    
    return min(1.0, total_quality_reach)

def gwi_platform_deduplication(platform_reach_dict: Dict[str, float], 
                              overlap_factors: Dict[Tuple[str, str], float]) -> float:
    """
    Apply GWI-based cross-platform deduplication
    
    Args:
        platform_reach_dict: Reach by platform
        overlap_factors: GWI overlap factors between platforms
    
    Returns:
        Net deduplicated reach across platforms
    """
    
    R_TV = platform_reach_dict.get('TV', 0)
    R_YT = platform_reach_dict.get('YouTube', 0)
    R_Meta = platform_reach_dict.get('Meta', 0)
    R_TT = platform_reach_dict.get('TikTok', 0)
    
    # GWI pairwise intersections (digital platforms only)
    I_YM = overlap_factors.get(('YT', 'Meta'), 0.85) * min(R_YT, R_Meta)
    I_YT = overlap_factors.get(('YT', 'TT'), 0.65) * min(R_YT, R_TT)
    I_MT = overlap_factors.get(('Meta', 'TT'), 0.70) * min(R_Meta, R_TT)
    
    # Net reach using GWI inclusion-exclusion principle
    net_reach = R_TV + R_YT + R_Meta + R_TT - I_YM - I_YT - I_MT
    
    # Cap at 100% to prevent impossible reach values
    return min(1.0, net_reach)

# =========================================================================
# MEDIA STRATEGY PRESETS
# =========================================================================

def get_media_strategies():
    """Define predefined media strategies with channel preferences"""
    
    strategies = {
        'long_format_focused': {
            'name': 'Long-Format Focused',
            'description': 'TV and YouTube long-format heavy strategy for storytelling and brand building',
            'channel_weights': {
                'TV_LF': 0.55,      # 55% preference for TV long-format
                'YT_LF': 0.35,      # 35% preference for YouTube long-format
                'YT_SF': 0.06,      # 6% for YouTube short-format (reach supplement)
                'Meta_SF': 0.03,    # 3% for Meta short-format (reach supplement)
                'TT_SF': 0.01       # 1% for TikTok short-format (reach supplement)
            },
            'min_constraints': {
                'TV_LF': 0.30,      # Minimum 30% of budget to TV (reduced from 40%)
                'YT_LF': 0.15,      # Minimum 15% of budget to YouTube LF (reduced from 25%)
                'YT_SF': 0.0,       # No minimum for short-format
                'Meta_SF': 0.0,
                'TT_SF': 0.0
            },
            'max_constraints': {
                'TV_LF': 0.80,      # Maximum 80% of budget to TV (increased from 70%)
                'YT_LF': 0.60,      # Maximum 60% of budget to YouTube LF (increased from 50%)
                'YT_SF': 0.25,      # Maximum 25% to short-format channels (increased from 6%)
                'Meta_SF': 0.15,    # Maximum 15% to short-format channels (increased from 3%)
                'TT_SF': 0.10       # Maximum 10% to short-format channels (increased from 1%)
            },
            'benefits': [
                'Higher brand recall and storytelling impact',
                'Premium audience engagement',
                'Better for complex product messaging',
                'Stronger emotional connection'
            ]
        },
        
        'short_format_focused': {
            'name': 'Short-Format Focused',
            'description': 'YouTube, TikTok, and Meta short-format heavy for reach and efficiency',
            'channel_weights': {
                'TV_LF': 0.0,       # 0% preference for TV
                'YT_LF': 0.0,       # 0% preference for YouTube long-format
                'YT_SF': 0.40,      # 40% preference for YouTube short-format
                'Meta_SF': 0.40,    # 40% preference for Meta short-format
                'TT_SF': 0.20       # 20% preference for TikTok short-format
            },
            'min_constraints': {
                'TV_LF': 0.0,       # No long-format
                'YT_LF': 0.0,
                'YT_SF': 0.30,      # Minimum 30% of budget to YouTube SF
                'Meta_SF': 0.30,    # Minimum 30% of budget to Meta SF
                'TT_SF': 0.15       # Minimum 15% of budget to TikTok SF
            },
            'max_constraints': {
                'TV_LF': 0.0,       # No long-format allowed
                'YT_LF': 0.0,
                'YT_SF': 0.60,      # Maximum 60% to YouTube SF
                'Meta_SF': 0.60,    # Maximum 60% to Meta SF
                'TT_SF': 0.40       # Maximum 40% to TikTok SF
            },
            'benefits': [
                'Maximum reach efficiency and cost-effectiveness',
                'Higher frequency and viral potential',
                'Better for younger demographics',
                'Faster campaign deployment'
            ]
        }
    }
    
    return strategies

def optimize_media_plan_with_strategy(target_reach_pct: float, budget: float, 
                                    strategy_name: str,
                                    use_sainsbury: bool = False, 
                                    convergence_factor: float = 0.85) -> Tuple[Dict[str, float], Dict]:
    """Optimize media plan using a predefined strategy"""
    
    strategies = get_media_strategies()
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    strategy = strategies[strategy_name]
    channels = load_realistic_media_data()
    channel_names = list(channels.keys())
    
    # Create bounds based on strategy constraints
    bounds = []
    for channel_name in channel_names:
        min_spend = budget * strategy['min_constraints'].get(channel_name, 0)
        max_spend = budget * strategy['max_constraints'].get(channel_name, 0.8)
        bounds.append((min_spend, max_spend))
    
    # Strategy-weighted objective function
    def strategy_objective_function(spend_array: np.ndarray) -> float:
        spend_dict = dict(zip(channel_names, spend_array))
        
        # Calculate KPIs
        results = calculate_campaign_kpis(spend_dict, use_sainsbury, convergence_factor)
        
        # Always use net reach for strategy comparison (not quality reach)
        # This ensures we can compare reach efficiency between strategies
        achieved_reach = results['net_reach_pct']
        
        # IMPROVED PENALTY SCALING for better convergence
        # Use gradual penalty for reach (same as main objective function)
        reach_diff = abs(achieved_reach - target_reach_pct)
        reach_penalty = (reach_diff ** 1.5) * 2  # Gradual penalty scaling
        
        # Budget constraint penalty (reduced)
        total_spend = sum(spend_array)
        budget_penalty = max(0, total_spend - budget) * 0.001
        
        # Strategy alignment penalty - encourage spending according to strategy weights
        strategy_penalty = 0
        if total_spend > 0:
            for i, channel_name in enumerate(channel_names):
                actual_weight = spend_array[i] / total_spend
                target_weight = strategy['channel_weights'].get(channel_name, 0)
                deviation = abs(actual_weight - target_weight)
                strategy_penalty += deviation * 10  # Reduced penalty for strategy deviation
        
        # Add efficiency bonus (same as main objective function)
        if total_spend > 0:
            efficiency_bonus = -(achieved_reach / (total_spend / 1_000_000)) * 0.3
        else:
            efficiency_bonus = 0
        
        total_penalty = reach_penalty + budget_penalty + strategy_penalty + efficiency_bonus
        
        return max(0.1, total_penalty)  # Ensure positive penalty
    
    # Try simple iterative optimization first (more reliable)
    print(f"🔄 Trying iterative optimization for {strategy_name}...")
    
    # Start with strategy-based allocation
    if strategy_name == 'long_format_focused':
        base_allocation = {
            'TV_LF': 0.55,
            'YT_LF': 0.35,
            'YT_SF': 0.06,
            'Meta_SF': 0.03,
            'TT_SF': 0.01
        }
    elif strategy_name == 'short_format_focused':
        base_allocation = {
            'TV_LF': 0.0,
            'YT_LF': 0.0,
            'YT_SF': 0.40,
            'Meta_SF': 0.40,
            'TT_SF': 0.20
        }
    else:
        base_allocation = {
            'TV_LF': 0.30,
            'YT_LF': 0.25,
            'YT_SF': 0.20,
            'Meta_SF': 0.15,
            'TT_SF': 0.10
        }
    
    # Try iterative improvement
    best_spend = {ch: budget * weight for ch, weight in base_allocation.items()}
    best_results = calculate_campaign_kpis(best_spend, use_sainsbury, convergence_factor)
    best_reach = best_results['net_reach_pct']
    initial_reach = best_reach  # Store initial reach for comparison
    
    # Simple hill climbing for 5 iterations
    for iteration in range(5):
        # Try adjusting allocations
        for main_channel in ['TV_LF', 'YT_LF', 'YT_SF', 'Meta_SF']:
            if main_channel not in best_spend:
                continue
                
            # Try increasing this channel by 10% and reducing others proportionally
            test_spend = best_spend.copy()
            increase_amount = budget * 0.1
            
            # Increase main channel (within bounds)
            max_allowed = budget * strategy['max_constraints'].get(main_channel, 0.8)
            if test_spend[main_channel] + increase_amount <= max_allowed:
                test_spend[main_channel] += increase_amount
                
                # Reduce other channels proportionally
                other_channels = [ch for ch in test_spend.keys() if ch != main_channel and test_spend[ch] > 0]
                if other_channels:
                    total_other = sum(test_spend[ch] for ch in other_channels)
                    if total_other > 0:
                        reduction_factor = (total_other - increase_amount) / total_other
                        for ch in other_channels:
                            min_allowed = budget * strategy['min_constraints'].get(ch, 0)
                            test_spend[ch] = max(min_allowed, test_spend[ch] * reduction_factor)
                
                # Test this allocation
                test_results = calculate_campaign_kpis(test_spend, use_sainsbury, convergence_factor)
                test_reach = test_results['net_reach_pct']
                
                # If better, keep it
                if test_reach > best_reach:
                    best_spend = test_spend
                    best_results = test_results
                    best_reach = test_reach
                    print(f"   Iteration {iteration+1}: Improved to {best_reach:.1f}% reach")
    
    # Check if iterative optimization was successful
    reach_gap = abs(best_reach - target_reach_pct)
    improvement = best_reach - initial_reach
    
    if reach_gap < 15 or improvement > 2:  # Within 15% of target OR 2% improvement
        print(f"✅ Iterative optimization successful: {best_reach:.1f}% reach (improved by {improvement:.1f}%)")
        
        # Add strategy information to results
        best_results['strategy_name'] = strategy['name']
        best_results['strategy_description'] = strategy['description']
        best_results['strategy_benefits'] = strategy['benefits']
        
        return best_spend, best_results
    
    # If iterative optimization failed, try differential evolution with more iterations
    print(f"🔄 Trying differential evolution optimization...")
    
    # Optimize with improved settings for better convergence
    result = differential_evolution(
        strategy_objective_function,
        bounds,
        seed=42,
        maxiter=50,     # More iterations for difficult cases
        popsize=12,     # Larger population for better exploration
        atol=1e-4,      # Tighter tolerance
        tol=1e-4,       # Tighter tolerance
        polish=True,    # Final polish for better results
        updating='deferred'  # Better convergence strategy
    )
    
    if result.success:
        optimal_spend = dict(zip(channel_names, result.x))
        
        # Ensure budget constraint
        total_spend = sum(optimal_spend.values())
        if total_spend > budget:
            scale_factor = budget / total_spend
            optimal_spend = {k: v * scale_factor for k, v in optimal_spend.items()}
        
        # Calculate final KPIs
        final_results = calculate_campaign_kpis(optimal_spend, use_sainsbury, convergence_factor)
        
        # Add strategy information to results
        final_results['strategy_name'] = strategy['name']
        final_results['strategy_description'] = strategy['description']
        final_results['strategy_benefits'] = strategy['benefits']
        
        return optimal_spend, final_results
    else:
        # FALLBACK: Use simple heuristic allocation if optimization fails
        print(f"⚠️  Optimization failed, using fallback heuristic allocation...")
        
        # Simple strategy-based allocation
        if strategy_name == 'long_format_focused':
            fallback_spend = {
                'TV_LF': budget * 0.60,    # 60% to TV Long-Format
                'YT_LF': budget * 0.30,    # 30% to YouTube Long-Format
                'YT_SF': budget * 0.05,    # 5% to YouTube Short-Format
                'Meta_SF': budget * 0.03,  # 3% to Meta Short-Format
                'TT_SF': budget * 0.02     # 2% to TikTok Short-Format
            }
        elif strategy_name == 'short_format_focused':
            fallback_spend = {
                'TV_LF': budget * 0.10,    # 10% to TV Long-Format
                'YT_LF': budget * 0.15,    # 15% to YouTube Long-Format
                'YT_SF': budget * 0.35,    # 35% to YouTube Short-Format
                'Meta_SF': budget * 0.30,  # 30% to Meta Short-Format
                'TT_SF': budget * 0.10     # 10% to TikTok Short-Format
            }
        else:
            # Balanced allocation
            fallback_spend = {
                'TV_LF': budget * 0.30,
                'YT_LF': budget * 0.25,
                'YT_SF': budget * 0.20,
                'Meta_SF': budget * 0.15,
                'TT_SF': budget * 0.10
            }
        
        # Calculate KPIs for fallback allocation
        fallback_results = calculate_campaign_kpis(fallback_spend, use_sainsbury, convergence_factor)
        
        return fallback_spend, fallback_results

def analyze_strategy_performance(spend_dict: Dict[str, float], 
                               use_sainsbury: bool = False,
                               convergence_factor: float = 0.85) -> Dict:
    """Analyze which strategy a given spend allocation most closely follows"""
    
    strategies = get_media_strategies()
    results = calculate_campaign_kpis(spend_dict, use_sainsbury, convergence_factor)
    
    total_spend = sum(spend_dict.values())
    if total_spend == 0:
        return results
    
    # Calculate actual channel weights
    actual_weights = {k: v / total_spend for k, v in spend_dict.items()}
    
    # Calculate strategy alignment scores
    strategy_scores = {}
    for strategy_name, strategy in strategies.items():
        score = 0
        for channel, target_weight in strategy['channel_weights'].items():
            actual_weight = actual_weights.get(channel, 0)
            # Lower deviation = higher score
            deviation = abs(actual_weight - target_weight)
            score += max(0, 1 - deviation * 2)  # Score from 0 to 1 per channel
        
        # Normalize by number of channels
        strategy_scores[strategy_name] = score / len(strategy['channel_weights'])
    
    # Find best matching strategy
    best_strategy = max(strategy_scores, key=strategy_scores.get)
    best_score = strategy_scores[best_strategy]
    
    # Add strategy analysis to results
    results['strategy_analysis'] = {
        'best_match': best_strategy,
        'alignment_score': best_score,
        'all_scores': strategy_scores,
        'actual_weights': actual_weights,
        'strategy_classification': 'Strong' if best_score > 0.7 else 'Moderate' if best_score > 0.4 else 'Weak'
    }
    
    return results

# =========================================================================
# CORE CALCULATION FUNCTIONS
# =========================================================================

def calculate_impressions(spend: float, cpm: float) -> int:
    """Calculate impressions from spend and CPM"""
    if cpm <= 0:
        return 0
    return int((spend / cpm) * 1000)

def calculate_grps(impressions: int, population: int) -> float:
    """Calculate GRPs (Gross Rating Points)"""
    if population <= 0:
        return 0
    return (impressions / population) * 100

def calculate_reach_from_grps(grps: float, max_reach: float, k: float) -> float:
    """
    Calculate reach using a more scalable logarithmic curve
    
    Formula: reach = max_reach * (grps / (grps + k))
    This provides better scalability at higher budget levels
    
    Args:
        grps: Gross Rating Points
        max_reach: Maximum possible reach (0-1)
        k: Reach curve steepness parameter (higher k = more gradual)
    """
    if grps <= 0:
        return 0
    
    # Convert GRPs to reach using logarithmic curve for better scalability
    # This formula allows for more gradual reach growth at higher budgets
    reach = max_reach * (grps / (grps + k * 100))
    
    # Cap at maximum reach
    return min(reach, max_reach)

def calculate_frequency(impressions: int, reach_people: int) -> float:
    """Calculate frequency from impressions and reach"""
    if reach_people <= 0:
        return 0
    return impressions / reach_people

# =========================================================================
# SEGMENT-LEVEL CALCULATIONS
# =========================================================================

def calculate_segment_reach(spend_dict: Dict[str, float], segment: AudienceSegment, 
                          channels: Dict[str, MediaChannel], 
                          use_sainsbury: bool = False,
                          convergence_factor: float = 0.85) -> Dict:
    """Calculate reach metrics for a single segment"""
    
    segment_results = {
        'segment_name': segment.name,
        'population': segment.population,
        'channel_results': {},
        'platform_reach': {},
        'net_reach': 0,
        'quality_reach': 0,
        'total_impressions': 0
    }
    
    # Calculate for each channel
    channel_reach_dict = {}
    for channel_name, spend in spend_dict.items():
        if spend <= 0 or channel_name not in channels:
            continue
            
        channel = channels[channel_name]
        
        # Calculate addressable population for this channel
        addressable_pop = int(segment.population * segment.channel_affinities.get(channel_name, 0))
        
        if addressable_pop <= 0:
            continue
        
        # Calculate impressions
        impressions = calculate_impressions(spend, channel.cpm)
        
        # Calculate GRPs
        grps = calculate_grps(impressions, addressable_pop)
        
        # Calculate reach
        reach_pct = calculate_reach_from_grps(grps, channel.max_reach, channel.reach_curve_k)
        reach_people = int(reach_pct * addressable_pop)
        
        # Calculate frequency
        frequency = calculate_frequency(impressions, reach_people) if reach_people > 0 else 0
        
        segment_results['channel_results'][channel_name] = {
            'spend': spend,
            'addressable_population': addressable_pop,
            'impressions': impressions,
            'grps': grps,
            'reach_pct': reach_pct,
            'reach_people': reach_people,
            'frequency': frequency
        }
        
        # Store for quality calculations
        channel_reach_dict[channel_name] = reach_pct
        
        segment_results['total_impressions'] += impressions
    
    # Calculate platform-level reach (combine formats within platforms)
    platform_reach_people = {}
    for channel_name, result in segment_results['channel_results'].items():
        platform = channels[channel_name].platform
        if platform not in platform_reach_people:
            platform_reach_people[platform] = 0
        
        # Simple additive model within platform (can be improved with overlap)
        platform_reach_people[platform] += result['reach_people']
    
    # Cap platform reach at addressable population
    for platform in platform_reach_people:
        max_platform_reach = int(segment.population * 0.90)  # 90% max platform reach (higher ceiling)
        platform_reach_people[platform] = min(platform_reach_people[platform], max_platform_reach)
    
    segment_results['platform_reach'] = platform_reach_people
    
    # Calculate net reach across platforms
    if use_sainsbury:
        # Use GWI-based deduplication for Sainsbury model
        platform_reach_pct = {k: v / segment.population for k, v in platform_reach_people.items()}
        overlap_factors = load_gwi_overlap_factors()
        net_reach_pct = gwi_platform_deduplication(platform_reach_pct, overlap_factors)
        net_reach_people = int(net_reach_pct * segment.population)
        
        # Calculate Sainsbury quality reach
        quality_weights = load_sainsbury_quality_weights()
        quality_reach_pct = sainsbury_quality_reach_segment(
            channel_reach_dict, quality_weights, convergence_factor
        )
        quality_reach_people = int(quality_reach_pct * segment.population)
        
    else:
        # Use simple deduplication for standard model
        total_platform_reach = sum(platform_reach_people.values())
        
        # Dynamic deduplication based on reach level - less overlap at higher budgets
        reach_ratio = min(1.0, total_platform_reach / segment.population)
        base_deduplication = 0.65  # Base 35% overlap reduction
        bonus_deduplication = 0.15 * reach_ratio  # Up to 15% bonus at high reach
        deduplication_factor = min(0.80, base_deduplication + bonus_deduplication)
        
        net_reach_people = int(total_platform_reach * deduplication_factor)
        
        # Simple quality reach (no sophisticated weighting)
        quality_reach_people = net_reach_people
    
    # Cap at segment population
    net_reach_people = min(net_reach_people, segment.population)
    quality_reach_people = min(quality_reach_people, segment.population)
    
    segment_results['net_reach'] = net_reach_people
    segment_results['quality_reach'] = quality_reach_people
    
    return segment_results

# =========================================================================
# CAMPAIGN-LEVEL CALCULATIONS
# =========================================================================

def calculate_campaign_kpis(spend_dict: Dict[str, float], 
                          use_sainsbury: bool = False,
                          convergence_factor: float = 0.85) -> Dict:
    """Calculate campaign-level KPIs"""
    
    channels = load_realistic_media_data()
    segments, TAM = load_realistic_audience_data()
    
    # Calculate for each segment
    segment_results = []
    for segment in segments.values():
        result = calculate_segment_reach(spend_dict, segment, channels, use_sainsbury, convergence_factor)
        segment_results.append(result)
    
    # Aggregate results
    campaign_results = {
        'total_spend': sum(spend_dict.values()),
        'segment_results': segment_results,
        'channel_totals': {},
        'platform_totals': {},
        'net_reach_people': 0,
        'net_reach_pct': 0,
        'quality_reach_people': 0,
        'quality_reach_pct': 0,
        'total_impressions': 0,
        'use_sainsbury': use_sainsbury
    }
    
    # Calculate channel totals - FIXED CALCULATION
    for channel_name in channels.keys():
        channel_spend = spend_dict.get(channel_name, 0)
        
        # Calculate impressions once per channel based on total spend
        channel_impressions = calculate_impressions(channel_spend, channels[channel_name].cpm)
        
        channel_total = {
            'spend': channel_spend,
            'impressions': channel_impressions,
            'reach_people': 0,
            'frequency': 0
        }
        
        if channel_spend > 0 and channel_impressions > 0:
            # Calculate weighted average frequency across segments
            total_weighted_frequency = 0
            total_segment_reach = 0
            
            for segment_result in segment_results:
                if channel_name in segment_result['channel_results']:
                    channel_data = segment_result['channel_results'][channel_name]
                    segment_reach = channel_data['reach_people']
                    segment_frequency = channel_data['frequency']
                    
                    total_segment_reach += segment_reach
                    total_weighted_frequency += segment_frequency * segment_reach
            
            # Calculate weighted average frequency
            if total_segment_reach > 0:
                weighted_avg_frequency = total_weighted_frequency / total_segment_reach
                
                # CAP FREQUENCY AT 5.0x TO PRIORITIZE REACH OVER FREQUENCY
                capped_frequency = min(5.0, max(1.0, weighted_avg_frequency))
                channel_total['frequency'] = capped_frequency
                
                # CORRECT REACH CALCULATION: Reach = Impressions ÷ Frequency
                # With frequency capping, this will increase reach when frequency would exceed 5x
                channel_total['reach_people'] = int(channel_impressions / channel_total['frequency'])
            else:
                channel_total['frequency'] = 0
                channel_total['reach_people'] = 0
        else:
            channel_total['frequency'] = 0
            channel_total['reach_people'] = 0
        
        campaign_results['channel_totals'][channel_name] = channel_total
    
    # Calculate platform totals - sum channel reach within platforms
    platform_totals = {}
    for channel_name, channel_total in campaign_results['channel_totals'].items():
        if channel_total['reach_people'] > 0:
            channel_info = channels[channel_name]
            platform = channel_info.platform
            
            if platform not in platform_totals:
                platform_totals[platform] = 0
            platform_totals[platform] += channel_total['reach_people']
    
    campaign_results['platform_totals'] = platform_totals
    
    # Calculate net reach using proper deduplication
    if use_sainsbury:
        # Use GWI-based deduplication for Sainsbury model
        platform_reach_pct = {}
        for platform, reach_people in platform_totals.items():
            platform_reach_pct[platform] = reach_people / TAM
        
        overlap_factors = load_gwi_overlap_factors()
        net_reach_pct = gwi_platform_deduplication(platform_reach_pct, overlap_factors)
        net_reach_people = int(net_reach_pct * TAM)
        
        # Calculate quality reach using channel totals
        channel_reach_dict = {}
        for channel_name, channel_total in campaign_results['channel_totals'].items():
            if channel_total['reach_people'] > 0:
                channel_reach_dict[channel_name] = channel_total['reach_people'] / TAM
        
        quality_weights = load_sainsbury_quality_weights()
        quality_reach_pct = sainsbury_quality_reach_segment(channel_reach_dict, quality_weights, convergence_factor)
        quality_reach_people = int(quality_reach_pct * TAM)
        
    else:
        # Use simple deduplication for standard model
        total_platform_reach = sum(platform_totals.values())
        
        # Apply realistic deduplication factor
        if total_platform_reach > 0:
            reach_ratio = min(1.0, total_platform_reach / TAM)
            base_deduplication = 0.75  # 25% overlap reduction
            bonus_deduplication = 0.10 * reach_ratio  # Up to 10% bonus at high reach
            deduplication_factor = min(0.85, base_deduplication + bonus_deduplication)
            
            net_reach_people = int(total_platform_reach * deduplication_factor)
        else:
            net_reach_people = 0
        
        # Simple quality reach (no sophisticated weighting)
        quality_reach_people = net_reach_people
    
    # Cap at TAM to prevent impossible reach values
    net_reach_people = min(net_reach_people, TAM)
    quality_reach_people = min(quality_reach_people, TAM)
    
    campaign_results['net_reach_people'] = net_reach_people
    campaign_results['net_reach_pct'] = (net_reach_people / TAM) * 100
    campaign_results['quality_reach_people'] = quality_reach_people
    campaign_results['quality_reach_pct'] = (quality_reach_people / TAM) * 100
    
    # Calculate total impressions (sum from channel totals)
    campaign_results['total_impressions'] = sum(
        channel_total['impressions'] for channel_total in campaign_results['channel_totals'].values()
    )
    
    # Calculate average frequency (weighted by reach)
    if net_reach_people > 0:
        total_weighted_frequency = 0
        total_channel_reach = 0
        
        # Weight each channel's frequency by its reach
        for channel_name, channel_total in campaign_results['channel_totals'].items():
            if channel_total['reach_people'] > 0:
                channel_frequency = channel_total['frequency']
                channel_reach = channel_total['reach_people']
                total_weighted_frequency += channel_frequency * channel_reach
                total_channel_reach += channel_reach
        
        if total_channel_reach > 0:
            weighted_avg_frequency = total_weighted_frequency / total_channel_reach
            campaign_results['avg_frequency'] = max(1.0, weighted_avg_frequency)
        else:
            campaign_results['avg_frequency'] = 1.0
    else:
        campaign_results['avg_frequency'] = 0
    
    return campaign_results

# =========================================================================
# OPTIMIZATION FUNCTIONS
# =========================================================================

def objective_function(spend_array: np.ndarray, target_reach_pct: float, budget: float,
                      use_sainsbury: bool = False, convergence_factor: float = 0.85) -> float:
    """Improved objective function with better penalty scaling"""
    
    channels = load_realistic_media_data()
    channel_names = list(channels.keys())
    
    # Convert array to spend dict
    spend_dict = dict(zip(channel_names, spend_array))
    
    # Calculate KPIs
    results = calculate_campaign_kpis(spend_dict, use_sainsbury, convergence_factor)
    
    # Use quality reach for Sainsbury model, net reach for standard model
    if use_sainsbury:
        achieved_reach = results['quality_reach_pct']
    else:
        achieved_reach = results['net_reach_pct']
    
    # IMPROVED PENALTY SCALING for better convergence
    # Use squared penalty for reach (more gradual near target)
    reach_diff = abs(achieved_reach - target_reach_pct)
    reach_penalty = (reach_diff ** 1.5) * 2  # Reduced scaling, gradual penalty
    
    # Budget penalty (keep reasonable)
    budget_penalty = max(0, sum(spend_array) - budget) * 0.001  # Reduced penalty
    
    # Frequency penalty (encourage efficiency)
    avg_frequency = results.get('avg_frequency', 1.0)
    frequency_penalty = max(0, (avg_frequency - 4.0) * 5) if avg_frequency > 4.0 else 0
    
    # Add efficiency bonus (reward higher reach per dollar)
    total_spend = sum(spend_array)
    if total_spend > 0:
        efficiency_bonus = -(achieved_reach / (total_spend / 1_000_000)) * 0.5  # Bonus for efficiency
    else:
        efficiency_bonus = 0
    
    total_penalty = reach_penalty + budget_penalty + frequency_penalty + efficiency_bonus
    
    return max(0.1, total_penalty)  # Ensure positive penalty

def optimize_media_plan(target_reach_pct: float, budget: float, 
                       use_sainsbury: bool = False, 
                       convergence_factor: float = 0.85) -> Tuple[Dict[str, float], Dict]:
    """Optimize media plan to achieve target reach"""
    
    channels = load_realistic_media_data()
    channel_names = list(channels.keys())
    
    # Set bounds (min 0, max 70% of budget per channel)
    bounds = [(0, budget * 0.7) for _ in channel_names]
    
    # Optimize
    result = differential_evolution(
        objective_function,
        bounds,
        args=(target_reach_pct, budget, use_sainsbury, convergence_factor),
        seed=42,
        maxiter=25,     # Slightly more iterations for better convergence
        popsize=6,      # Balanced population size
        atol=1e-3,      # Reasonable tolerance
        tol=1e-3,       # Reasonable tolerance
        polish=True,    # Final polish for better results
        updating='deferred'  # Better convergence strategy
    )
    
    if result.success:
        optimal_spend = dict(zip(channel_names, result.x))
        
        # Ensure budget constraint
        total_spend = sum(optimal_spend.values())
        if total_spend > budget:
            scale_factor = budget / total_spend
            optimal_spend = {k: v * scale_factor for k, v in optimal_spend.items()}
        
        # Calculate final KPIs
        final_results = calculate_campaign_kpis(optimal_spend, use_sainsbury, convergence_factor)
        
        return optimal_spend, final_results
    else:
        return None, None

# =========================================================================
# TESTING AND VALIDATION
# =========================================================================

def test_calculation_system():
    """Test the calculation system with sample data"""
    
    print("🧪 Testing Clean Calculation System v3.0 with Sainsbury Option")
    print("=" * 65)
    
    # Test spend allocation
    test_spend = {
        'TV_LF': 2_000_000,
        'YT_LF': 800_000,
        'YT_SF': 600_000,
        'Meta_SF': 800_000,
        'TT_SF': 300_000
    }
    
    print(f"Test spend: {test_spend}")
    print(f"Total spend: ₺{sum(test_spend.values()):,}")
    
    # Test both models
    print(f"\n📊 STANDARD MODEL RESULTS:")
    print("-" * 40)
    
    standard_results = calculate_campaign_kpis(test_spend, use_sainsbury=False)
    
    print(f"Net Reach: {standard_results['net_reach_pct']:.1f}% ({standard_results['net_reach_people']:,} people)")
    print(f"Total Impressions: {standard_results['total_impressions']:,}")
    
    print(f"\n📊 SAINSBURY MODEL RESULTS:")
    print("-" * 40)
    
    sainsbury_results = calculate_campaign_kpis(test_spend, use_sainsbury=True)
    
    print(f"Net Reach: {sainsbury_results['net_reach_pct']:.1f}% ({sainsbury_results['net_reach_people']:,} people)")
    print(f"Quality Reach: {sainsbury_results['quality_reach_pct']:.1f}% ({sainsbury_results['quality_reach_people']:,} people)")
    print(f"Total Impressions: {sainsbury_results['total_impressions']:,}")
    
    # Compare models
    print(f"\n🔄 MODEL COMPARISON:")
    print("-" * 40)
    reach_diff = sainsbury_results['net_reach_pct'] - standard_results['net_reach_pct']
    print(f"Reach Difference: {reach_diff:+.1f}pp (Sainsbury vs Standard)")
    print(f"Quality Insight: {sainsbury_results['quality_reach_pct']:.1f}% quality reach with Sainsbury")
    
    # Test optimization with both models
    print(f"\n🚀 OPTIMIZATION TEST:")
    print("-" * 40)
    
    target_reach = 45
    budget = 5_000_000
    
    print(f"Target: {target_reach}% reach with ₺{budget:,} budget")
    
    # Standard optimization
    standard_optimal, standard_opt_results = optimize_media_plan(
        target_reach, budget, use_sainsbury=False
    )
    
    if standard_optimal:
        print(f"✅ Standard optimization: {standard_opt_results['net_reach_pct']:.1f}% achieved")
    
    # Sainsbury optimization
    sainsbury_optimal, sainsbury_opt_results = optimize_media_plan(
        target_reach, budget, use_sainsbury=True
    )
    
    if sainsbury_optimal:
        print(f"✅ Sainsbury optimization: {sainsbury_opt_results['quality_reach_pct']:.1f}% quality reach achieved")
        print(f"   Net reach: {sainsbury_opt_results['net_reach_pct']:.1f}%")

if __name__ == "__main__":
    test_calculation_system() 