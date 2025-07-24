#!/usr/bin/env python3
"""
Sainsbury vs Standard Model Comparison
====================================

This script demonstrates the differences between the Standard and Sainsbury 
weighted models in the v3.0 system.
"""

from reach_calc_v3 import calculate_campaign_kpis, optimize_media_plan

def compare_models():
    print("🔄 SAINSBURY vs STANDARD MODEL COMPARISON")
    print("=" * 60)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'TV-Heavy Campaign',
            'spend': {
                'TV_LF': 3_000_000,
                'YT_LF': 500_000,
                'YT_SF': 300_000,
                'Meta_SF': 500_000,
                'TT_SF': 200_000
            }
        },
        {
            'name': 'Digital-Heavy Campaign',
            'spend': {
                'TV_LF': 500_000,
                'YT_LF': 1_000_000,
                'YT_SF': 1_000_000,
                'Meta_SF': 1_500_000,
                'TT_SF': 500_000
            }
        },
        {
            'name': 'Balanced Campaign',
            'spend': {
                'TV_LF': 1_500_000,
                'YT_LF': 1_000_000,
                'YT_SF': 800_000,
                'Meta_SF': 1_000_000,
                'TT_SF': 700_000
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 {scenario['name'].upper()}")
        print("-" * 50)
        
        spend_dict = scenario['spend']
        total_spend = sum(spend_dict.values())
        
        print(f"Total Budget: ₺{total_spend:,}")
        print(f"Spend Mix:")
        for channel, spend in spend_dict.items():
            pct = spend / total_spend * 100
            print(f"  {channel}: ₺{spend:,} ({pct:.1f}%)")
        
        # Calculate with both models
        standard_results = calculate_campaign_kpis(spend_dict, use_sainsbury=False)
        sainsbury_results = calculate_campaign_kpis(spend_dict, use_sainsbury=True)
        
        print(f"\n📈 RESULTS COMPARISON:")
        print(f"{'Metric':<20} {'Standard':<15} {'Sainsbury':<15} {'Difference':<15}")
        print("-" * 70)
        
        # Net Reach
        std_net = standard_results['net_reach_pct']
        sai_net = sainsbury_results['net_reach_pct']
        net_diff = sai_net - std_net
        print(f"{'Net Reach':<20} {std_net:<15.1f}% {sai_net:<15.1f}% {net_diff:+.1f}pp")
        
        # Quality Reach (only for Sainsbury)
        sai_quality = sainsbury_results['quality_reach_pct']
        quality_vs_net = sai_quality - sai_net
        print(f"{'Quality Reach':<20} {'N/A':<15} {sai_quality:<15.1f}% {quality_vs_net:+.1f}pp vs net")
        
        # Impressions
        std_imp = standard_results['total_impressions']
        sai_imp = sainsbury_results['total_impressions']
        imp_diff = (sai_imp - std_imp) / std_imp * 100 if std_imp > 0 else 0
        print(f"{'Impressions':<20} {std_imp/1_000_000:<15.1f}M {sai_imp/1_000_000:<15.1f}M {imp_diff:+.1f}%")
        
        # Efficiency metrics
        std_eff = std_net / (total_spend / 1_000_000)
        sai_eff = sai_quality / (total_spend / 1_000_000)
        eff_diff = sai_eff - std_eff
        print(f"{'Reach per ₺1M':<20} {std_eff:<15.1f}% {sai_eff:<15.1f}% {eff_diff:+.1f}%")
        
        # Key insights
        print(f"\n🔍 KEY INSIGHTS:")
        if quality_vs_net < -5:
            print(f"  • Sainsbury model shows {abs(quality_vs_net):.1f}pp quality penalty")
            print(f"  • Suggests low-quality channel mix or excessive spread")
        elif quality_vs_net > 2:
            print(f"  • Sainsbury model shows {quality_vs_net:.1f}pp quality bonus")
            print(f"  • Suggests high-quality, focused channel strategy")
        else:
            print(f"  • Quality vs net reach difference is minimal ({quality_vs_net:+.1f}pp)")
        
        # Format analysis
        lf_spend = spend_dict.get('TV_LF', 0) + spend_dict.get('YT_LF', 0)
        sf_spend = total_spend - lf_spend
        lf_pct = lf_spend / total_spend * 100
        
        if lf_pct > 60:
            print(f"  • Long-form heavy ({lf_pct:.1f}%) - benefits from Sainsbury weighting")
        elif lf_pct < 30:
            print(f"  • Short-form heavy ({lf_pct:.1f}%) - penalized by Sainsbury model")
        else:
            print(f"  • Balanced format mix ({lf_pct:.1f}% long-form)")
    
    # Optimization comparison
    print(f"\n🚀 OPTIMIZATION COMPARISON")
    print("=" * 60)
    
    target_reach = 50
    budget = 5_000_000
    
    print(f"Target: {target_reach}% reach with ₺{budget:,} budget")
    
    # Standard optimization
    print(f"\n📊 STANDARD MODEL OPTIMIZATION:")
    std_optimal, std_results = optimize_media_plan(target_reach, budget, use_sainsbury=False)
    
    if std_optimal:
        print(f"✅ Achieved: {std_results['net_reach_pct']:.1f}% net reach")
        print(f"Budget used: ₺{std_results['total_spend']:,.0f}")
        print(f"Top allocations:")
        
        sorted_spend = sorted(std_optimal.items(), key=lambda x: x[1], reverse=True)
        for channel, spend in sorted_spend[:3]:
            if spend > 50000:
                pct = spend / std_results['total_spend'] * 100
                print(f"  {channel}: ₺{spend:,.0f} ({pct:.1f}%)")
    
    # Sainsbury optimization
    print(f"\n🧠 SAINSBURY MODEL OPTIMIZATION:")
    sai_optimal, sai_results = optimize_media_plan(target_reach, budget, use_sainsbury=True)
    
    if sai_optimal:
        print(f"✅ Achieved: {sai_results['quality_reach_pct']:.1f}% quality reach")
        print(f"Net reach: {sai_results['net_reach_pct']:.1f}%")
        print(f"Budget used: ₺{sai_results['total_spend']:,.0f}")
        print(f"Top allocations:")
        
        sorted_spend = sorted(sai_optimal.items(), key=lambda x: x[1], reverse=True)
        for channel, spend in sorted_spend[:3]:
            if spend > 50000:
                pct = spend / sai_results['total_spend'] * 100
                print(f"  {channel}: ₺{spend:,.0f} ({pct:.1f}%)")
    
    # Strategy comparison
    if std_optimal and sai_optimal:
        print(f"\n📋 STRATEGY COMPARISON:")
        print(f"{'Channel':<12} {'Standard':<15} {'Sainsbury':<15} {'Difference':<15}")
        print("-" * 60)
        
        all_channels = set(std_optimal.keys()) | set(sai_optimal.keys())
        for channel in sorted(all_channels):
            std_spend = std_optimal.get(channel, 0)
            sai_spend = sai_optimal.get(channel, 0)
            diff = sai_spend - std_spend
            
            if std_spend > 10000 or sai_spend > 10000:
                print(f"{channel:<12} ₺{std_spend:<14,.0f} ₺{sai_spend:<14,.0f} {diff:+,.0f}")
        
        print(f"\n🎯 STRATEGIC INSIGHTS:")
        
        # Long-form vs short-form preference
        std_lf = std_optimal.get('TV_LF', 0) + std_optimal.get('YT_LF', 0)
        sai_lf = sai_optimal.get('TV_LF', 0) + sai_optimal.get('YT_LF', 0)
        
        if sai_lf > std_lf * 1.2:
            print(f"  • Sainsbury model prefers long-form content (+₺{sai_lf - std_lf:,.0f})")
        elif sai_lf < std_lf * 0.8:
            print(f"  • Standard model allocates more to long-form (+₺{std_lf - sai_lf:,.0f})")
        
        # Platform concentration
        std_platforms = len([s for s in std_optimal.values() if s > 100000])
        sai_platforms = len([s for s in sai_optimal.values() if s > 100000])
        
        if sai_platforms < std_platforms:
            print(f"  • Sainsbury model is more concentrated ({sai_platforms} vs {std_platforms} major channels)")
        elif sai_platforms > std_platforms:
            print(f"  • Standard model is more concentrated ({std_platforms} vs {sai_platforms} major channels)")
    
    print(f"\n✨ SUMMARY:")
    print(f"• Standard model: Fast, reliable, mathematically consistent")
    print(f"• Sainsbury model: Sophisticated quality weighting, format hierarchy")
    print(f"• Use Standard for: Budget planning, quick analysis, consistent results")
    print(f"• Use Sainsbury for: Quality insights, format strategy, attention-based optimization")

if __name__ == "__main__":
    compare_models() 