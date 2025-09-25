import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the processed data"""
    print("Loading processed data...")
    
    # Read the processed CSV file
    df = pd.read_csv('Book1_processed.csv')
    print(f"Loaded {len(df):,} rows from Book1_processed.csv")
    
    # Convert time columns to datetime
    time_cols = ['INTRO_DATE', 'START_DATE', 'END_DATE']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate processing time in minutes
    df['PROCESSING_TIME_MINUTES'] = (df['END_DATE'] - df['START_DATE']).dt.total_seconds() / 60
    
    # Create chamber type from first 3 characters of CHAMBER
    df['CHAMBER_TYPE'] = df['CHAMBER'].str[:3]
    
    print(f"Unique TRACK_RCPs: {df['TRACK_RCP'].nunique()}")
    print(f"TRACK_RCP recipes: {sorted(df['TRACK_RCP'].unique())}")
    
    return df

def calculate_recipe_stats(df, recipe):
    """Calculate comprehensive statistics for a specific TRACK_RCP"""
    recipe_data = df[df['TRACK_RCP'] == recipe].copy()
    
    # Chamber type analysis
    chamber_stats = recipe_data.groupby('CHAMBER_TYPE').agg({
        'PROCESSING_TIME_MINUTES': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'CHAMBER': 'nunique'
    }).round(2)
    
    chamber_stats.columns = ['Count', 'Mean_Minutes', 'Median_Minutes', 'Std_Minutes', 
                           'Min_Minutes', 'Max_Minutes', 'Unique_Chambers']
    chamber_stats = chamber_stats.sort_values('Mean_Minutes', ascending=False)
    
    # Individual chamber analysis
    chamber_detail = recipe_data.groupby('CHAMBER').agg({
        'PROCESSING_TIME_MINUTES': ['count', 'mean', 'median', 'std'],
    }).round(2)
    chamber_detail.columns = ['Count', 'Mean_Minutes', 'Median_Minutes', 'Std_Minutes']
    chamber_detail['CHAMBER_TYPE'] = chamber_detail.index.str[:3]
    chamber_detail = chamber_detail.sort_values('Mean_Minutes', ascending=False)
    
    # Lot-level analysis (for 25-wafer lots)
    lot_analysis = recipe_data[recipe_data.groupby('LOT')['LOT'].transform('count') == 25].copy()
    if len(lot_analysis) > 0:
        lot_stats = lot_analysis.groupby('LOT').agg({
            'PROCESSING_TIME_MINUTES': 'sum',
            'START_DATE': 'min',
            'END_DATE': 'max'
        })
        lot_stats['Total_Hours'] = lot_stats['PROCESSING_TIME_MINUTES'] / 60
        lot_stats['Span_Minutes'] = (lot_stats['END_DATE'] - lot_stats['START_DATE']).dt.total_seconds() / 60
        lot_stats['Span_Hours'] = lot_stats['Span_Minutes'] / 60
    else:
        lot_stats = pd.DataFrame()
    
    return chamber_stats, chamber_detail, lot_stats, recipe_data

def create_optimization_analysis(df, recipe, chamber_stats, chamber_detail, lot_stats, recipe_data, output_dir):
    """Create comprehensive optimization analysis for a specific recipe"""
    
    # Create output directory for this recipe
    safe_recipe_name = recipe.replace('/', '_').replace('{', '').replace('}', '').replace(':', '_').replace(';', '_').replace('=', '_')
    recipe_dir = os.path.join(output_dir, safe_recipe_name)
    os.makedirs(recipe_dir, exist_ok=True)
    
    # Create comprehensive figure with 6 subplots
    fig = plt.figure(figsize=(24, 16))
    
    # 1. Chamber Type Performance (Top Left)
    ax1 = plt.subplot(2, 3, 1)
    chamber_means = chamber_stats['Mean_Minutes'].sort_values(ascending=True)
    bars1 = ax1.barh(chamber_means.index, chamber_means.values, color='steelblue', alpha=0.8)
    ax1.set_title(f'Average Processing Time by Chamber Type\n{recipe}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Average Minutes per Wafer', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    # Add value labels
    for i, (chamber, value) in enumerate(chamber_means.items()):
        ax1.text(value + 0.1, i, f'{value:.1f}', va='center', fontweight='bold')
    
    # 2. Chamber Type Variability (Top Center)
    ax2 = plt.subplot(2, 3, 2)
    std_data = chamber_stats['Std_Minutes'].sort_values(ascending=False)
    bars2 = ax2.bar(std_data.index, std_data.values, color='coral', alpha=0.8)
    ax2.set_title('Processing Time Variation by Chamber Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Standard Deviation (Minutes)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Individual Chamber Performance (Top Right)
    ax3 = plt.subplot(2, 3, 3)
    # Show top 15 chambers by processing time
    top_chambers = chamber_detail.head(15)
    colors = ['red' if std > 1.0 else 'orange' if std > 0.5 else 'green' for std in top_chambers['Std_Minutes']]
    bars3 = ax3.barh(range(len(top_chambers)), top_chambers['Mean_Minutes'], color=colors, alpha=0.7)
    ax3.set_yticks(range(len(top_chambers)))
    ax3.set_yticklabels(top_chambers.index, fontsize=10)
    ax3.set_title('Top 15 Slowest Individual Chambers', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Average Minutes per Wafer', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='High Variation (σ>1.0 min)'),
                      plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.7, label='Medium Variation (σ>0.5 min)'),
                      plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.7, label='Low Variation (σ≤0.5 min)')]
    ax3.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # 4. Chamber Utilization (Bottom Left)
    ax4 = plt.subplot(2, 3, 4)
    utilization = chamber_stats['Count'].sort_values(ascending=False)
    bars4 = ax4.bar(utilization.index, utilization.values, color='lightgreen', alpha=0.8)
    ax4.set_title('Chamber Type Utilization (Wafer Count)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Wafers Processed', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(utilization)*0.01,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Processing Time Distribution (Bottom Center)
    ax5 = plt.subplot(2, 3, 5)
    # Box plot for each chamber type
    chamber_types = sorted(recipe_data['CHAMBER_TYPE'].unique())
    box_data = [recipe_data[recipe_data['CHAMBER_TYPE']==ct]['PROCESSING_TIME_MINUTES'].values 
                for ct in chamber_types]
    bp = ax5.boxplot(box_data, labels=chamber_types, patch_artist=True)
    # Color boxes by average time
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(chamber_types)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax5.set_title('Processing Time Distribution by Chamber Type', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Processing Time (Minutes)', fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Lot Analysis (Bottom Right)
    ax6 = plt.subplot(2, 3, 6)
    if len(lot_stats) > 0:
        # Scatter plot of total processing time vs span time
        scatter = ax6.scatter(lot_stats['Span_Hours'], lot_stats['Total_Hours'], 
                            alpha=0.7, s=60, c='purple')
        ax6.set_xlabel('Lot Span Time (Hours)', fontweight='bold')
        ax6.set_ylabel('Total Processing Time (Hours)', fontweight='bold')
        ax6.set_title(f'Lot Efficiency Analysis\n({len(lot_stats)} lots with 25 wafers)', 
                     fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Add trend line
        if len(lot_stats) > 1:
            z = np.polyfit(lot_stats['Span_Hours'], lot_stats['Total_Hours'], 1)
            p = np.poly1d(z)
            ax6.plot(lot_stats['Span_Hours'], p(lot_stats['Span_Hours']), "r--", alpha=0.8)
        
        # Add statistics text
        avg_span = lot_stats['Span_Hours'].mean()
        avg_total = lot_stats['Total_Hours'].mean()
        efficiency = (avg_total / avg_span) if avg_span > 0 else 0
        ax6.text(0.05, 0.95, f'Avg Span: {avg_span:.1f}h\nAvg Total: {avg_total:.1f}h\nEfficiency: {efficiency:.1f}x', 
                transform=ax6.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax6.text(0.5, 0.5, 'No 25-wafer lots found\nfor this recipe', 
                ha='center', va='center', transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax6.set_title('Lot Analysis', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the comprehensive analysis figure
    fig_path = os.path.join(recipe_dir, f'{safe_recipe_name}_optimization_analysis.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return recipe_dir, fig_path

def generate_optimization_recommendations(chamber_stats, chamber_detail, lot_stats, recipe):
    """Generate specific optimization recommendations for a recipe"""
    recommendations = []
    
    # 1. Bottleneck Analysis
    slowest_chamber_type = chamber_stats.index[0]
    slowest_time = chamber_stats.loc[slowest_chamber_type, 'Mean_Minutes']
    fastest_chamber_type = chamber_stats.index[-1]
    fastest_time = chamber_stats.loc[fastest_chamber_type, 'Mean_Minutes']
    
    recommendations.append(f"🚩 BOTTLENECK: {slowest_chamber_type} chambers are the primary bottleneck")
    recommendations.append(f"   └─ Average time: {slowest_time:.1f} minutes vs {fastest_time:.1f} minutes for {fastest_chamber_type}")
    recommendations.append(f"   └─ Improvement potential: {((slowest_time - fastest_time) / slowest_time * 100):.1f}% time reduction possible")
    
    # 2. Variability Analysis
    high_variability = chamber_stats[chamber_stats['Std_Minutes'] > 1.0]
    if len(high_variability) > 0:
        recommendations.append(f"\n⚠️  HIGH VARIATION CHAMBERS (>1.0 min std dev):")
        for chamber_type, stats in high_variability.iterrows():
            recommendations.append(f"   └─ {chamber_type}: {stats['Std_Minutes']:.2f} min std dev (Range: {stats['Min_Minutes']:.1f}-{stats['Max_Minutes']:.1f} min)")
    
    # 3. Individual Chamber Issues
    problem_chambers = chamber_detail[chamber_detail['Std_Minutes'] > 1.0].head(5)
    if len(problem_chambers) > 0:
        recommendations.append(f"\n🔧 INDIVIDUAL CHAMBERS WITH HIGH VARIATION (>1.0 min std dev):")
        for chamber, stats in problem_chambers.iterrows():
            recommendations.append(f"   └─ {chamber}: {stats['Mean_Minutes']:.1f} min avg, {stats['Std_Minutes']:.2f} min std dev")
    
    # 4. Utilization Analysis
    underutilized = chamber_stats[chamber_stats['Count'] < chamber_stats['Count'].median() * 0.5]
    if len(underutilized) > 0:
        recommendations.append(f"\n📊 UNDERUTILIZED CHAMBER TYPES:")
        for chamber_type, stats in underutilized.iterrows():
            recommendations.append(f"   └─ {chamber_type}: Only {stats['Count']} wafers processed")
    
    # 5. Lot Efficiency (if data available)
    if len(lot_stats) > 0:
        avg_span = lot_stats['Span_Hours'].mean()
        avg_total = lot_stats['Total_Hours'].mean()
        efficiency = avg_total / avg_span if avg_span > 0 else 0
        
        recommendations.append(f"\n⏱️  LOT PROCESSING EFFICIENCY:")
        recommendations.append(f"   └─ Average span time: {avg_span:.1f} hours")
        recommendations.append(f"   └─ Average total processing: {avg_total:.1f} hours")
        recommendations.append(f"   └─ Efficiency ratio: {efficiency:.1f}x")
        
        if efficiency < 10:
            recommendations.append(f"   └─ 💡 Consider parallel processing optimization")
        
        span_std = lot_stats['Span_Hours'].std()
        if span_std > 0.3:  # More than 18 minutes standard deviation
            recommendations.append(f"   └─ ⚠️  High span time variation: {span_std:.2f} hours std dev")
    
    # 6. Specific Recommendations
    recommendations.append(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    
    if slowest_time > fastest_time * 2:
        recommendations.append(f"   1. Focus on {slowest_chamber_type} chamber optimization - highest impact")
    
    high_std_chambers = chamber_stats[chamber_stats['Std_Minutes'] > 1.0]
    if len(high_std_chambers) > 0:
        recommendations.append(f"   2. Investigate process consistency in {', '.join(high_std_chambers.index)} chambers (>1.0 min std dev)")
    
    if 'TEL' in chamber_stats.index and chamber_stats.loc['TEL', 'Std_Minutes'] > 2.0:
        recommendations.append(f"   3. TEL chamber variation ({chamber_stats.loc['TEL', 'Std_Minutes']:.2f} min std dev) suggests equipment maintenance needs")
    
    recommendations.append(f"   4. Consider load balancing across available chambers")
    recommendations.append(f"   5. Monitor and address chambers with >1.0 min std dev for process consistency")
    
    return recommendations

def main():
    """Main analysis function"""
    print("=== TRACK_RCP OPTIMIZATION ANALYSIS ===\n")
    
    # Load data
    df = load_and_prepare_data()
    
    # Create output directory
    output_dir = 'track_rcp_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all unique TRACK_RCP recipes
    recipes = sorted(df['TRACK_RCP'].unique())
    
    print(f"\nAnalyzing {len(recipes)} TRACK_RCP recipes...")
    print("=" * 80)
    
    # Create summary report
    summary_report = []
    
    for i, recipe in enumerate(recipes, 1):
        print(f"\n[{i}/{len(recipes)}] Analyzing: {recipe}")
        print("-" * 60)
        
        # Calculate statistics for this recipe
        chamber_stats, chamber_detail, lot_stats, recipe_data = calculate_recipe_stats(df, recipe)
        
        # Print summary
        total_wafers = len(recipe_data)
        chamber_types = len(chamber_stats)
        bottleneck = chamber_stats.index[0]
        bottleneck_time = chamber_stats.loc[bottleneck, 'Mean_Minutes']
        
        print(f"Total wafers: {total_wafers:,}")
        print(f"Chamber types: {chamber_types}")
        print(f"Bottleneck: {bottleneck} ({bottleneck_time:.1f} min avg)")
        
        if len(lot_stats) > 0:
            print(f"25-wafer lots: {len(lot_stats)}")
        
        # Create optimization analysis and visualizations
        recipe_dir, fig_path = create_optimization_analysis(
            df, recipe, chamber_stats, chamber_detail, lot_stats, recipe_data, output_dir)
        
        # Generate recommendations
        recommendations = generate_optimization_recommendations(
            chamber_stats, chamber_detail, lot_stats, recipe)
        
        # Save detailed statistics
        stats_file = os.path.join(recipe_dir, f'{recipe.replace("/", "_").replace("{", "").replace("}", "").replace(":", "_").replace(";", "_").replace("=", "_")}_chamber_stats.csv')
        chamber_stats.to_csv(stats_file)
        
        detail_file = os.path.join(recipe_dir, f'{recipe.replace("/", "_").replace("{", "").replace("}", "").replace(":", "_").replace(";", "_").replace("=", "_")}_chamber_detail.csv')
        chamber_detail.to_csv(detail_file)
        
        if len(lot_stats) > 0:
            lot_file = os.path.join(recipe_dir, f'{recipe.replace("/", "_").replace("{", "").replace("}", "").replace(":", "_").replace(";", "_").replace("=", "_")}_lot_stats.csv')
            lot_stats.to_csv(lot_file)
        
        # Save recommendations
        rec_file = os.path.join(recipe_dir, f'{recipe.replace("/", "_").replace("{", "").replace("}", "").replace(":", "_").replace(";", "_").replace("=", "_")}_recommendations.txt')
        with open(rec_file, 'w', encoding='utf-8') as f:
            f.write(f"OPTIMIZATION RECOMMENDATIONS FOR: {recipe}\n")
            f.write("=" * 80 + "\n\n")
            f.write("\n".join(recommendations))
        
        print(f"✓ Analysis saved to: {recipe_dir}")
        
        # Add to summary report
        summary_report.append({
            'Recipe': recipe,
            'Total_Wafers': total_wafers,
            'Chamber_Types': chamber_types,
            'Bottleneck_Chamber': bottleneck,
            'Bottleneck_Time_Min': bottleneck_time,
            'Lots_25_Wafers': len(lot_stats) if len(lot_stats) > 0 else 0,
            'Analysis_Directory': recipe_dir
        })
    
    # Save summary report
    summary_df = pd.DataFrame(summary_report)
    summary_file = os.path.join(output_dir, 'analysis_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n" + "=" * 80)
    print(f"✓ ANALYSIS COMPLETE")
    print(f"✓ {len(recipes)} recipes analyzed")
    print(f"✓ Results saved in: {output_dir}/")
    print(f"✓ Summary report: {summary_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()