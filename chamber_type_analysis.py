import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_chamber_type_timing(input_file='Book1_processed.csv'):
    """
    Analyze average time and variation spent in each chamber type by TRACK_RCP.
    Creates CHAMBER_TYPE column and calculates processing time statistics.
    """
    
    # Read the processed data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows from {input_file}")
    
    # Convert date columns to datetime
    df['START_DATE'] = pd.to_datetime(df['START_DATE'])
    df['END_DATE'] = pd.to_datetime(df['END_DATE'])
    
    # Create CHAMBER_TYPE column (first 3 characters of CHAMBER)
    df['CHAMBER_TYPE'] = df['CHAMBER'].str[:3]
    
    # Remove rows with invalid CHAMBER_TYPE (NaN values)
    df = df.dropna(subset=['CHAMBER_TYPE'])
    
    print(f"After cleaning: {len(df):,} rows")
    
    # Calculate processing time in minutes for each wafer in each chamber
    df['PROCESSING_TIME_MINUTES'] = (df['END_DATE'] - df['START_DATE']).dt.total_seconds() / 60
    
    # Calculate total time (processing + wait) if CHAMBER_WAIT_DURATION exists
    if 'CHAMBER_WAIT_DURATION' in df.columns:
        df['TOTAL_TIME_MINUTES'] = df['PROCESSING_TIME_MINUTES'] + df['CHAMBER_WAIT_DURATION']
        print("✓ Including wait duration analysis...")
        has_wait_data = True
    else:
        df['TOTAL_TIME_MINUTES'] = df['PROCESSING_TIME_MINUTES']
        print("⚠ No wait duration data found, using processing time only...")
        has_wait_data = False
    
    print(f"\nCreated CHAMBER_TYPE column")
    print(f"Unique CHAMBER_TYPEs: {sorted(df['CHAMBER_TYPE'].unique())}")
    print(f"Unique TRACK_RCPs: {df['TRACK_RCP'].nunique()}")
    
    # Analyze by TRACK_RCP and CHAMBER_TYPE
    chamber_analysis = df.groupby(['TRACK_RCP', 'CHAMBER_TYPE'])['PROCESSING_TIME_MINUTES'].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(2)
    chamber_analysis.columns = ['Count', 'Mean_Minutes', 'Std_Minutes', 'Min_Minutes', 'Max_Minutes', 'Median_Minutes']
    chamber_analysis = chamber_analysis.reset_index()
    
    # Add wait time analysis if available
    if has_wait_data:
        wait_analysis = df.groupby(['TRACK_RCP', 'CHAMBER_TYPE'])['CHAMBER_WAIT_DURATION'].agg([
            'mean', 'std', 'min', 'max', 'median'
        ]).round(2)
        wait_analysis.columns = ['Wait_Mean_Minutes', 'Wait_Std_Minutes', 'Wait_Min_Minutes', 'Wait_Max_Minutes', 'Wait_Median_Minutes']
        wait_analysis = wait_analysis.reset_index()
        
        # Merge wait analysis with chamber analysis
        chamber_analysis = chamber_analysis.merge(wait_analysis, on=['TRACK_RCP', 'CHAMBER_TYPE'])
        
        # Add total time analysis
        total_analysis = df.groupby(['TRACK_RCP', 'CHAMBER_TYPE'])['TOTAL_TIME_MINUTES'].agg([
            'mean', 'std', 'min', 'max', 'median'
        ]).round(2)
        total_analysis.columns = ['Total_Mean_Minutes', 'Total_Std_Minutes', 'Total_Min_Minutes', 'Total_Max_Minutes', 'Total_Median_Minutes']
        total_analysis = total_analysis.reset_index()
        
        chamber_analysis = chamber_analysis.merge(total_analysis, on=['TRACK_RCP', 'CHAMBER_TYPE'])
    
    # Calculate coefficient of variation (CV) for better comparison
    chamber_analysis['CV_Percent'] = (chamber_analysis['Std_Minutes'] / chamber_analysis['Mean_Minutes'] * 100).round(1)
    
    print("\n" + "="*100)
    if has_wait_data:
        print("COMPREHENSIVE CHAMBER ANALYSIS (Processing + Wait Times) BY TRACK_RCP")
    else:
        print("CHAMBER TYPE TIMING ANALYSIS BY TRACK_RCP")
    print("="*100)
    
    # Display results for each TRACK_RCP
    for track_rcp in sorted(df['TRACK_RCP'].unique()):
        print(f"\n{track_rcp}")
        print("-" * len(track_rcp))
        
        rcp_data = chamber_analysis[chamber_analysis['TRACK_RCP'] == track_rcp].copy()
        if len(rcp_data) > 0:
            rcp_data = rcp_data.sort_values('Mean_Minutes', ascending=False)
            
            for _, row in rcp_data.iterrows():
                if has_wait_data:
                    print(f"  {row['CHAMBER_TYPE']:8s} | Proc: {row['Mean_Minutes']:5.1f} min | "
                          f"Wait: {row['Wait_Mean_Minutes']:5.1f} min | Total: {row['Total_Mean_Minutes']:6.1f} min | "
                          f"Std: {row['Std_Minutes']:4.1f} | Count: {row['Count']:4.0f} | "
                          f"Range: {row['Min_Minutes']:4.1f}-{row['Max_Minutes']:4.1f}")
                else:
                    print(f"  {row['CHAMBER_TYPE']:8s} | Avg: {row['Mean_Minutes']:6.1f} min | "
                          f"Std: {row['Std_Minutes']:5.1f} | CV: {row['CV_Percent']:4.1f}% | "
                          f"Count: {row['Count']:4.0f} | Range: {row['Min_Minutes']:4.1f}-{row['Max_Minutes']:4.1f}")
    
    # Analyze total processing time per lot
    print("\n" + "="*100)
    print("TOTAL PROCESSING TIME ANALYSIS BY TRACK_RCP")
    print("="*100)
    
    # Calculate total time per LOT for each TRACK_RCP
    lot_analysis = df.groupby(['TRACK_RCP', 'LOT']).agg({
        'PROCESSING_TIME_MINUTES': 'sum',
        'WAFERID': 'nunique'  # Count unique wafers
    }).reset_index()
    lot_analysis.columns = ['TRACK_RCP', 'LOT', 'Total_Minutes', 'Wafer_Count']
    
    # Calculate time span (delta between min and max times) per LOT and INTRO_DATE combination
    lot_time_span = df.groupby(['TRACK_RCP', 'LOT', 'INTRO_DATE']).agg({
        'START_DATE': ['min', 'max'],
        'END_DATE': ['min', 'max'],
        'WAFERID': 'nunique'
    }).reset_index()
    
    # Flatten column names
    lot_time_span.columns = ['TRACK_RCP', 'LOT', 'INTRO_DATE', 'First_Start', 'Last_Start', 
                            'First_End', 'Last_End', 'Wafer_Count']
    
    # Calculate the time span from first start to last end (total lot duration)
    lot_time_span['Lot_Span_Minutes'] = (
        lot_time_span['Last_End'] - lot_time_span['First_Start']
    ).dt.total_seconds() / 60
    
    # Filter for lots with 25 wafers (as mentioned in the question)
    lot_25_wafers = lot_analysis[lot_analysis['Wafer_Count'] == 25].copy()
    lot_25_wafers_span = lot_time_span[lot_time_span['Wafer_Count'] == 25].copy()
    
    if len(lot_25_wafers) > 0:
        print(f"\nProcessing Time Analysis for lots with exactly 25 wafers:")
        lot_25_stats = lot_25_wafers.groupby('TRACK_RCP')['Total_Minutes'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(1)
        lot_25_stats.columns = ['Lot_Count', 'Mean_Total_Min', 'Std_Total_Min', 'Min_Total_Min', 'Max_Total_Min', 'Median_Total_Min']
        lot_25_stats['CV_Percent'] = (lot_25_stats['Std_Total_Min'] / lot_25_stats['Mean_Total_Min'] * 100).round(1)
        
        print(f"\nLot Span Analysis (First Start to Last End) for lots with exactly 25 wafers:")
        lot_25_span_stats = lot_25_wafers_span.groupby('TRACK_RCP')['Lot_Span_Minutes'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(1)
        lot_25_span_stats.columns = ['Lot_Count', 'Mean_Span_Min', 'Std_Span_Min', 'Min_Span_Min', 'Max_Span_Min', 'Median_Span_Min']
        lot_25_span_stats['CV_Percent'] = (lot_25_span_stats['Std_Span_Min'] / lot_25_span_stats['Mean_Span_Min'] * 100).round(1)
        
        for track_rcp, stats in lot_25_stats.iterrows():
            span_stats = lot_25_span_stats.loc[track_rcp] if track_rcp in lot_25_span_stats.index else None
            print(f"\n{track_rcp}")
            print(f"  Lots analyzed: {stats['Lot_Count']:3.0f}")
            print(f"  Average processing time: {stats['Mean_Total_Min']:6.1f} minutes ({stats['Mean_Total_Min']/60:.1f} hours)")
            print(f"  Processing time std dev: {stats['Std_Total_Min']:6.1f} minutes (CV: {stats['CV_Percent']:4.1f}%)")
            if span_stats is not None:
                print(f"  Average lot span (first→last): {span_stats['Mean_Span_Min']:6.1f} minutes ({span_stats['Mean_Span_Min']/60:.1f} hours)")
                print(f"  Lot span std dev: {span_stats['Std_Span_Min']:6.1f} minutes (CV: {span_stats['CV_Percent']:4.1f}%)")
                print(f"  Span range: {span_stats['Min_Span_Min']:5.1f} - {span_stats['Max_Span_Min']:5.1f} minutes")
    else:
        print("\nNo lots found with exactly 25 wafers. Analyzing all lot sizes:")
        lot_stats = lot_analysis.groupby('TRACK_RCP').agg({
            'Total_Minutes': ['count', 'mean', 'std', 'min', 'max'],
            'Wafer_Count': ['mean', 'min', 'max']
        }).round(1)
        print(lot_stats)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Determine layout based on whether wait data is available
    if has_wait_data:
        # 2x3 layout for comprehensive analysis including wait times
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    else:
        # 2x2 layout for processing time only
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Median processing time by chamber type
    chamber_summary = chamber_analysis.groupby('CHAMBER_TYPE')['Median_Minutes'].median().sort_values(ascending=False)
    bars1 = axes[0,0].bar(chamber_summary.index, chamber_summary.values, color='steelblue', alpha=0.8)
    axes[0,0].set_title('Median Processing Time by Chamber Type', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Chamber Type', fontweight='bold')
    axes[0,0].set_ylabel('Median Minutes per Wafer', fontweight='bold')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                      f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Standard deviation by chamber type
    std_summary = chamber_analysis.groupby('CHAMBER_TYPE')['Std_Minutes'].mean().sort_values(ascending=False)
    bars2 = axes[0,1].bar(std_summary.index, std_summary.values, color='lightcoral', alpha=0.8)
    axes[0,1].set_title('Processing Time Variation by Chamber Type', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Chamber Type', fontweight='bold')
    axes[0,1].set_ylabel('Standard Deviation (Minutes)', fontweight='bold')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                      f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    if has_wait_data:
        # Plot 3: Wait time analysis
        wait_summary = chamber_analysis.groupby('CHAMBER_TYPE')['Wait_Mean_Minutes'].mean().sort_values(ascending=False)
        bars3 = axes[0,2].bar(wait_summary.index, wait_summary.values, color='orange', alpha=0.8)
        axes[0,2].set_title('Average Wait Time by Chamber Type', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('Chamber Type', fontweight='bold')
        axes[0,2].set_ylabel('Average Wait Minutes', fontweight='bold')
        axes[0,2].tick_params(axis='x', rotation=45)
        axes[0,2].grid(True, alpha=0.3)
        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            axes[0,2].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                          f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Total time (processing + wait)
        total_summary = chamber_analysis.groupby('CHAMBER_TYPE')['Total_Mean_Minutes'].mean().sort_values(ascending=False)
        bars4 = axes[1,0].bar(total_summary.index, total_summary.values, color='purple', alpha=0.8)
        axes[1,0].set_title('Total Time by Chamber Type (Processing + Wait)', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Chamber Type', fontweight='bold')
        axes[1,0].set_ylabel('Total Minutes per Wafer', fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        # Add value labels on bars
        for bar in bars4:
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                          f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 5: Wait vs Processing time comparison
        chamber_types = sorted(chamber_analysis['CHAMBER_TYPE'].unique())
        processing_times = [chamber_analysis[chamber_analysis['CHAMBER_TYPE']==ct]['Mean_Minutes'].mean() for ct in chamber_types]
        wait_times = [chamber_analysis[chamber_analysis['CHAMBER_TYPE']==ct]['Wait_Mean_Minutes'].mean() for ct in chamber_types]
        
        x = np.arange(len(chamber_types))
        width = 0.35
        
        bars_proc = axes[1,1].bar(x - width/2, processing_times, width, label='Processing Time', color='steelblue', alpha=0.8)
        bars_wait = axes[1,1].bar(x + width/2, wait_times, width, label='Wait Time', color='orange', alpha=0.8)
        
        axes[1,1].set_title('Processing vs Wait Time by Chamber Type', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Chamber Type', fontweight='bold')
        axes[1,1].set_ylabel('Minutes per Wafer', fontweight='bold')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(chamber_types, rotation=45)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Efficiency ratio (Processing / Total Time)
        efficiency_data = []
        for ct in chamber_types:
            ct_data = chamber_analysis[chamber_analysis['CHAMBER_TYPE']==ct]
            if len(ct_data) > 0:
                avg_proc = ct_data['Mean_Minutes'].mean()
                avg_total = ct_data['Total_Mean_Minutes'].mean()
                efficiency = (avg_proc / avg_total * 100) if avg_total > 0 else 0
                efficiency_data.append(efficiency)
            else:
                efficiency_data.append(0)
        
        bars6 = axes[1,2].bar(chamber_types, efficiency_data, color='green', alpha=0.8)
        axes[1,2].set_title('Chamber Efficiency (Processing/Total Time %)', fontsize=14, fontweight='bold')
        axes[1,2].set_xlabel('Chamber Type', fontweight='bold')
        axes[1,2].set_ylabel('Efficiency Percentage', fontweight='bold')
        axes[1,2].tick_params(axis='x', rotation=45)
        axes[1,2].grid(True, alpha=0.3)
        axes[1,2].set_ylim(0, 100)
        # Add value labels on bars
        for bar in bars6:
            height = bar.get_height()
            axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 1,
                          f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    else:
        # Original plots for processing time only analysis
        # Plot 3: Processing time distribution for top chamber types
        top_chambers = chamber_summary.head(6).index
        chamber_data = df[df['CHAMBER_TYPE'].isin(top_chambers)]
        axes[1,0].boxplot([chamber_data[chamber_data['CHAMBER_TYPE']==ct]['PROCESSING_TIME_MINUTES'].values 
                          for ct in top_chambers], labels=top_chambers)
        axes[1,0].set_title('Processing Time Distribution (Top 6 Chamber Types)')
        axes[1,0].set_xlabel('Chamber Type')
        axes[1,0].set_ylabel('Processing Time (Minutes)')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # Common plots for both cases
    
    # Plot 4: Lot time span variation by TRACK_RCP (time delta from first start to last end)
    if len(lot_25_wafers_span) > 0:
        # Show all TRACK_RCP values instead of just top 6
        track_rcps = sorted(lot_25_wafers_span['TRACK_RCP'].unique())
        span_data = lot_25_wafers_span[lot_25_wafers_span['TRACK_RCP'].isin(track_rcps)]
        
        box_data = []
        labels = []
        for rcp in track_rcps:
            rcp_span_data = span_data[span_data['TRACK_RCP'] == rcp]['Lot_Span_Minutes'].values
            if len(rcp_span_data) > 0:
                box_data.append(rcp_span_data)
                # Create shorter labels for better readability
                short_label = rcp.split('/')[-1].replace('-', '-\n')[:20]  # Line break for long names
                labels.append(short_label)
        
        if box_data:
            # Increase figure size to accommodate all TRACK_RCP values
            fig.set_size_inches(18, 12)
            axes[1,1].boxplot(box_data, tick_labels=labels)
            axes[1,1].set_title('Lot Time Span Variation (First Start → Last End)\nAll TRACK_RCP Values')
            axes[1,1].set_xlabel('TRACK_RCP (shortened)')
            axes[1,1].set_ylabel('Lot Span Minutes (First→Last)')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True, alpha=0.3)
    else:
        # Fallback to original total time analysis if no span data
        if len(lot_analysis) > 0:
            track_rcps = lot_analysis['TRACK_RCP'].value_counts().head(6).index
            lot_data = lot_analysis[lot_analysis['TRACK_RCP'].isin(track_rcps)]
            
            box_data = []
            labels = []
            for rcp in track_rcps:
                rcp_data = lot_data[lot_data['TRACK_RCP'] == rcp]['Total_Minutes'].values
                if len(rcp_data) > 0:
                    box_data.append(rcp_data)
                    labels.append(rcp.split('/')[-1][:15])  # Shortened labels
            
            if box_data:
                axes[1,1].boxplot(box_data, tick_labels=labels)
                axes[1,1].set_title('Total Lot Processing Time Variation')
                axes[1,1].set_xlabel('TRACK_RCP (shortened)')
                axes[1,1].set_ylabel('Total Minutes per Lot')
                axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plots
    output_file = 'chamber_type_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as: {output_file}")
    
    # Save detailed analysis to CSV
    chamber_analysis_file = 'chamber_type_timing_analysis.csv'
    chamber_analysis.to_csv(chamber_analysis_file, index=False)
    print(f"Detailed analysis saved as: {chamber_analysis_file}")
    
    if len(lot_25_wafers) > 0:
        lot_analysis_file = 'lot_25_wafer_analysis.csv'
        lot_25_wafers.to_csv(lot_analysis_file, index=False)
        print(f"25-wafer lot analysis saved as: {lot_analysis_file}")
        
        lot_span_analysis_file = 'lot_25_wafer_span_analysis.csv'
        lot_25_wafers_span.to_csv(lot_span_analysis_file, index=False)
        print(f"25-wafer lot span analysis saved as: {lot_span_analysis_file}")
    
    plt.show()
    
    return chamber_analysis, lot_analysis, lot_time_span

if __name__ == "__main__":
    chamber_stats, lot_stats, lot_span_stats = analyze_chamber_type_timing('Book1_processed.csv')