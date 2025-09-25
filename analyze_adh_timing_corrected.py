import pandas as pd
from datetime import datetime

def analyze_adh_timing_corrected(input_file='Book1.csv'):
    """
    Analyze ADH chamber timing by LOT and INTRO_DATE combinations.
    
    Steps:
    1. Sort by INTRO_DATE (newest to oldest), then START_DATE (newest to oldest)
    2. For each LOT and INTRO_DATE combination, find the lowest WAF3 number with ADH chamber
    3. For the previous LOT and INTRO_DATE combination, find the highest WAF3 number with ADH chamber
    4. Calculate delta between these times
    """
    
    # Read the data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows")
    
    # Convert date columns to datetime
    df['INTRO_DATE'] = pd.to_datetime(df['INTRO_DATE'])
    df['START_DATE'] = pd.to_datetime(df['START_DATE'])
    df['END_DATE'] = pd.to_datetime(df['END_DATE'])
    
    # Sort by INTRO_DATE (newest to oldest), then START_DATE (newest to oldest)
    df_sorted = df.sort_values(['INTRO_DATE', 'START_DATE'], ascending=[False, False]).copy()
    
    print("\nData sorted by INTRO_DATE (newest first), then START_DATE (newest first)")
    print(f"Date range: {df['INTRO_DATE'].min()} to {df['INTRO_DATE'].max()}")
    
    # Filter for ADH chambers only
    adh_df = df_sorted[df_sorted['CHAMBER'].str.startswith('ADH', na=False)].copy()
    print(f"\nFound {len(adh_df):,} ADH chamber operations")
    
    # Group by LOT and INTRO_DATE
    lot_intro_groups = adh_df.groupby(['LOT', 'INTRO_DATE'])
    
    # For each group, get records with minimum and maximum WAF3
    min_max_waf3_records = []
    for (lot, intro_date), group in lot_intro_groups:
        # Get lowest WAF3 record
        min_waf3_row = group.loc[group['WAF3'].idxmin()].copy()
        min_waf3_row['WAF3_TYPE'] = 'MIN'
        
        # Get highest WAF3 record
        max_waf3_row = group.loc[group['WAF3'].idxmax()].copy()
        max_waf3_row['WAF3_TYPE'] = 'MAX'
        
        min_max_waf3_records.append(min_waf3_row)
        if min_waf3_row.name != max_waf3_row.name:  # Only add max if different from min
            min_max_waf3_records.append(max_waf3_row)
    
    # Create DataFrame from the records
    analysis_df = pd.DataFrame(min_max_waf3_records)
    analysis_df = analysis_df.sort_values(['INTRO_DATE', 'START_DATE'], ascending=[False, False])
    
    print(f"\nFound {len(analysis_df)} records (min/max WAF3 for each LOT+INTRO_DATE combination)")
    
    # Get unique LOT+INTRO_DATE combinations and their min WAF3 records, sorted chronologically
    unique_combinations = analysis_df[analysis_df['WAF3_TYPE'] == 'MIN'].copy()
    unique_combinations = unique_combinations.sort_values('INTRO_DATE').reset_index(drop=True)
    
    print(f"\nAnalyzing {len(unique_combinations)} unique LOT+INTRO_DATE combinations")
    
    # Calculate deltas: current lot's min WAF3 time - previous lot's max WAF3 time
    unique_combinations['PREV_LOT'] = None
    unique_combinations['PREV_MAX_WAF3'] = None
    unique_combinations['PREV_START_TIME'] = None
    unique_combinations['DELTA_MINUTES'] = None
    unique_combinations['DELTA_HOURS'] = None
    
    for i in range(1, len(unique_combinations)):
        current_row = unique_combinations.iloc[i]
        current_lot = current_row['LOT']
        current_intro = current_row['INTRO_DATE']
        
        # Find previous lot+intro combination
        prev_row = unique_combinations.iloc[i-1]
        prev_lot = prev_row['LOT']
        prev_intro = prev_row['INTRO_DATE']
        
        # Find the max WAF3 record for the previous lot+intro combination
        prev_max_records = analysis_df[
            (analysis_df['LOT'] == prev_lot) & 
            (analysis_df['INTRO_DATE'] == prev_intro) & 
            (analysis_df['WAF3_TYPE'] == 'MAX')
        ]
        
        if len(prev_max_records) > 0:
            prev_max_row = prev_max_records.iloc[0]
            
            # Calculate time delta: current min time - previous max time
            time_delta = current_row['START_DATE'] - prev_max_row['START_DATE']
            delta_minutes = time_delta.total_seconds() / 60
            delta_hours = delta_minutes / 60
            
            # Update the row
            unique_combinations.loc[i, 'PREV_LOT'] = prev_lot
            unique_combinations.loc[i, 'PREV_MAX_WAF3'] = prev_max_row['WAF3']
            unique_combinations.loc[i, 'PREV_START_TIME'] = prev_max_row['START_DATE']
            unique_combinations.loc[i, 'DELTA_MINUTES'] = delta_minutes
            unique_combinations.loc[i, 'DELTA_HOURS'] = delta_hours
    
    # Sort back to newest first for display
    result_df = unique_combinations.sort_values(['INTRO_DATE', 'START_DATE'], ascending=[False, False])
    
    # Select and display relevant columns
    display_cols = ['LOT', 'WAF3', 'INTRO_DATE', 'START_DATE', 'CHAMBER', 
                   'PREV_LOT', 'PREV_MAX_WAF3', 'PREV_START_TIME', 'DELTA_MINUTES', 'DELTA_HOURS']
    
    print("\n" + "="*120)
    print("CORRECTED ADH TIMING ANALYSIS RESULTS")
    print("="*120)
    print("Current LOT: MIN WAF3 ADH time - Previous LOT: MAX WAF3 ADH time")
    print("="*120)
    
    for _, row in result_df.iterrows():
        print(f"\nCURRENT LOT: {row['LOT']}, MIN WAF3: {row['WAF3']}")
        print(f"  INTRO: {row['INTRO_DATE']}")
        print(f"  START: {row['START_DATE']}")
        print(f"  CHAMBER: {row['CHAMBER']}")
        if pd.notna(row['PREV_LOT']):
            print(f"  PREV LOT: {row['PREV_LOT']}, MAX WAF3: {row['PREV_MAX_WAF3']}")
            print(f"  PREV START: {row['PREV_START_TIME']}")
            print(f"  DELTA: {row['DELTA_HOURS']:.2f} hours ({row['DELTA_MINUTES']:.1f} minutes)")
        else:
            print(f"  DELTA: N/A (first record)")
    
    # Summary statistics
    valid_deltas = result_df['DELTA_HOURS'].dropna()
    if len(valid_deltas) > 0:
        print(f"\n" + "="*50)
        print("DELTA STATISTICS")
        print("="*50)
        print(f"Count: {len(valid_deltas)}")
        print(f"Mean: {valid_deltas.mean():.2f} hours")
        print(f"Median: {valid_deltas.median():.2f} hours")
        print(f"Min: {valid_deltas.min():.2f} hours")
        print(f"Max: {valid_deltas.max():.2f} hours")
        print(f"Std Dev: {valid_deltas.std():.2f} hours")
    
    # Save results
    output_file = 'adh_timing_analysis_corrected.csv'
    result_df[display_cols].to_csv(output_file, index=False)
    print(f"\nCorrected results saved to: {output_file}")
    
    return result_df

if __name__ == "__main__":
    result = analyze_adh_timing_corrected('Book1.csv')