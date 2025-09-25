import pandas as pd

def process_track_chamber_data(input_file='Book1.csv', output_file=None):
    """
    Process track chamber analysis data by creating new columns:
    - TRACK_RCP: text before '+' in RECIPE column
    - LAYER: 7th and 8th characters from RETICLE column
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
    
    Returns:
        pandas.DataFrame: Processed dataframe
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows from {input_file}")
    
    # Create TRACK_RCP column: text before the '+' in RECIPE column, stripped of trailing spaces
    df['TRACK_RCP'] = df['RECIPE'].str.split('+').str[0].str.strip()
    
    # Create LAYER column: 7th and 8th character from RETICLE column
    df['LAYER'] = df['RETICLE'].str[6:8]
    
    # Display summary
    print(f"\nCreated columns:")
    print(f"- TRACK_RCP: {df['TRACK_RCP'].nunique()} unique values")
    print(f"- LAYER: {df['LAYER'].nunique()} unique values")
    
    # Save if output file specified
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nSaved processed data to {output_file}")
    
    return df

# Example usage
if __name__ == "__main__":
    df = process_track_chamber_data('Book1.csv', 'Book1_processed.csv')
    
    # Show sample of the new columns
    print("\nSample of new columns:")
    print(df[['RECIPE', 'TRACK_RCP', 'RETICLE', 'LAYER']].head(3))
    
    print("\nUnique TRACK_RCP values:")
    for recipe in sorted(df['TRACK_RCP'].dropna().unique()):
        print(f"  - {recipe}")
    
    print(f"\nUnique LAYER values:")
    layers = sorted([x for x in df['LAYER'].dropna().unique() if x])
    print(f"  {', '.join(layers)}")