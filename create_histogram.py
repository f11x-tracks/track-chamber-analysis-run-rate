import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_delta_minutes_histogram(input_file='adh_timing_analysis_corrected.csv'):
    """
    Create a histogram of DELTA_MINUTES from the corrected ADH timing analysis.
    """
    
    # Read the corrected analysis data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records from {input_file}")
    
    # Filter out null values for the histogram
    delta_minutes = df['DELTA_MINUTES'].dropna()
    print(f"Valid DELTA_MINUTES values: {len(delta_minutes)}")
    
    # Basic statistics
    print(f"\nDELTA_MINUTES Statistics:")
    print(f"Mean: {delta_minutes.mean():.1f} minutes")
    print(f"Median: {delta_minutes.median():.1f} minutes")
    print(f"Std Dev: {delta_minutes.std():.1f} minutes")
    print(f"Min: {delta_minutes.min():.1f} minutes")
    print(f"Max: {delta_minutes.max():.1f} minutes")
    print(f"25th percentile: {delta_minutes.quantile(0.25):.1f} minutes")
    print(f"75th percentile: {delta_minutes.quantile(0.75):.1f} minutes")
    
    # Calculate number of bins for ~3 minute resolution
    data_range = delta_minutes.max() - delta_minutes.min()
    num_bins = max(50, int(data_range / 3))  # At least 50 bins, or enough for 3-minute resolution
    print(f"Using {num_bins} bins for ~3 minute resolution")
    
    # Create the histogram
    plt.figure(figsize=(14, 10))
    
    # Main histogram
    plt.subplot(2, 1, 1)
    n, bins, patches = plt.hist(delta_minutes, bins=num_bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(delta_minutes.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {delta_minutes.mean():.1f} min')
    plt.axvline(delta_minutes.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {delta_minutes.median():.1f} min')
    plt.axvline(0, color='green', linestyle='-', linewidth=2, label='Zero (No Gap)', alpha=0.7)
    
    plt.title('Histogram of ADH Chamber Delta Minutes\n(Current Lot MIN WAF3 - Previous Lot MAX WAF3)', fontsize=14, pad=20)
    plt.xlabel('Delta Minutes', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'N = {len(delta_minutes)}\nMean = {delta_minutes.mean():.1f} min\nMedian = {delta_minutes.median():.1f} min\nStd = {delta_minutes.std():.1f} min'
    plt.text(0.75, 0.75, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    # Zoomed in histogram (focusing on main distribution, excluding extreme outliers)
    plt.subplot(2, 1, 2)
    # Filter data to show 95% of the distribution (remove extreme outliers for better view)
    p5, p95 = delta_minutes.quantile([0.025, 0.975])
    filtered_data = delta_minutes[(delta_minutes >= p5) & (delta_minutes <= p95)]
    
    # Calculate bins for zoomed view with 3-minute resolution
    filtered_range = p95 - p5
    zoomed_bins = max(30, int(filtered_range / 3))
    print(f"Using {zoomed_bins} bins for zoomed view (~3 minute resolution)")
    
    plt.hist(filtered_data, bins=zoomed_bins, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(delta_minutes.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {delta_minutes.mean():.1f} min')
    plt.axvline(delta_minutes.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {delta_minutes.median():.1f} min')
    plt.axvline(0, color='green', linestyle='-', linewidth=2, label='Zero (No Gap)', alpha=0.7)
    
    plt.title(f'Zoomed View: 95% of Data ({p5:.1f} to {p95:.1f} minutes)', fontsize=12)
    plt.xlabel('Delta Minutes', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'delta_minutes_histogram.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved as: {output_file}")
    
    # Show the plot
    plt.show()
    
    # Additional analysis: categorize the deltas
    print(f"\nDelta Categories:")
    negative_count = len(delta_minutes[delta_minutes < 0])
    zero_to_30_count = len(delta_minutes[(delta_minutes >= 0) & (delta_minutes <= 30)])
    thirty_to_60_count = len(delta_minutes[(delta_minutes > 30) & (delta_minutes <= 60)])
    sixty_to_120_count = len(delta_minutes[(delta_minutes > 60) & (delta_minutes <= 120)])
    over_120_count = len(delta_minutes[delta_minutes > 120])
    
    print(f"Negative (Parallel Processing): {negative_count} ({negative_count/len(delta_minutes)*100:.1f}%)")
    print(f"0-30 minutes (Fast Handoff): {zero_to_30_count} ({zero_to_30_count/len(delta_minutes)*100:.1f}%)")
    print(f"30-60 minutes (Normal): {thirty_to_60_count} ({thirty_to_60_count/len(delta_minutes)*100:.1f}%)")
    print(f"60-120 minutes (Slower): {sixty_to_120_count} ({sixty_to_120_count/len(delta_minutes)*100:.1f}%)")
    print(f"Over 120 minutes (Gaps/Delays): {over_120_count} ({over_120_count/len(delta_minutes)*100:.1f}%)")
    
    return delta_minutes

if __name__ == "__main__":
    delta_data = create_delta_minutes_histogram('adh_timing_analysis_corrected.csv')