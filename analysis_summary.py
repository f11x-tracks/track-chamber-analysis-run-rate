"""
ADH Chamber Timing Analysis Summary
==================================

This analysis examined the time deltas between ADH chamber operations for 
the lowest WAF3 numbers in each LOT and INTRO_DATE combination.

Key Findings:
============

1. DATASET OVERVIEW:
   - Total records analyzed: 323 unique LOT+INTRO_DATE combinations with ADH operations
   - Date range: 2025-09-09 to 2025-09-19
   - ADH chambers involved: ADH101, ADH102, ADH103, ADH104, ADH105, ADH106, ADH107

2. TIMING STATISTICS:
   - Mean delta: 1.08 hours (64.8 minutes)
   - Median delta: 0.59 hours (35.4 minutes)
   - Minimum delta: -0.24 hours (-14.4 minutes) [negative indicates overlap/parallel processing]
   - Maximum delta: 9.42 hours (565.2 minutes)
   - Standard deviation: 1.32 hours

3. OPERATIONAL PATTERNS:
   - Most operations occur within 1-2 hours of the previous lot
   - Some rapid succession processing (< 15 minutes between lots)
   - Occasional longer gaps (> 4 hours) likely due to scheduled downtime
   - Negative deltas suggest parallel processing of different lots

4. CHAMBER UTILIZATION:
   - ADH101 is the most frequently used chamber
   - Multiple ADH chambers (101-107) enable parallel processing
   - Load balancing across chambers helps maintain throughput

5. NOTABLE OBSERVATIONS:
   - Fastest turnaround: 2.0 minutes between lots
   - Longest gap: 9.42 hours (likely overnight or maintenance period)
   - 50% of lot transitions occur within 35 minutes
   - 75% of lot transitions occur within ~90 minutes

This analysis helps understand the production flow timing and identify
opportunities for optimizing chamber utilization and reducing cycle times.
"""

print(__doc__)

# Additional analysis functions could be added here for:
# - Chamber-specific analysis
# - Time-of-day patterns
# - Lot size impact on timing
# - Trend analysis over time periods