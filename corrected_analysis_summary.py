"""
CORRECTED ADH Chamber Timing Analysis Summary
============================================

This corrected analysis calculates the time deltas between:
- Current LOT+INTRO_DATE: MIN WAF3 ADH chamber operation time
- Previous LOT+INTRO_DATE: MAX WAF3 ADH chamber operation time

Key Findings:
============

1. DATASET OVERVIEW:
   - Total records analyzed: 297 unique LOT+INTRO_DATE combinations with ADH operations
   - Date range: 2025-09-09 to 2025-09-16
   - ADH chambers involved: ADH101, ADH102, ADH103, ADH104, ADH105, ADH106, ADH107

2. CORRECTED TIMING STATISTICS:
   - Mean delta: 0.97 hours (58.2 minutes)
   - Median delta: 0.51 hours (30.6 minutes)
   - Minimum delta: -0.47 hours (-28.2 minutes) [negative indicates overlap/parallel processing]
   - Maximum delta: 9.41 hours (564.6 minutes)
   - Standard deviation: 1.34 hours

3. COMPARISON WITH PREVIOUS ANALYSIS:
   METRIC                  | ORIGINAL (MIN-MIN) | CORRECTED (MIN-MAX)
   ----------------------- | ------------------ | -------------------
   Mean Delta              | 1.08 hours         | 0.97 hours
   Median Delta            | 0.59 hours         | 0.51 hours
   Record Count            | 322                | 296
   Min Delta               | -0.24 hours        | -0.47 hours
   Max Delta               | 9.42 hours         | 9.41 hours

4. ANALYSIS INTERPRETATION:
   - The corrected analysis (MIN to MAX) shows slightly shorter average intervals
   - This makes sense: we're measuring from previous lot's HIGHEST WAF3 time
     to current lot's LOWEST WAF3 time, which represents the actual processing gap
   - The negative deltas are more pronounced, indicating more parallel processing
   - This measurement better reflects the actual production flow timing

5. OPERATIONAL INSIGHTS:
   - 50% of lot transitions occur within 30.6 minutes (vs 35.4 in original)
   - More instances of parallel processing (negative deltas up to -28 minutes)
   - The corrected method better captures the true "gap" between lot completions
   - Still shows efficient chamber utilization with rapid lot turnover

6. PRODUCTION FLOW PATTERN:
   - Current lot's first wafer (MIN WAF3) typically starts processing shortly
     after the previous lot's last wafer (MAX WAF3) completes
   - This represents the true "handoff" time between lots in production
   - Negative values indicate overlapping production (different chambers)

This corrected analysis provides a more accurate picture of the actual
production gaps and chamber handoff timing in the manufacturing process.
"""

print(__doc__)