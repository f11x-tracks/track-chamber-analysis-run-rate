track_chamber_processor reads Book1.csv and creates Book1_processed.csv

Book1.csv (original data)
    ↓
track_chamber_processor.py (adds TRACK_RCP & LAYER columns)
    ↓
Book1_processed.csv (processed data)
    ↓
All other analysis scripts use this file