"""
Main Orchestrator — Event Impact & Response Intelligence Platform
=================================================================
Runs the complete pipeline in order:
  1. Data pipeline (clean, feature engineer, target derivation)
  2. Model training (3 LightGBM models with evaluation)
  3. Hotspot clustering (HDBSCAN)
  4. Diversion heuristic (co-occurrence lookup table)

Then starts the Flask server to serve the dashboard.
"""

import sys
import os
import io

# Fix Windows console encoding for emoji/unicode
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))


def main():
    print("=" * 70)
    print("  EVENT IMPACT & RESPONSE INTELLIGENCE PLATFORM")
    print("  Full Pipeline Execution")
    print("=" * 70)

    # Step 1: Data Pipeline
    print("\n")
    print("=" * 70)
    print("PHASE 1: DATA PIPELINE")
    print("=" * 70)
    from data_pipeline import run_pipeline
    df, clf_features, reg_features = run_pipeline()

    # Step 2: Model Training
    print("\n")
    print("=" * 70)
    print("PHASE 2: MODEL TRAINING")
    print("=" * 70)
    from train_models import run_training
    results = run_training()

    # Step 3: Hotspot Clustering
    print("\n")
    print("=" * 70)
    print("PHASE 3: HOTSPOT CLUSTERING")
    print("=" * 70)
    from clustering import run_clustering
    hotspots = run_clustering()

    # Step 4: Diversion Heuristic
    print("\n")
    print("=" * 70)
    print("PHASE 4: DIVERSION HEURISTIC")
    print("=" * 70)
    from diversion import build_diversion_table
    diversions = build_diversion_table()

    # Final summary
    print("\n")
    print("=" * 70)
    print("[OK] ALL PIPELINE PHASES COMPLETE")
    print("=" * 70)
    print("\nOutputs generated:")
    print("  - data/features_full.csv")
    print("  - data/features_clearance_subset.csv")
    print("  - models/*.pkl (3 models + encoders)")
    print("  - outputs/model_evaluation_report.json")
    print("  - outputs/hotspot_clusters.json")
    print("  - outputs/diversion_lookup_table.json")
    print("\nTo start the dashboard:")
    print("  python src/server.py")
    print("  Then open http://localhost:5000")


if __name__ == "__main__":
    main()
