"""Run just clustering and diversion (models already trained)."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from clustering import run_clustering
run_clustering()

from diversion import build_diversion_table
build_diversion_table()

print("\n[OK] Clustering and diversion complete!")
