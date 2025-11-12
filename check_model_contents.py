"""
Quick script to check what's in the saved model
"""
import pickle
from pathlib import Path

model_path = Path("models/model_balanced.pkl")

with open(model_path, 'rb') as f:
    model_package = pickle.load(f)

print("Keys in model_package:")
for key in model_package.keys():
    print(f"  - {key}")
    if key == 'performance':
        print(f"    Performance keys: {list(model_package[key].keys())}")
