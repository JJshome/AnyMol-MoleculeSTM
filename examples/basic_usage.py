"""
Basic usage examples for AnyMol-MoleculeSTM.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from anymol import AnyMolSTM
from anymol.utils import compute_molecular_descriptors, calculate_pairwise_similarities

def main():
    print("AnyMol-MoleculeSTM Basic Usage Example")
    print("=====================================")
    
    # Initialize model
    print("\nInitializing model...")
    model = AnyMolSTM()
    model.load_pretrained("default")
    
    # Example molecules
    example_smiles = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "Cc1ccccc1",  # Toluene
        "c1ccccc1O",  # Phenol
    ]
    
    print(f"\nWorking with {len(example_smiles)} example molecules:")
    for i, smile in enumerate(example_smiles):
        print(f"  {i+1}. {smile} - {Chem.MolFromSmiles(smile).GetProp('_Name') if Chem.MolFromSmiles(smile).HasProp('_Name') else 'No name'}")
    
    # Compute molecular descriptors
    print("\nComputing molecular descriptors...")
    descriptors = compute_molecular_descriptors(example_smiles)
    print(descriptors)
    
    # Calculate pairwise similarities
    print("\nCalculating pairwise similarities...")
    similarities = calculate_pairwise_similarities(example_smiles)
    print(pd.DataFrame(similarities, index=example_smiles, columns=example_smiles))
    
    # Generate new molecules with target properties
    print("\nGenerating new molecules with target properties...")
    target_properties = {
        "LogP": 2.5,
        "MolWt": 200.0,
    }
    new_molecules = model.generate(target_properties, num_molecules=5)
    print("Generated molecules:")
    for i, smile in enumerate(new_molecules):
        print(f"  {i+1}. {smile}")
    
    # Visualize molecules
    print("\nVisualizing molecules...")
    model.visualize(new_molecules, filename="generated_molecules.png")
    print("Molecules saved to 'generated_molecules.png'")
    
    # Predict properties
    print("\nPredicting properties for generated molecules...")
    properties = model.predict_properties(new_molecules, ["LogP", "MolWt", "TPSA"])
    print("Predicted properties:")
    for prop, values in properties.items():
        print(f"  {prop}: {values}")
    
    # Optimize molecules
    print("\nOptimizing molecules towards target properties...")
    optimized_molecules = model.optimize(example_smiles, target_properties)
    print("Optimized molecules:")
    for i, (original, optimized) in enumerate(zip(example_smiles, optimized_molecules)):
        print(f"  {i+1}. {original} -> {optimized}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
