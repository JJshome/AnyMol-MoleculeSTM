"""
Utility functions for AnyMol-MoleculeSTM.
"""

import os
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Convert a SMILES string to an RDKit molecule object.
    
    Args:
        smiles: SMILES string.
        
    Returns:
        RDKit molecule object or None if invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
        return mol
    except Exception as e:
        logger.error(f"Error converting SMILES to molecule: {e}")
        return None

def mol_to_smiles(mol: Chem.Mol) -> Optional[str]:
    """
    Convert an RDKit molecule to a SMILES string.
    
    Args:
        mol: RDKit molecule object.
        
    Returns:
        SMILES string or None if invalid.
    """
    try:
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception as e:
        logger.error(f"Error converting molecule to SMILES: {e}")
        return None

def compute_molecular_fingerprints(
    mols: List[Union[str, Chem.Mol]],
    fp_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """
    Compute molecular fingerprints for a list of molecules.
    
    Args:
        mols: List of SMILES strings or RDKit molecules.
        fp_type: Fingerprint type ('morgan', 'maccs', 'rdkit').
        radius: Radius for Morgan fingerprints.
        n_bits: Number of bits in fingerprint.
        
    Returns:
        Array of fingerprints.
    """
    # Convert SMILES to RDKit molecules if needed
    mol_objs = []
    for mol in mols:
        if isinstance(mol, str):
            mol_obj = smiles_to_mol(mol)
            if mol_obj is not None:
                mol_objs.append(mol_obj)
        else:
            mol_objs.append(mol)
    
    # Compute fingerprints
    fingerprints = []
    for mol in mol_objs:
        try:
            if fp_type == "morgan":
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            elif fp_type == "maccs":
                fp = AllChem.GetMACCSKeysFingerprint(mol)
            elif fp_type == "rdkit":
                fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
            else:
                raise ValueError(f"Unknown fingerprint type: {fp_type}")
            
            # Convert to numpy array
            fp_array = np.zeros((0,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, fp_array)
            fingerprints.append(fp_array)
        except Exception as e:
            logger.error(f"Error computing fingerprint: {e}")
            # Add a zero array for failed fingerprints
            fingerprints.append(np.zeros(n_bits, dtype=np.int8))
    
    return np.array(fingerprints)

def compute_molecular_descriptors(
    mols: List[Union[str, Chem.Mol]],
    descriptors: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute molecular descriptors for a list of molecules.
    
    Args:
        mols: List of SMILES strings or RDKit molecules.
        descriptors: List of descriptor names to compute.
        
    Returns:
        DataFrame with descriptor values.
    """
    # Default descriptors if not specified
    if descriptors is None:
        descriptors = [
            "MolWt", "LogP", "NumHDonors", "NumHAcceptors", 
            "NumRotatableBonds", "NumAromaticRings", "TPSA",
            "QED", "NumAtoms", "NumBonds", "NumRings"
        ]
    
    # Convert SMILES to RDKit molecules if needed
    mol_objs = []
    for mol in mols:
        if isinstance(mol, str):
            mol_obj = smiles_to_mol(mol)
            if mol_obj is not None:
                mol_objs.append(mol_obj)
        else:
            mol_objs.append(mol)
    
    # Define descriptor functions
    descriptor_funcs = {
        "MolWt": Descriptors.MolWt,
        "LogP": Descriptors.MolLogP,
        "NumHDonors": Descriptors.NumHDonors,
        "NumHAcceptors": Descriptors.NumHAcceptors,
        "NumRotatableBonds": Descriptors.NumRotatableBonds,
        "NumAromaticRings": lambda m: Chem.Lipinski.NumAromaticRings(m),
        "TPSA": Descriptors.TPSA,
        "QED": QED.qed,
        "NumAtoms": lambda m: m.GetNumAtoms(),
        "NumBonds": lambda m: m.GetNumBonds(),
        "NumRings": lambda m: Chem.Lipinski.NumRings(m),
    }
    
    # Compute descriptors
    results = []
    for mol in mol_objs:
        mol_descriptors = {}
        for desc in descriptors:
            if desc in descriptor_funcs:
                try:
                    mol_descriptors[desc] = descriptor_funcs[desc](mol)
                except Exception as e:
                    logger.error(f"Error computing descriptor {desc}: {e}")
                    mol_descriptors[desc] = np.nan
            else:
                logger.warning(f"Unknown descriptor: {desc}")
                mol_descriptors[desc] = np.nan
        results.append(mol_descriptors)
    
    return pd.DataFrame(results)

def validate_molecules(mols: List[Union[str, Chem.Mol]]) -> List[bool]:
    """
    Validate molecules.
    
    Args:
        mols: List of SMILES strings or RDKit molecules.
        
    Returns:
        List of Boolean values indicating if each molecule is valid.
    """
    results = []
    for mol in mols:
        if isinstance(mol, str):
            mol_obj = smiles_to_mol(mol)
            results.append(mol_obj is not None)
        else:
            results.append(mol is not None)
    return results

def filter_valid_molecules(mols: List[Union[str, Chem.Mol]]) -> List[Union[str, Chem.Mol]]:
    """
    Filter out invalid molecules.
    
    Args:
        mols: List of SMILES strings or RDKit molecules.
        
    Returns:
        List of valid molecules.
    """
    valid_indices = validate_molecules(mols)
    return [mol for mol, is_valid in zip(mols, valid_indices) if is_valid]

def calculate_similarity(
    mol1: Union[str, Chem.Mol],
    mol2: Union[str, Chem.Mol],
    fp_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    """
    Calculate similarity between two molecules.
    
    Args:
        mol1: First molecule (SMILES or RDKit mol).
        mol2: Second molecule (SMILES or RDKit mol).
        fp_type: Fingerprint type.
        radius: Radius for Morgan fingerprints.
        n_bits: Number of bits in fingerprint.
        
    Returns:
        Tanimoto similarity score.
    """
    # Convert to RDKit molecules if needed
    if isinstance(mol1, str):
        mol1 = smiles_to_mol(mol1)
    if isinstance(mol2, str):
        mol2 = smiles_to_mol(mol2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    # Compute fingerprints
    if fp_type == "morgan":
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=n_bits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=n_bits)
    elif fp_type == "maccs":
        fp1 = AllChem.GetMACCSKeysFingerprint(mol1)
        fp2 = AllChem.GetMACCSKeysFingerprint(mol2)
    elif fp_type == "rdkit":
        fp1 = Chem.RDKFingerprint(mol1, fpSize=n_bits)
        fp2 = Chem.RDKFingerprint(mol2, fpSize=n_bits)
    else:
        raise ValueError(f"Unknown fingerprint type: {fp_type}")
    
    # Calculate Tanimoto similarity
    return Chem.DataStructs.TanimotoSimilarity(fp1, fp2)

def calculate_pairwise_similarities(
    mols: List[Union[str, Chem.Mol]],
    fp_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """
    Calculate pairwise similarities between molecules.
    
    Args:
        mols: List of molecules (SMILES or RDKit mol).
        fp_type: Fingerprint type.
        radius: Radius for Morgan fingerprints.
        n_bits: Number of bits in fingerprint.
        
    Returns:
        Similarity matrix.
    """
    n = len(mols)
    sim_matrix = np.zeros((n, n))
    
    # Convert to RDKit molecules if needed
    mol_objs = []
    for mol in mols:
        if isinstance(mol, str):
            mol_obj = smiles_to_mol(mol)
            mol_objs.append(mol_obj)
        else:
            mol_objs.append(mol)
    
    # Compute fingerprints
    fps = []
    for mol in mol_objs:
        if mol is None:
            fp = None
        elif fp_type == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        elif fp_type == "maccs":
            fp = AllChem.GetMACCSKeysFingerprint(mol)
        elif fp_type == "rdkit":
            fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")
        fps.append(fp)
    
    # Calculate pairwise similarities
    for i in range(n):
        for j in range(i, n):
            if fps[i] is None or fps[j] is None:
                sim = 0.0
            else:
                sim = Chem.DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    
    return sim_matrix
