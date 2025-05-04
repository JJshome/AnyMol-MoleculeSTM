"""
Utility functions for data processing in AnyMol-MoleculeSTM.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, Draw
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def load_smiles_from_file(file_path: str, smiles_column: str = "SMILES") -> List[str]:
    """
    Load SMILES strings from file.
    
    Args:
        file_path: Path to file containing SMILES
        smiles_column: Column name containing SMILES (for CSV/TSV)
        
    Returns:
        List of SMILES strings
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
        return df[smiles_column].tolist()
    elif file_extension == '.tsv':
        df = pd.read_csv(file_path, sep='\t')
        return df[smiles_column].tolist()
    elif file_extension == '.smi':
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip().split()[0] for line in f if line.strip()]
    elif file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    elif file_extension == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                if isinstance(data[0], str):
                    return data
                elif isinstance(data[0], dict) and smiles_column in data[0]:
                    return [item[smiles_column] for item in data]
            elif isinstance(data, dict) and smiles_column in data:
                return data[smiles_column]
        raise ValueError(f"Could not extract SMILES from JSON file: {file_path}")
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

def canonicalize_smiles(smiles: str) -> str:
    """
    Convert SMILES to canonical form.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Canonical SMILES or empty string if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        else:
            return ""
    except Exception:
        return ""

def filter_valid_smiles(smiles_list: List[str]) -> List[str]:
    """
    Filter out invalid SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        List of valid SMILES strings
    """
    valid_smiles = []
    for smiles in tqdm(smiles_list, desc="Filtering valid SMILES"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
    
    logger.info(f"Filtered {len(smiles_list) - len(valid_smiles)} invalid SMILES out of {len(smiles_list)}")
    return valid_smiles

def split_dataset(
    data_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    smiles_column: str = "SMILES",
    shuffle: bool = True,
    random_state: int = 42,
) -> Dict[str, str]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        data_file: Path to data file
        output_dir: Directory to save split datasets
        train_ratio: Ratio of training set
        val_ratio: Ratio of validation set
        test_ratio: Ratio of test set
        smiles_column: Column name containing SMILES
        shuffle: Whether to shuffle data before splitting
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with paths to train, validation, and test files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    file_extension = os.path.splitext(data_file)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(data_file)
    elif file_extension == '.tsv':
        df = pd.read_csv(data_file, sep='\t')
    elif file_extension == '.json':
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
    elif file_extension == '.pkl' or file_extension == '.pickle':
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    # Check if SMILES column exists
    if smiles_column not in df.columns:
        raise ValueError(f"SMILES column '{smiles_column}' not found in data file")
    
    # Split data
    train_val_data, test_data = train_test_split(
        df, test_size=test_ratio, random_state=random_state, shuffle=shuffle
    )
    
    # Adjust validation ratio relative to remaining data
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_ratio_adjusted, random_state=random_state, shuffle=shuffle
    )
    
    # Save split datasets
    train_file = os.path.join(output_dir, 'train.csv')
    val_file = os.path.join(output_dir, 'validation.csv')
    test_file = os.path.join(output_dir, 'test.csv')
    
    train_data.to_csv(train_file, index=False)
    val_data.to_csv(val_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    logger.info(f"Split dataset into {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples")
    
    return {
        'train': train_file,
        'validation': val_file,
        'test': test_file,
    }

def calculate_molecular_properties(
    smiles_list: List[str],
    property_names: List[str] = ["logP", "TPSA", "MolWt", "QED"],
) -> pd.DataFrame:
    """
    Calculate molecular properties for a list of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        property_names: List of property names to calculate
        
    Returns:
        DataFrame with calculated properties
    """
    property_calculators = {
        "logP": lambda mol: Descriptors.MolLogP(mol),
        "TPSA": lambda mol: Descriptors.TPSA(mol),
        "MolWt": lambda mol: Descriptors.MolWt(mol),
        "QED": lambda mol: QED.qed(mol),
        "NumHDonors": lambda mol: Descriptors.NumHDonors(mol),
        "NumHAcceptors": lambda mol: Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": lambda mol: Descriptors.NumRotatableBonds(mol),
        "NumRings": lambda mol: Chem.rdMolDescriptors.CalcNumRings(mol),
        "NumAromaticRings": lambda mol: Chem.rdMolDescriptors.CalcNumAromaticRings(mol),
        "FractionCSP3": lambda mol: Chem.rdMolDescriptors.CalcFractionCSP3(mol),
        "HeavyAtomCount": lambda mol: mol.GetNumHeavyAtoms(),
    }
    
    # Check if all requested properties can be calculated
    for prop in property_names:
        if prop not in property_calculators:
            logger.warning(f"Property '{prop}' not available. Available properties: {list(property_calculators.keys())}")
    
    # Initialize results dictionary
    results = {
        "SMILES": smiles_list,
    }
    
    # Calculate properties
    for prop in property_names:
        if prop in property_calculators:
            results[prop] = []
    
    # Process each molecule
    for smiles in tqdm(smiles_list, desc="Calculating properties"):
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            # Add NaN for invalid SMILES
            for prop in property_names:
                if prop in property_calculators:
                    results[prop].append(float('nan'))
            continue
        
        # Calculate each property
        for prop in property_names:
            if prop in property_calculators:
                try:
                    value = property_calculators[prop](mol)
                    results[prop].append(value)
                except Exception as e:
                    logger.warning(f"Error calculating {prop} for {smiles}: {e}")
                    results[prop].append(float('nan'))
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def visualize_property_distribution(
    data: Union[pd.DataFrame, str],
    property_names: List[str] = ["logP", "TPSA", "MolWt", "QED"],
    bins: int = 30,
    figsize: Tuple[int, int] = (15, 10),
    output_file: Optional[str] = None,
    smiles_column: str = "SMILES",
):
    """
    Visualize distribution of molecular properties.
    
    Args:
        data: DataFrame or path to data file
        property_names: List of property names to visualize
        bins: Number of bins for histograms
        figsize: Figure size
        output_file: Path to save the visualization
        smiles_column: Column name containing SMILES
    """
    # Load data if needed
    if isinstance(data, str):
        file_extension = os.path.splitext(data)[1].lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(data)
        elif file_extension == '.tsv':
            df = pd.read_csv(data, sep='\t')
        elif file_extension == '.json':
            with open(data, 'r', encoding='utf-8') as f:
                df = pd.DataFrame(json.load(f))
        elif file_extension == '.pkl' or file_extension == '.pickle':
            with open(data, 'rb') as f:
                df = pd.DataFrame(pickle.load(f))
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    else:
        df = data
    
    # Check if all properties are in the data
    missing_props = [prop for prop in property_names if prop not in df.columns]
    
    if missing_props:
        # Calculate missing properties
        logger.info(f"Properties {missing_props} not found in data, calculating...")
        
        if smiles_column not in df.columns:
            raise ValueError(f"SMILES column '{smiles_column}' not found in data")
        
        smiles_list = df[smiles_column].tolist()
        props_df = calculate_molecular_properties(smiles_list, missing_props)
        
        # Add calculated properties to dataframe
        for prop in missing_props:
            if prop in props_df.columns:
                df[prop] = props_df[prop]
    
    # Create visualization
    n_props = len(property_names)
    n_cols = min(3, n_props)
    n_rows = (n_props + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_props == 1:
        axes = np.array([axes])
    
    axes = axes.flatten()
    
    for i, prop in enumerate(property_names):
        if i < len(axes):
            if prop in df.columns:
                ax = axes[i]
                sns.histplot(df[prop].dropna(), bins=bins, kde=True, ax=ax)
                ax.set_title(f"Distribution of {prop}")
                ax.set_xlabel(prop)
                ax.set_ylabel("Count")
            else:
                logger.warning(f"Property '{prop}' not found in data")
    
    # Hide empty subplots
    for i in range(n_props, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved property distribution visualization to {output_file}")
    
    plt.show()

def visualize_molecule_grid(
    smiles_list: List[str],
    n_cols: int = 4,
    n_rows: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 10),
    output_file: Optional[str] = None,
    mol_size: Tuple[int, int] = (300, 300),
    legends: Optional[List[str]] = None,
):
    """
    Visualize a grid of molecules.
    
    Args:
        smiles_list: List of SMILES strings
        n_cols: Number of columns in the grid
        n_rows: Number of rows (if None, calculated from n_cols)
        figsize: Figure size
        output_file: Path to save the visualization
        mol_size: Size of each molecule image
        legends: List of captions for each molecule
    """
    # Convert SMILES to RDKit molecules
    mols = []
    valid_indices = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mols.append(mol)
            valid_indices.append(i)
    
    if not mols:
        logger.warning("No valid molecules to visualize")
        return
    
    # Prepare legends if provided
    if legends:
        mol_legends = [legends[i] for i in valid_indices if i < len(legends)]
        if len(mol_legends) < len(mols):
            mol_legends.extend([f"Mol {i+1}" for i in range(len(mol_legends), len(mols))])
    else:
        mol_legends = [f"Mol {i+1}" for i in range(len(mols))]
    
    # Calculate grid dimensions
    if n_rows is None:
        n_rows = (len(mols) + n_cols - 1) // n_cols
    
    # Create molecule grid
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_cols,
        subImgSize=mol_size,
        legends=mol_legends[:len(mols)],
        useSVG=False
    )
    
    # Display or save the image
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    
    if output_file:
        img.save(output_file)
        logger.info(f"Saved molecule grid visualization to {output_file}")
    
    plt.show()

def preprocess_dataset(
    input_file: str,
    output_file: str,
    smiles_column: str = "SMILES",
    canonicalize: bool = True,
    filter_invalid: bool = True,
    calculate_props: bool = True,
    property_names: List[str] = ["logP", "TPSA", "MolWt", "QED"],
    remove_duplicates: bool = True,
):
    """
    Preprocess a molecular dataset.
    
    Args:
        input_file: Path to input data file
        output_file: Path to save processed data
        smiles_column: Column name containing SMILES
        canonicalize: Whether to canonicalize SMILES
        filter_invalid: Whether to filter out invalid SMILES
        calculate_props: Whether to calculate molecular properties
        property_names: List of property names to calculate
        remove_duplicates: Whether to remove duplicate molecules
    """
    # Load data
    file_extension = os.path.splitext(input_file)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(input_file)
    elif file_extension == '.tsv':
        df = pd.read_csv(input_file, sep='\t')
    elif file_extension == '.smi':
        df = pd.DataFrame({
            smiles_column: load_smiles_from_file(input_file)
        })
    elif file_extension == '.txt':
        df = pd.DataFrame({
            smiles_column: load_smiles_from_file(input_file)
        })
    elif file_extension == '.json':
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                if isinstance(data[0], str):
                    df = pd.DataFrame({smiles_column: data})
                else:
                    df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    # Check if SMILES column exists
    if smiles_column not in df.columns:
        raise ValueError(f"SMILES column '{smiles_column}' not found in data file")
    
    # Process SMILES
    if canonicalize:
        logger.info("Canonicalizing SMILES...")
        df[smiles_column] = df[smiles_column].apply(canonicalize_smiles)
    
    # Filter invalid SMILES
    if filter_invalid:
        logger.info("Filtering invalid SMILES...")
        df = df[df[smiles_column].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
    
    # Remove duplicates
    if remove_duplicates:
        logger.info("Removing duplicate molecules...")
        original_len = len(df)
        df = df.drop_duplicates(subset=[smiles_column])
        logger.info(f"Removed {original_len - len(df)} duplicate molecules")
    
    # Calculate properties
    if calculate_props:
        logger.info("Calculating molecular properties...")
        # Determine which properties need to be calculated
        props_to_calc = [prop for prop in property_names if prop not in df.columns]
        
        if props_to_calc:
            props_df = calculate_molecular_properties(df[smiles_column].tolist(), props_to_calc)
            
            # Add calculated properties to dataframe
            for prop in props_to_calc:
                if prop in props_df.columns:
                    df[prop] = props_df[prop]
    
    # Save processed data
    output_extension = os.path.splitext(output_file)[1].lower()
    
    if output_extension == '.csv':
        df.to_csv(output_file, index=False)
    elif output_extension == '.tsv':
        df.to_csv(output_file, sep='\t', index=False)
    elif output_extension == '.json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(df.to_dict('records'), f)
    elif output_extension == '.pkl' or output_extension == '.pickle':
        with open(output_file, 'wb') as f:
            pickle.dump(df.to_dict('records'), f)
    else:
        raise ValueError(f"Unsupported output extension: {output_extension}")
    
    logger.info(f"Processed dataset saved to {output_file} with {len(df)} molecules")
    
    return df
