"""
Script to create and preprocess example datasets for AnyMol-MoleculeSTM.
"""

import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
import logging
from tqdm import tqdm
import argparse
import json
import yaml
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from anymol.data.utils import (
    load_smiles_from_file,
    preprocess_dataset,
    visualize_property_distribution,
    visualize_molecule_grid,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_chembl_subset(output_file: str, num_molecules: int = 10000, random_seed: int = 42):
    """
    Download a subset of molecules from ChEMBL dataset.
    
    Args:
        output_file: Path to save the dataset
        num_molecules: Number of molecules to download
        random_seed: Random seed for reproducibility
    """
    # Import here to make it optional
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        logger.error("chembl_webresource_client package is required. Install it with 'pip install chembl_webresource_client'")
        return
    
    logger.info(f"Downloading {num_molecules} molecules from ChEMBL")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Define ChEMBL client
    molecule = new_client.molecule
    
    # First get count of all molecules
    total_count = molecule.filter(molecule_properties__acd_logp__isnull=False).count()
    logger.info(f"Total molecules in ChEMBL: {total_count}")
    
    # Generate random offsets
    max_offset = min(total_count - num_molecules, total_count - 1)
    offset = np.random.randint(0, max_offset)
    
    # Download molecules
    molecules = molecule.filter(
        molecule_properties__acd_logp__isnull=False
    ).order_by('molecule_chembl_id')[offset:offset+num_molecules]
    
    # Extract relevant information
    data = []
    
    for mol in tqdm(molecules, desc="Processing molecules"):
        try:
            smiles = mol['molecule_structures']['canonical_smiles']
            if not smiles:
                continue
                
            # Check if SMILES is valid
            rdkit_mol = Chem.MolFromSmiles(smiles)
            if rdkit_mol is None:
                continue
                
            # Get properties
            item = {
                'SMILES': smiles,
                'chembl_id': mol['molecule_chembl_id'],
                'pref_name': mol.get('pref_name', ''),
                'logP': mol['molecule_properties'].get('acd_logp', None),
                'TPSA': mol['molecule_properties'].get('psa', None),
                'MolWt': mol['molecule_properties'].get('full_mwt', None),
                'HBA': mol['molecule_properties'].get('hba', None),
                'HBD': mol['molecule_properties'].get('hbd', None),
                'RotBonds': mol['molecule_properties'].get('rtb', None),
            }
            
            data.append(item)
            
        except Exception as e:
            logger.warning(f"Error processing molecule: {e}")
    
    # Save to file
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    file_extension = os.path.splitext(output_file)[1].lower()
    if file_extension == '.csv':
        df.to_csv(output_file, index=False)
    elif file_extension == '.json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(df.to_dict('records'), f, indent=2)
    else:
        logger.warning(f"Unsupported file extension: {file_extension}, saving as CSV")
        df.to_csv(output_file.rsplit('.', 1)[0] + '.csv', index=False)
    
    logger.info(f"Downloaded {len(data)} molecules from ChEMBL to {output_file}")
    
    return df

def download_zinc_subset(output_file: str, num_molecules: int = 10000, random_seed: int = 42):
    """
    Download a subset of molecules from ZINC dataset.
    
    Args:
        output_file: Path to save the dataset
        num_molecules: Number of molecules to download
        random_seed: Random seed for reproducibility
    """
    try:
        import requests
    except ImportError:
        logger.error("requests package is required. Install it with 'pip install requests'")
        return
    
    logger.info(f"Downloading {num_molecules} molecules from ZINC")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # ZINC tranches URL
    tranches_url = "https://zinc.docking.org/tranches/all.txt"
    
    # Get all tranches
    try:
        response = requests.get(tranches_url)
        response.raise_for_status()
        
        tranches = [line.strip() for line in response.text.split('\n') if line.strip()]
        logger.info(f"Found {len(tranches)} tranches in ZINC")
        
        # Randomly select tranches
        np.random.shuffle(tranches)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting ZINC tranches: {e}")
        return
    
    # Download molecules from tranches until we have enough
    data = []
    
    for tranche in tqdm(tranches, desc="Processing ZINC tranches"):
        if len(data) >= num_molecules:
            break
            
        # Tranche URL
        tranche_url = f"https://zinc.docking.org/substances/subsets/{tranche}"
        
        try:
            response = requests.get(tranche_url)
            if response.status_code != 200:
                continue
                
            # Extract SMILES from tranche
            lines = response.text.strip().split('\n')
            for line in lines:
                if not line.strip():
                    continue
                    
                parts = line.split()
                if len(parts) >= 2:
                    zinc_id, smiles = parts[0], parts[1]
                    
                    # Check if SMILES is valid
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    
                    # Calculate properties
                    try:
                        logP = Descriptors.MolLogP(mol)
                        TPSA = Descriptors.TPSA(mol)
                        MolWt = Descriptors.MolWt(mol)
                        qed = QED.qed(mol)
                        
                        data.append({
                            'SMILES': smiles,
                            'zinc_id': zinc_id,
                            'logP': logP,
                            'TPSA': TPSA,
                            'MolWt': MolWt,
                            'QED': qed,
                            'HBA': Descriptors.NumHAcceptors(mol),
                            'HBD': Descriptors.NumHDonors(mol),
                            'RotBonds': Descriptors.NumRotatableBonds(mol),
                        })
                        
                        if len(data) >= num_molecules:
                            break
                    except Exception as e:
                        logger.warning(f"Error calculating properties: {e}")
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error getting ZINC tranche {tranche}: {e}")
    
    # Save to file
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    file_extension = os.path.splitext(output_file)[1].lower()
    if file_extension == '.csv':
        df.to_csv(output_file, index=False)
    elif file_extension == '.json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(df.to_dict('records'), f, indent=2)
    else:
        logger.warning(f"Unsupported file extension: {file_extension}, saving as CSV")
        df.to_csv(output_file.rsplit('.', 1)[0] + '.csv', index=False)
    
    logger.info(f"Downloaded {len(data)} molecules from ZINC to {output_file}")
    
    return df

def create_custom_dataset(output_file: str, num_molecules: int = 1000, random_seed: int = 42):
    """
    Create a custom dataset with drug-like molecules.
    
    Args:
        output_file: Path to save the dataset
        num_molecules: Number of molecules to create
        random_seed: Random seed for reproducibility
    """
    logger.info(f"Creating custom dataset with {num_molecules} molecules")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Define common fragments
    fragments = {
        'rings': [
            'c1ccccc1', 'C1CCCCC1', 'c1cccnc1', 'c1ccncc1', 
            'c1ccsc1', 'c1ccoc1', 'C1CCNCC1', 'c1cncnc1',
        ],
        'functional_groups': [
            'C(=O)O', 'CN', 'C(=O)N', 'CO', 'CS', 'CF', 'CCl', 'CBr',
            'C#N', 'C=O', 'C=C', 'C#C', 'CS(=O)(=O)O', 'CNS(=O)(=O)C',
        ],
        'linkers': [
            'CC', 'CCC', 'CCCC', 'CCCCC', 'CNC', 'COC', 'CSC', 'CC(=O)C',
        ],
    }
    
    # Generate molecules
    data = []
    unique_smiles = set()
    
    # Try to generate the requested number of molecules
    attempts = 0
    max_attempts = num_molecules * 10
    
    while len(data) < num_molecules and attempts < max_attempts:
        attempts += 1
        
        try:
            # Randomly select fragments
            ring = np.random.choice(fragments['rings'])
            func_group = np.random.choice(fragments['functional_groups'])
            linker = np.random.choice(fragments['linkers'])
            
            # Randomly order fragments
            order = np.random.permutation(3)
            fragments_list = [ring, func_group, linker]
            ordered_fragments = [fragments_list[i] for i in order]
            
            # Connect fragments
            smiles = '.'.join(ordered_fragments)
            
            # Convert to molecule and back to canonical SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            # Sanitize molecule
            try:
                Chem.SanitizeMol(mol)
            except:
                continue
                
            # Convert to single molecule if possible
            mol = AllChem.MolFromSmiles(Chem.MolToSmiles(mol))
            if mol is None:
                continue
                
            canonical_smiles = Chem.MolToSmiles(mol)
            
            # Skip if we've already seen this SMILES
            if canonical_smiles in unique_smiles:
                continue
                
            unique_smiles.add(canonical_smiles)
            
            # Calculate properties
            try:
                logP = Descriptors.MolLogP(mol)
                TPSA = Descriptors.TPSA(mol)
                MolWt = Descriptors.MolWt(mol)
                qed = QED.qed(mol)
                
                # Create entry
                data.append({
                    'SMILES': canonical_smiles,
                    'logP': logP,
                    'TPSA': TPSA,
                    'MolWt': MolWt,
                    'QED': qed,
                    'HBA': Descriptors.NumHAcceptors(mol),
                    'HBD': Descriptors.NumHDonors(mol),
                    'RotBonds': Descriptors.NumRotatableBonds(mol),
                })
                
            except Exception as e:
                logger.debug(f"Error calculating properties: {e}")
                
        except Exception as e:
            logger.debug(f"Error generating molecule: {e}")
    
    logger.info(f"Generated {len(data)} unique molecules in {attempts} attempts")
    
    # Save to file
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    file_extension = os.path.splitext(output_file)[1].lower()
    if file_extension == '.csv':
        df.to_csv(output_file, index=False)
    elif file_extension == '.json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(df.to_dict('records'), f, indent=2)
    else:
        logger.warning(f"Unsupported file extension: {file_extension}, saving as CSV")
        df.to_csv(output_file.rsplit('.', 1)[0] + '.csv', index=False)
    
    logger.info(f"Created custom dataset with {len(data)} molecules at {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Create example datasets for AnyMol-MoleculeSTM')
    
    parser.add_argument('--output_dir', type=str, default='./data/examples',
                        help='Directory to save example datasets')
    parser.add_argument('--dataset_type', type=str, default='custom',
                        choices=['custom', 'chembl', 'zinc'],
                        help='Type of dataset to create')
    parser.add_argument('--num_molecules', type=int, default=1000,
                        help='Number of molecules to include in dataset')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize dataset properties and sample molecules')
    parser.add_argument('--preprocess', action='store_true',
                        help='Preprocess the dataset')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate dataset
    output_file = os.path.join(args.output_dir, f"{args.dataset_type}_dataset.csv")
    
    if args.dataset_type == 'custom':
        df = create_custom_dataset(
            output_file=output_file,
            num_molecules=args.num_molecules,
            random_seed=args.random_seed,
        )
    elif args.dataset_type == 'chembl':
        df = download_chembl_subset(
            output_file=output_file,
            num_molecules=args.num_molecules,
            random_seed=args.random_seed,
        )
    elif args.dataset_type == 'zinc':
        df = download_zinc_subset(
            output_file=output_file,
            num_molecules=args.num_molecules,
            random_seed=args.random_seed,
        )
    
    # Preprocess dataset if requested
    if args.preprocess:
        preprocessed_file = os.path.join(args.output_dir, f"{args.dataset_type}_dataset_processed.csv")
        preprocess_dataset(
            input_file=output_file,
            output_file=preprocessed_file,
            canonicalize=True,
            filter_invalid=True,
            calculate_props=True,
            property_names=["logP", "TPSA", "MolWt", "QED"],
            remove_duplicates=True,
        )
        
        # Update output file
        output_file = preprocessed_file
    
    # Visualize dataset if requested
    if args.visualize:
        # Visualize property distributions
        property_viz_file = os.path.join(args.output_dir, f"{args.dataset_type}_properties.png")
        visualize_property_distribution(
            data=output_file,
            property_names=["logP", "TPSA", "MolWt", "QED"],
            output_file=property_viz_file,
        )
        
        # Visualize sample molecules
        smiles_list = load_smiles_from_file(output_file)
        sample_size = min(20, len(smiles_list))
        sample_indices = np.random.choice(len(smiles_list), sample_size, replace=False)
        
        sample_smiles = [smiles_list[i] for i in sample_indices]
        molecule_viz_file = os.path.join(args.output_dir, f"{args.dataset_type}_molecules.png")
        
        visualize_molecule_grid(
            smiles_list=sample_smiles,
            output_file=molecule_viz_file,
        )

if __name__ == '__main__':
    main()
