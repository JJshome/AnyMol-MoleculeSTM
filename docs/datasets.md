# Datasets for AnyMol-MoleculeSTM

This document describes the datasets and data loading methods available for AnyMol-MoleculeSTM.

## Data Sources

AnyMol-MoleculeSTM supports multiple ways to acquire molecular data for training and evaluation:

### 1. ChEMBL Database

[ChEMBL](https://www.ebi.ac.uk/chembl/) is a manually curated database of bioactive molecules with drug-like properties. It brings together chemical, bioactivity and genomic data to aid the translation of genomic information into effective new drugs.

**How to Access**:
- Uses the `chembl_webresource_client` Python library to query ChEMBL's web API
- Retrieves real molecules with their associated properties and bioactivity data
- Requires internet connection
- Automatically calculates missing molecular properties

**Installation Requirement**:
```bash
pip install chembl_webresource_client
```

**Example Usage**:
```bash
python examples/create_dataset.py --dataset_type chembl --num_molecules 5000 --output_dir ./data/chembl
```

**Data Processing**:
1. Queries the ChEMBL API for molecules with specific properties (e.g., LogP values)
2. Randomly selects a subset of the available molecules
3. Extracts SMILES strings and known properties
4. Validates molecular structures using RDKit
5. Calculates additional molecular properties if not available
6. Saves the dataset to the specified output directory

### 2. ZINC Database

[ZINC](https://zinc.docking.org/) is a free database of commercially-available compounds for virtual screening. It contains over 230 million purchasable compounds in ready-to-dock, 3D formats.

**How to Access**:
- Uses the `requests` library to download data from ZINC's web server
- Accesses "tranches" (subsets) of the ZINC database
- Requires internet connection
- Calculates molecular properties using RDKit

**Installation Requirement**:
```bash
pip install requests
```

**Example Usage**:
```bash
python examples/create_dataset.py --dataset_type zinc --num_molecules 3000 --output_dir ./data/zinc
```

**Data Processing**:
1. Retrieves a list of available tranches from the ZINC server
2. Randomly selects tranches to download
3. Parses the downloaded data to extract SMILES strings
4. Validates molecular structures using RDKit
5. Calculates molecular properties (LogP, TPSA, MolWt, QED, etc.)
6. Saves the dataset to the specified output directory

### 3. Custom Generation

This method algorithmically generates new molecules by combining chemical fragments in a valid manner. It doesn't depend on external data sources and can be used offline.

**How it Works**:
- Uses predefined chemical fragments (rings, functional groups, linkers)
- Combines fragments randomly to create new molecules
- Validates structures using RDKit
- Calculates molecular properties
- No internet connection required

**Example Usage**:
```bash
python examples/create_dataset.py --dataset_type custom --num_molecules 2000 --output_dir ./data/custom
```

**Data Processing**:
1. Randomly selects chemical fragments from predefined lists
2. Combines the fragments to create SMILES strings
3. Validates and sanitizes the molecular structures
4. Converts to canonical SMILES to ensure uniqueness
5. Calculates molecular properties
6. Saves the dataset to the specified output directory

## Data Format

All datasets are saved in a standardized format with the following properties:

- **SMILES**: The canonical SMILES representation of the molecule
- **logP**: Calculated octanol-water partition coefficient
- **TPSA**: Topological Polar Surface Area
- **MolWt**: Molecular Weight
- **QED**: Quantitative Estimate of Drug-likeness
- Other properties may include:
  - **HBA**: Number of Hydrogen Bond Acceptors
  - **HBD**: Number of Hydrogen Bond Donors
  - **RotBonds**: Number of Rotatable Bonds
  - **zinc_id** or **chembl_id**: Identifier from the original database (if applicable)

## Data Preprocessing

The `examples/create_dataset.py` script includes several preprocessing options:

- **--preprocess**: Enable additional preprocessing steps
- **--visualize**: Generate visualizations of the dataset properties and sample molecules

When preprocessing is enabled, the following steps are performed:

1. **Canonicalization**: Convert SMILES to canonical form
2. **Validation**: Remove invalid molecules
3. **Property Calculation**: Calculate missing molecular properties
4. **Deduplication**: Remove duplicate molecules

## Custom Data Loading

If you have your own molecular dataset, you can extend the script to load data from local files. See the [examples/notebooks/data_loading.ipynb](../examples/notebooks/data_loading.ipynb) notebook for examples of how to load custom datasets.

## Using the Dataset with AnyMol-MoleculeSTM

After creating a dataset, you can use it with AnyMol-MoleculeSTM's model initialization system:

```python
from anymol.initialization import AnyMolSTMInitializer

# Initialize the system with your dataset
initializer = AnyMolSTMInitializer()
initializer.prepare_datasets(data_file="./data/custom/custom_dataset.csv")

# Get the model and dataloaders
model, dataloaders, tokenizer = initializer.initialize_system()
```

## Dataset Statistics

Typical datasets created with the provided methods have the following characteristics:

| Source  | Size Range     | Avg. MW   | Avg. LogP | Avg. QED | Notes                              |
|---------|----------------|-----------|-----------|----------|-----------------------------------|
| ChEMBL  | 1K-1M+         | 350-450   | 2.0-4.0   | 0.4-0.6  | Bioactive, drug-like compounds    |
| ZINC    | 1K-230M+       | 300-400   | 1.5-3.5   | 0.5-0.7  | Purchasable, diverse compounds    |
| Custom  | 100-10K        | 200-350   | 1.0-3.0   | 0.3-0.6  | Algorithmically generated compounds|

## Additional Resources

- [RDKit Documentation](https://www.rdkit.org/docs/index.html): Used for molecular processing
- [ChEMBL Web Services Documentation](https://chembl.gitbook.io/chembl-interface-documentation/web-services): API reference for ChEMBL
- [ZINC Database Documentation](https://zinc.docking.org/documentation/): Information about ZINC database
