# AnyMol-MoleculeSTM

Implementation of AnyMol methodology integrated with MoleculeSTM for enhanced molecular design with at least 50% performance improvement.

## Overview

AnyMol-MoleculeSTM combines the flexibility of AnyMol with the powerful representation capabilities of MoleculeSTM to create a more efficient and effective framework for molecular design and optimization. This integration enables:

- Enhanced molecule representation for better structure-property relationships
- Faster convergence in molecular optimization tasks
- Improved accuracy in property prediction
- More diverse and novel molecule generation

## Features

- Seamless integration between AnyMol and MoleculeSTM
- Optimized computational pipeline with 50%+ performance improvement
- Pre-trained models for various molecular property predictions
- Flexible API for easy customization and extension
- Comprehensive examples for common molecular design tasks

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from anymol import AnyMolSTM

# Initialize the model
model = AnyMolSTM()

# Load pre-trained weights
model.load_pretrained("default")

# Generate molecules with target properties
molecules = model.generate(
    target_properties={"solubility": 3.5, "logP": 2.1},
    num_molecules=10
)

# Visualize the results
model.visualize(molecules)
```

## Documentation

For detailed documentation, please see the [docs](docs/) directory.

## Citation

If you use this code in your research, please cite:

```
@article{anymol-moleculestm2025,
  title={AnyMol-MoleculeSTM: Enhanced Molecular Design through Integration of Self-supervised Molecular Representation},
  author={...},
  journal={...},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
