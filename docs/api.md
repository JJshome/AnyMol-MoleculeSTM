# AnyMol-MoleculeSTM API Reference

This document provides a comprehensive reference for the AnyMol-MoleculeSTM API, including all classes, methods, and functions available for molecular design, generation, and property prediction.

## Table of Contents

- [Installation](#installation)
- [Core API](#core-api)
  - [AnyMolSTM](#anymolstm)
  - [Initialization](#initialization)
  - [Training](#training)
  - [Inference](#inference)
- [Components API](#components-api)
  - [Structure Encoders](#structure-encoders)
  - [Text Encoders](#text-encoders)
  - [STM Module](#stm-module)
  - [Molecular Generation](#molecular-generation)
- [Utilities](#utilities)
  - [Molecule Processing](#molecule-processing)
  - [Visualization](#visualization)
  - [Performance Metrics](#performance-metrics)
  - [Data Management](#data-management)
- [Advanced Usage](#advanced-usage)
  - [Custom Tasks](#custom-tasks)
  - [Model Customization](#model-customization)
  - [Integration with External Tools](#integration-with-external-tools)

## Installation

```bash
# Install from PyPI
pip install anymol-moleculestm

# Install from source
git clone https://github.com/JJshome/AnyMol-MoleculeSTM.git
cd AnyMol-MoleculeSTM
pip install -e .
```

## Core API

### AnyMolSTM

The main class for working with the AnyMol-MoleculeSTM framework.

```python
from anymol import AnyMolSTM

# Initialize with default settings
model = AnyMolSTM()

# Initialize with custom configuration
model = AnyMolSTM(
    hidden_dim=1024,
    n_layers=16,
    n_heads=16,
    dropout=0.1,
    use_gru_decoder=True,
    property_prediction=["logP", "solubility", "binding_affinity"]
)
```

#### Key Methods

```python
# Load pre-trained weights
model.load_pretrained(model_name="default")  # Options: "default", "large", "property_focused"

# Generate molecules with target properties
molecules = model.generate(
    target_properties={"solubility": 3.5, "logP": 2.1},
    num_molecules=10,
    diversity_threshold=0.7,
    validity_check=True
)

# Predict properties for existing molecules
properties = model.predict_properties(
    molecules=["C1=CC=CC=C1", "CC(=O)OC1=CC=CC=C1"],  # SMILES strings
    properties=["logP", "solubility"]
)

# Encode molecules into latent space
embeddings = model.encode_molecules(
    molecules=["C1=CC=CC=C1", "CC(=O)OC1=CC=CC=C1"],
    return_text_embedding=False,  # Whether to return text embeddings as well
    batch_size=32
)

# Optimize molecules for target properties
optimized_molecules = model.optimize(
    starting_molecules=["C1=CC=CC=C1"],
    target_properties={"logP": 2.5, "QED": 0.8},
    optimization_steps=100,
    learning_rate=0.01
)

# Visualize molecules
model.visualize(
    molecules,
    property_display=True,
    highlight_substructures=True,
    save_path="molecules.png"
)
```

### Initialization

The initialization module provides utilities for setting up the AnyMol-MoleculeSTM system.

```python
from anymol.initialization import AnyMolSTMInitializer

# Initialize the system
initializer = AnyMolSTMInitializer(
    config_path="configs/default.yaml",  # Path to configuration file
    seed=42,  # Random seed for reproducibility
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Prepare datasets
initializer.prepare_datasets(
    data_file="./data/custom/custom_dataset.csv",
    split_ratio=[0.8, 0.1, 0.1],  # Train, validation, test
    preprocess=True,
    cache=True
)

# Initialize system components
model, dataloaders, tokenizer = initializer.initialize_system(
    load_checkpoint=True,
    checkpoint_path="checkpoints/latest.pt"
)
```

### Training

The training module provides utilities for training and fine-tuning models.

```python
from anymol.training import Trainer

# Initialize trainer
trainer = Trainer(
    model=model,
    dataloaders=dataloaders,
    optimizer_type="adam",
    learning_rate=1e-4,
    weight_decay=1e-6,
    scheduler="cosine",
    gradient_clipping=1.0
)

# Train the model
trainer.train(
    epochs=100,
    early_stopping=True,
    patience=10,
    checkpoint_dir="checkpoints/",
    log_interval=10,
    validation_interval=1
)

# Fine-tune on specific task
trainer.fine_tune(
    task="binding_affinity",
    task_data="data/binding_data.csv",
    epochs=20,
    freeze_encoder=True
)

# Evaluate model
metrics = trainer.evaluate(
    test_dataloader=dataloaders["test"],
    metrics=["rmse", "mae", "r2", "accuracy"]
)
```

### Inference

The inference module provides utilities for using trained models.

```python
from anymol.inference import Predictor

# Initialize predictor
predictor = Predictor(
    model=model,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=32
)

# Predict properties
predictions = predictor.predict(
    molecules=["C1=CC=CC=C1", "CC(=O)OC1=CC=CC=C1"],
    properties=["logP", "solubility"]
)

# Generate molecules
generated_molecules = predictor.generate(
    num_molecules=100,
    target_properties={"drug_likeness": 0.7},
    temperature=0.8
)

# Get similarity between molecules
similarities = predictor.calculate_similarity(
    molecule_1="C1=CC=CC=C1",
    molecule_2="CC1=CC=CC=C1",
    method="tanimoto"  # Options: "tanimoto", "embedding", "scaffold"
)
```

## Components API

### Structure Encoders

```python
from anymol.encoders import StructureEncoder

# Initialize encoder
encoder = StructureEncoder(
    hidden_dim=768,
    n_layers=12,
    n_heads=12,
    dropout=0.1,
    atom_features=["element", "hybridization", "aromaticity"],
    bond_features=["bond_type", "conjugated", "in_ring"]
)

# Encode molecule
embedding = encoder.encode(
    molecule="C1=CC=CC=C1",
    return_attention=False
)
```

### Text Encoders

```python
from anymol.encoders import TextEncoder

# Initialize encoder
encoder = TextEncoder(
    model_type="bert",  # Options: "bert", "roberta", "scibert"
    hidden_dim=768,
    n_layers=12,
    fine_tune=True
)

# Encode molecular description
embedding = encoder.encode(
    text="Benzene is an aromatic hydrocarbon with a ring of six carbon atoms",
    return_pooled=True
)
```

### STM Module

```python
from anymol.stm import STMModule

# Initialize module
stm_module = STMModule(
    hidden_dim=768,
    temperature=0.07,
    projection_dim=256,
    use_momentum=True,
    momentum_update_rate=0.99
)

# Match structure and text embeddings
similarity_scores = stm_module.match(
    structure_embeddings=structure_embeddings,
    text_embeddings=text_embeddings
)

# Train with contrastive loss
loss = stm_module.contrastive_loss(
    structure_embeddings=structure_embeddings,
    text_embeddings=text_embeddings,
    labels=labels,
    hard_negatives=True
)
```

### Molecular Generation

```python
from anymol.generation import MoleculeGenerator

# Initialize generator
generator = MoleculeGenerator(
    hidden_dim=768,
    gru_layers=3,
    max_length=100,
    sampling_method="beam"  # Options: "beam", "greedy", "topk", "nucleus"
)

# Generate from scratch
molecules = generator.generate(
    num_molecules=10,
    target_properties={"solubility": 3.5},
    temperature=0.8
)

# Generate from seed molecules
modified_molecules = generator.modify(
    seed_molecules=["C1=CC=CC=C1"],
    modification_type="scaffold",  # Options: "scaffold", "functional_group", "full"
    constraints=["keep_rings=true", "max_atoms=30"]
)
```

## Utilities

### Molecule Processing

```python
from anymol.utils.molecule import MoleculeProcessor

# Initialize processor
processor = MoleculeProcessor()

# Convert between formats
smiles = processor.mol_to_smiles(mol_obj)
mol_obj = processor.smiles_to_mol("C1=CC=CC=C1")
inchi = processor.mol_to_inchi(mol_obj)

# Calculate molecular properties
properties = processor.calculate_properties(
    mol_obj,
    properties=["logP", "molecular_weight", "num_rotatable_bonds"]
)

# Check validity
is_valid = processor.validate_molecule("C1=CC=CC=C1")

# Canonicalize SMILES
canonical_smiles = processor.canonicalize("C=1C=CC=CC1")
```

### Visualization

```python
from anymol.utils.visualization import MoleculeVisualizer

# Initialize visualizer
visualizer = MoleculeVisualizer()

# Visualize single molecule
visualizer.visualize_molecule(
    molecule="C1=CC=CC=C1",
    highlight_atoms=[0, 1, 2],
    property_display=True,
    save_path="molecule.png"
)

# Visualize multiple molecules
visualizer.visualize_grid(
    molecules=["C1=CC=CC=C1", "CC1=CC=CC=C1"],
    labels=["Benzene", "Toluene"],
    grid_size=(2, 3),
    save_path="grid.png"
)

# Create animation of optimization process
visualizer.create_optimization_animation(
    molecules_sequence=optimization_trajectory,
    property_values=property_trajectory,
    save_path="optimization.gif"
)
```

### Performance Metrics

```python
from anymol.utils.metrics import PerformanceEvaluator

# Initialize evaluator
evaluator = PerformanceEvaluator()

# Evaluate property predictions
metrics = evaluator.evaluate_predictions(
    predicted=predicted_values,
    actual=actual_values,
    metrics=["rmse", "mae", "r2", "spearman"]
)

# Evaluate generation quality
diversity_score = evaluator.calculate_diversity(generated_molecules)
novelty_score = evaluator.calculate_novelty(
    generated_molecules=generated_molecules,
    reference_molecules=training_molecules
)
```

### Data Management

```python
from anymol.utils.data import DataManager

# Initialize manager
data_manager = DataManager()

# Load dataset
dataset = data_manager.load_dataset(
    path="data/dataset.csv",
    molecule_column="SMILES",
    property_columns=["logP", "solubility"],
    split=True,
    split_ratio=[0.8, 0.1, 0.1]
)

# Create dataloaders
dataloaders = data_manager.create_dataloaders(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Preprocess dataset
processed_dataset = data_manager.preprocess(
    dataset=dataset,
    normalize=True,
    augmentation=True,
    cache=True
)
```

## Advanced Usage

### Custom Tasks

```python
from anymol.tasks import CustomTask

# Define custom task
binding_task = CustomTask(
    name="protein_binding",
    prediction_type="regression",
    loss_function="mse",
    metrics=["rmse", "r2"],
    target_range=[0, 10]
)

# Add task to model
model.add_task(binding_task)

# Train on custom task
model.train_task(
    task_name="protein_binding",
    dataset="data/binding_data.csv",
    epochs=50
)
```

### Model Customization

```python
from anymol.customization import ModelCustomizer

# Initialize customizer
customizer = ModelCustomizer(base_model=model)

# Replace structure encoder
customizer.replace_structure_encoder(
    new_encoder=custom_encoder,
    transfer_weights=True
)

# Add property predictor
customizer.add_property_predictor(
    property_name="new_property",
    predictor_architecture="mlp",
    hidden_layers=[512, 256, 128]
)

# Get customized model
custom_model = customizer.get_model()
```

### Integration with External Tools

```python
from anymol.integration import ExternalToolIntegrator

# Initialize integrator
integrator = ExternalToolIntegrator()

# RDKit integration
rdkit_mol = integrator.convert_to_rdkit(molecule)
rdkit_properties = integrator.calculate_with_rdkit(molecule, properties=["logP", "MolWt"])

# PyTorch Geometric integration
pyg_graph = integrator.convert_to_pyg(molecule)
```

For more detailed information and examples, refer to the code documentation in the source files or check the [examples directory](../examples/) for practical usage examples.