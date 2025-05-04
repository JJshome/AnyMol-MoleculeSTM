"""
Initialization and integration of AnyMol-MoleculeSTM model and datasets.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging
import json
import yaml
from torch.utils.data import DataLoader

from .models.stm_model import AnyMolSTMModel, MoleculeSTM
from .data.dataset import MoleculeTokenizer, MoleculeDataset, MoleculeCollator
from .data.utils import (
    load_smiles_from_file,
    canonicalize_smiles,
    filter_valid_smiles,
    split_dataset,
    calculate_molecular_properties,
    visualize_property_distribution,
    visualize_molecule_grid,
    preprocess_dataset,
)

logger = logging.getLogger(__name__)

class AnyMolSTMInitializer:
    """
    Initializer for AnyMol-MoleculeSTM model and datasets.
    """
    def __init__(
        self,
        config_file: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize AnyMol-MoleculeSTM system.
        
        Args:
            config_file: Path to configuration file (YAML or JSON)
            config_dict: Configuration dictionary (overrides config_file)
        """
        # Load configuration
        self.config = self._load_config(config_file, config_dict)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.datasets = {}
        self.dataloaders = {}
        
        logger.info("AnyMol-MoleculeSTM initializer created")
    
    def _load_config(
        self,
        config_file: Optional[str],
        config_dict: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Load configuration from file or dictionary."""
        # Default configuration
        default_config = {
            # Logging
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": None,
            },
            
            # Model configuration
            "model": {
                "embedding_dim": 768,
                "hidden_dim": 512,
                "num_layers": 6,
                "num_heads": 8,
                "dropout": 0.1,
                "max_seq_length": 256,
                "vocabulary_size": 1000,
                "property_hidden_dims": [512, 256, 128],
                "property_names": ["logP", "TPSA", "MolWt", "QED"],
                "anymol_integration_method": "attention",
                "checkpoint_path": None,
            },
            
            # Data configuration
            "data": {
                "train_file": None,
                "val_file": None,
                "test_file": None,
                "data_dir": "./data",
                "processed_dir": "./data/processed",
                "smiles_column": "SMILES",
                "vocab_file": "./data/vocabulary.txt",
                "preprocess": True,
                "canonicalize": True,
                "filter_invalid": True,
                "calculate_props": True,
                "remove_duplicates": True,
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "batch_size": 32,
                "num_workers": 4,
            },
            
            # Training configuration
            "training": {
                "output_dir": "./outputs",
                "learning_rate": 1e-4,
                "weight_decay": 1e-6,
                "max_epochs": 100,
                "patience": 10,
                "save_steps": 1000,
                "eval_steps": 500,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 1000,
                "fp16": False,
                "seed": 42,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
        }
        
        # Load from file if provided
        config = default_config.copy()
        
        if config_file and os.path.exists(config_file):
            file_extension = os.path.splitext(config_file)[1].lower()
            
            try:
                if file_extension == '.yaml' or file_extension == '.yml':
                    with open(config_file, 'r', encoding='utf-8') as f:
                        loaded_config = yaml.safe_load(f)
                elif file_extension == '.json':
                    with open(config_file, 'r', encoding='utf-8') as f:
                        loaded_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file extension: {file_extension}")
                
                # Update configuration with loaded values
                self._update_nested_dict(config, loaded_config)
                logger.info(f"Loaded configuration from {config_file}")
                
            except Exception as e:
                logger.error(f"Error loading configuration from {config_file}: {e}")
        
        # Update with provided dictionary if any
        if config_dict:
            self._update_nested_dict(config, config_dict)
        
        return config
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Recursively update nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_config = self.config["logging"]
        
        logging.basicConfig(
            level=getattr(logging, log_config["level"]),
            format=log_config["format"],
            filename=log_config["file"],
        )
    
    def initialize_tokenizer(self, smiles_list: Optional[List[str]] = None):
        """
        Initialize tokenizer from vocabulary file or SMILES list.
        
        Args:
            smiles_list: List of SMILES strings to build vocabulary (optional)
        """
        vocab_file = self.config["data"]["vocab_file"]
        
        # Initialize tokenizer
        self.tokenizer = MoleculeTokenizer(
            vocab_file=vocab_file if os.path.exists(vocab_file) else None,
            max_length=self.config["model"]["max_seq_length"],
        )
        
        # Build vocabulary from SMILES if provided
        if smiles_list and (not os.path.exists(vocab_file) or len(self.tokenizer.token2idx) <= 5):
            logger.info(f"Building vocabulary from {len(smiles_list)} SMILES")
            self.tokenizer.build_vocab_from_smiles(smiles_list)
            
            # Save vocabulary
            os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
            self.tokenizer.save_vocab(vocab_file)
            
            # Update model configuration
            self.config["model"]["vocabulary_size"] = len(self.tokenizer.token2idx)
        else:
            logger.info(f"Using existing vocabulary with {len(self.tokenizer.token2idx)} tokens")
            # Update model configuration
            self.config["model"]["vocabulary_size"] = len(self.tokenizer.token2idx)
        
        return self.tokenizer
    
    def initialize_model(self, device: Optional[str] = None):
        """
        Initialize the AnyMol-MoleculeSTM model.
        
        Args:
            device: Device to load the model on (overrides config)
        """
        model_config = self.config["model"]
        
        # Determine device
        if device is None:
            device = self.config["training"]["device"]
        
        # Check if tokenizer exists
        if self.tokenizer is None:
            logger.warning("Tokenizer not initialized, using default vocabulary size")
        
        # Create model
        self.model = AnyMolSTMModel(
            embedding_dim=model_config["embedding_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            dropout=model_config["dropout"],
            max_seq_length=model_config["max_seq_length"],
            vocabulary_size=model_config["vocabulary_size"],
            property_hidden_dims=model_config["property_hidden_dims"],
            property_names=model_config["property_names"],
            anymol_integration_method=model_config["anymol_integration_method"],
        )
        
        # Load checkpoint if provided
        if model_config["checkpoint_path"] and os.path.exists(model_config["checkpoint_path"]):
            logger.info(f"Loading model checkpoint from {model_config['checkpoint_path']}")
            checkpoint = torch.load(model_config["checkpoint_path"], map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Move model to device
        self.model.to(device)
        
        return self.model
    
    def prepare_datasets(self, data_file: Optional[str] = None, process_data: bool = True):
        """
        Prepare datasets for training.
        
        Args:
            data_file: Path to data file (overrides config)
            process_data: Whether to preprocess the data
        """
        data_config = self.config["data"]
        
        # Override data file if provided
        if data_file:
            data_config["train_file"] = data_file
        
        # Create data directories if they don't exist
        os.makedirs(data_config["data_dir"], exist_ok=True)
        os.makedirs(data_config["processed_dir"], exist_ok=True)
        
        # Preprocess data if needed
        if process_data and data_config["preprocess"] and data_config["train_file"]:
            logger.info(f"Preprocessing data from {data_config['train_file']}")
            
            # Define output file path
            input_filename = os.path.basename(data_config["train_file"])
            processed_file = os.path.join(
                data_config["processed_dir"],
                f"processed_{input_filename}"
            )
            
            # Preprocess data
            processed_df = preprocess_dataset(
                input_file=data_config["train_file"],
                output_file=processed_file,
                smiles_column=data_config["smiles_column"],
                canonicalize=data_config["canonicalize"],
                filter_invalid=data_config["filter_invalid"],
                calculate_props=data_config["calculate_props"],
                property_names=self.config["model"]["property_names"],
                remove_duplicates=data_config["remove_duplicates"],
            )
            
            # Update data file path
            data_config["train_file"] = processed_file
        
        # Split data if we only have a single file
        if data_config["train_file"] and not (data_config["val_file"] and data_config["test_file"]):
            logger.info(f"Splitting data from {data_config['train_file']}")
            
            # Define split directory
            split_dir = os.path.join(data_config["processed_dir"], "split")
            os.makedirs(split_dir, exist_ok=True)
            
            # Split data
            split_files = split_dataset(
                data_file=data_config["train_file"],
                output_dir=split_dir,
                train_ratio=data_config["train_ratio"],
                val_ratio=data_config["val_ratio"],
                test_ratio=data_config["test_ratio"],
                smiles_column=data_config["smiles_column"],
            )
            
            # Update data file paths
            data_config["train_file"] = split_files["train"]
            data_config["val_file"] = split_files["validation"]
            data_config["test_file"] = split_files["test"]
        
        # Initialize tokenizer if needed
        if self.tokenizer is None:
            logger.info("Initializing tokenizer from data")
            smiles_list = load_smiles_from_file(
                data_config["train_file"],
                smiles_column=data_config["smiles_column"]
            )
            self.initialize_tokenizer(smiles_list)
        
        # Create datasets
        logger.info("Creating datasets")
        
        if data_config["train_file"]:
            self.datasets["train"] = MoleculeDataset(
                data_file=data_config["train_file"],
                tokenizer=self.tokenizer,
                property_names=self.config["model"]["property_names"],
                smiles_column=data_config["smiles_column"],
                max_length=self.config["model"]["max_seq_length"],
            )
        
        if data_config["val_file"]:
            self.datasets["validation"] = MoleculeDataset(
                data_file=data_config["val_file"],
                tokenizer=self.tokenizer,
                property_names=self.config["model"]["property_names"],
                smiles_column=data_config["smiles_column"],
                max_length=self.config["model"]["max_seq_length"],
            )
        
        if data_config["test_file"]:
            self.datasets["test"] = MoleculeDataset(
                data_file=data_config["test_file"],
                tokenizer=self.tokenizer,
                property_names=self.config["model"]["property_names"],
                smiles_column=data_config["smiles_column"],
                max_length=self.config["model"]["max_seq_length"],
            )
        
        return self.datasets
    
    def create_dataloaders(self):
        """Create data loaders for training and evaluation."""
        data_config = self.config["data"]
        
        # Create collator
        collator = MoleculeCollator(pad_token_id=self.tokenizer.pad_token_id)
        
        # Create data loaders
        logger.info("Creating dataloaders")
        
        if "train" in self.datasets:
            self.dataloaders["train"] = DataLoader(
                self.datasets["train"],
                batch_size=data_config["batch_size"],
                shuffle=True,
                num_workers=data_config["num_workers"],
                collate_fn=collator,
                pin_memory=True,
            )
        
        if "validation" in self.datasets:
            self.dataloaders["validation"] = DataLoader(
                self.datasets["validation"],
                batch_size=data_config["batch_size"],
                shuffle=False,
                num_workers=data_config["num_workers"],
                collate_fn=collator,
                pin_memory=True,
            )
        
        if "test" in self.datasets:
            self.dataloaders["test"] = DataLoader(
                self.datasets["test"],
                batch_size=data_config["batch_size"],
                shuffle=False,
                num_workers=data_config["num_workers"],
                collate_fn=collator,
                pin_memory=True,
            )
        
        return self.dataloaders
    
    def initialize_system(self, data_file: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the complete AnyMol-MoleculeSTM system.
        
        Args:
            data_file: Path to data file (overrides config)
            device: Device to load the model on (overrides config)
            
        Returns:
            Tuple of (model, dataloaders, tokenizer)
        """
        logger.info("Initializing AnyMol-MoleculeSTM system")
        
        # Prepare datasets and tokenizer
        self.prepare_datasets(data_file)
        
        # Initialize model
        self.initialize_model(device)
        
        # Create dataloaders
        self.create_dataloaders()
        
        logger.info("System initialization complete")
        
        return self.model, self.dataloaders, self.tokenizer
    
    def visualize_dataset_properties(self, dataset_key: str = "train", output_file: Optional[str] = None):
        """
        Visualize property distributions in a dataset.
        
        Args:
            dataset_key: Dataset to visualize ('train', 'validation', or 'test')
            output_file: Path to save the visualization
        """
        if dataset_key not in self.datasets:
            raise ValueError(f"Dataset '{dataset_key}' not found")
        
        # Get data file path
        data_config = self.config["data"]
        data_file = data_config[f"{dataset_key}_file"]
        
        # Visualize properties
        visualize_property_distribution(
            data=data_file,
            property_names=self.config["model"]["property_names"],
            output_file=output_file,
            smiles_column=data_config["smiles_column"],
        )
    
    def visualize_sample_molecules(
        self,
        dataset_key: str = "train",
        num_samples: int = 20,
        output_file: Optional[str] = None,
    ):
        """
        Visualize sample molecules from a dataset.
        
        Args:
            dataset_key: Dataset to visualize ('train', 'validation', or 'test')
            num_samples: Number of molecules to sample
            output_file: Path to save the visualization
        """
        if dataset_key not in self.datasets:
            raise ValueError(f"Dataset '{dataset_key}' not found")
        
        # Sample molecules
        dataset = self.datasets[dataset_key]
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        smiles_list = [dataset[i]["smiles"] for i in indices]
        
        # Visualize molecules
        visualize_molecule_grid(
            smiles_list=smiles_list,
            output_file=output_file,
        )
    
    def save_config(self, config_file: str):
        """
        Save current configuration to file.
        
        Args:
            config_file: Path to save configuration
        """
        file_extension = os.path.splitext(config_file)[1].lower()
        
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        try:
            if file_extension == '.yaml' or file_extension == '.yml':
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif file_extension == '.json':
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file extension: {file_extension}")
            
            logger.info(f"Configuration saved to {config_file}")
        
        except Exception as e:
            logger.error(f"Error saving configuration to {config_file}: {e}")
