"""
Dataset loaders for AnyMol-MoleculeSTM.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
import logging
import json
import csv
import pickle
from tqdm import tqdm
from collections import Counter

logger = logging.getLogger(__name__)

class MoleculeTokenizer:
    """
    Tokenizer for converting SMILES strings to token indices and vice versa.
    """
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        special_tokens: Dict[str, str] = {
            "PAD": "[PAD]",
            "UNK": "[UNK]",
            "CLS": "[CLS]",
            "SEP": "[SEP]",
            "MASK": "[MASK]"
        },
        max_length: int = 256,
    ):
        self.special_tokens = special_tokens
        self.max_length = max_length
        
        # Special token indices
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.mask_token_id = 4
        
        # Initialize vocabulary
        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
        else:
            self.token2idx = {
                special_tokens["PAD"]: self.pad_token_id,
                special_tokens["UNK"]: self.unk_token_id,
                special_tokens["CLS"]: self.cls_token_id,
                special_tokens["SEP"]: self.sep_token_id,
                special_tokens["MASK"]: self.mask_token_id,
            }
            self.idx2token = {v: k for k, v in self.token2idx.items()}
            
    def load_vocab(self, vocab_file: str):
        """Load vocabulary from file."""
        self.token2idx = {}
        self.idx2token = {}
        
        # Load special tokens first
        self.token2idx[self.special_tokens["PAD"]] = self.pad_token_id
        self.token2idx[self.special_tokens["UNK"]] = self.unk_token_id
        self.token2idx[self.special_tokens["CLS"]] = self.cls_token_id
        self.token2idx[self.special_tokens["SEP"]] = self.sep_token_id
        self.token2idx[self.special_tokens["MASK"]] = self.mask_token_id
        
        # Load vocabulary from file
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.strip()
                if token not in self.token2idx:
                    self.token2idx[token] = len(self.token2idx)
        
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        
        logger.info(f"Loaded vocabulary with {len(self.token2idx)} tokens")
        
    def save_vocab(self, vocab_file: str):
        """Save vocabulary to file."""
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for token in self.token2idx.keys():
                f.write(f"{token}\n")
        
        logger.info(f"Saved vocabulary with {len(self.token2idx)} tokens to {vocab_file}")
    
    def build_vocab_from_smiles(self, smiles_list: List[str], min_freq: int = 1):
        """Build vocabulary from a list of SMILES strings."""
        # Initialize with special tokens
        self.token2idx = {
            self.special_tokens["PAD"]: self.pad_token_id,
            self.special_tokens["UNK"]: self.unk_token_id,
            self.special_tokens["CLS"]: self.cls_token_id,
            self.special_tokens["SEP"]: self.sep_token_id,
            self.special_tokens["MASK"]: self.mask_token_id,
        }
        
        # Collect all tokens
        token_counts = Counter()
        
        for smiles in tqdm(smiles_list, desc="Building vocabulary"):
            tokens = self._smiles_to_tokens(smiles)
            token_counts.update(tokens)
        
        # Add tokens that meet minimum frequency
        for token, count in token_counts.items():
            if count >= min_freq and token not in self.token2idx:
                self.token2idx[token] = len(self.token2idx)
        
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        
        logger.info(f"Built vocabulary with {len(self.token2idx)} tokens")
        
    def _smiles_to_tokens(self, smiles: str) -> List[str]:
        """
        Convert SMILES to a list of tokens by atom-level tokenization.
        This is a simplified implementation; consider using RDKit or more 
        sophisticated tokenization for production use.
        """
        # For simplicity, we'll tokenize character by character
        # with some basic rules for common substructures
        tokens = []
        i = 0
        
        # Add special tokens at start
        tokens.append(self.special_tokens["CLS"])
        
        while i < len(smiles):
            # Check for two-character tokens
            if i < len(smiles) - 1:
                two_char = smiles[i:i+2]
                if two_char in ["Cl", "Br", "Si", "Se", "Na", "Li", "Al", "Ca", "Mg"]:
                    tokens.append(two_char)
                    i += 2
                    continue
            
            # Check for numbered rings (1-9)
            if smiles[i].isdigit() and i > 0 and smiles[i-1] in "[]()":
                tokens.append(smiles[i])
                i += 1
                continue
                
            # Check for atom with brackets
            if smiles[i] == '[':
                j = i
                while j < len(smiles) and smiles[j] != ']':
                    j += 1
                if j < len(smiles):  # Found closing bracket
                    tokens.append(smiles[i:j+1])
                    i = j + 1
                    continue
            
            # Default: add single character
            tokens.append(smiles[i])
            i += 1
        
        # Add special tokens at end
        tokens.append(self.special_tokens["SEP"])
        
        return tokens
    
    def encode(self, smiles: str, padding: bool = True, truncation: bool = True) -> Dict[str, torch.Tensor]:
        """
        Encode a SMILES string to token indices.
        
        Args:
            smiles: SMILES string to encode
            padding: Whether to pad the sequence to max_length
            truncation: Whether to truncate the sequence if longer than max_length
            
        Returns:
            Dictionary with input_ids, attention_mask
        """
        tokens = self._smiles_to_tokens(smiles)
        
        # Convert tokens to indices
        input_ids = []
        for token in tokens:
            if token in self.token2idx:
                input_ids.append(self.token2idx[token])
            else:
                input_ids.append(self.unk_token_id)
        
        # Truncate if needed
        if truncation and len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad if needed
        if padding:
            padding_length = self.max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [self.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
    
    def batch_encode(self, smiles_list: List[str], padding: bool = True, truncation: bool = True) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of SMILES strings to token indices.
        
        Args:
            smiles_list: List of SMILES strings to encode
            padding: Whether to pad the sequences to max_length
            truncation: Whether to truncate sequences if longer than max_length
            
        Returns:
            Dictionary with batched input_ids, attention_mask
        """
        encodings = [self.encode(smiles, padding=False, truncation=truncation) for smiles in smiles_list]
        
        # Find max length in batch
        max_length = max([len(encoding["input_ids"]) for encoding in encodings])
        max_length = min(max_length, self.max_length)
        
        # Prepare batched tensors
        batch_input_ids = []
        batch_attention_mask = []
        
        for encoding in encodings:
            input_ids = encoding["input_ids"].tolist()
            attention_mask = encoding["attention_mask"].tolist()
            
            # Truncate if needed
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
            
            # Pad if needed
            if padding:
                padding_length = max_length - len(input_ids)
                if padding_length > 0:
                    input_ids = input_ids + [self.pad_token_id] * padding_length
                    attention_mask = attention_mask + [0] * padding_length
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }
    
    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """
        Decode token indices back to a SMILES string.
        
        Args:
            token_ids: List or tensor of token indices
            
        Returns:
            SMILES string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        
        # Convert indices to tokens
        tokens = []
        for idx in token_ids:
            if idx == self.pad_token_id:
                continue
            if idx == self.sep_token_id:
                break
            if idx == self.cls_token_id:
                continue
            
            token = self.idx2token.get(idx, self.special_tokens["UNK"])
            if token != self.special_tokens["UNK"] and token not in self.special_tokens.values():
                tokens.append(token)
        
        # Join tokens to form SMILES
        smiles = "".join(tokens)
        
        return smiles

class MoleculeDataset(Dataset):
    """
    Dataset class for molecular data.
    """
    def __init__(
        self,
        data_file: str,
        tokenizer: MoleculeTokenizer,
        property_names: List[str] = ["logP", "TPSA", "MolWt", "QED"],
        smiles_column: str = "SMILES",
        max_length: int = 256,
        preprocessing_fn: Optional[Callable] = None,
        calculate_properties: bool = False,
    ):
        """
        Initialize MoleculeDataset.
        
        Args:
            data_file: Path to data file (CSV, TSV, or JSON)
            tokenizer: MoleculeTokenizer instance
            property_names: List of property names to use
            smiles_column: Name of the column containing SMILES strings
            max_length: Maximum sequence length
            preprocessing_fn: Optional function to preprocess SMILES strings
            calculate_properties: Whether to calculate properties if not present
        """
        self.tokenizer = tokenizer
        self.property_names = property_names
        self.smiles_column = smiles_column
        self.max_length = max_length
        self.preprocessing_fn = preprocessing_fn
        
        # Load data
        self.data = self._load_data(data_file)
        
        # Calculate properties if needed
        if calculate_properties:
            self._calculate_missing_properties()
        
        logger.info(f"Loaded {len(self.data)} molecules from {data_file}")
        
    def _load_data(self, data_file: str) -> List[Dict[str, Any]]:
        """Load data from file."""
        file_extension = os.path.splitext(data_file)[1].lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(data_file)
        elif file_extension == '.tsv':
            df = pd.read_csv(data_file, sep='\t')
        elif file_extension == '.json':
            with open(data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_extension == '.pkl' or file_extension == '.pickle':
            with open(data_file, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        # Convert DataFrame to list of dictionaries
        data = df.to_dict('records')
        
        # Apply preprocessing to SMILES if provided
        if self.preprocessing_fn is not None:
            for item in data:
                if self.smiles_column in item:
                    item[self.smiles_column] = self.preprocessing_fn(item[self.smiles_column])
        
        return data
    
    def _calculate_missing_properties(self):
        """Calculate missing properties for each molecule."""
        property_calculators = {
            "logP": lambda mol: Descriptors.MolLogP(mol),
            "TPSA": lambda mol: Descriptors.TPSA(mol),
            "MolWt": lambda mol: Descriptors.MolWt(mol),
            "QED": lambda mol: QED.qed(mol),
            "NumHDonors": lambda mol: Descriptors.NumHDonors(mol),
            "NumHAcceptors": lambda mol: Descriptors.NumHAcceptors(mol),
            "NumRotatableBonds": lambda mol: Descriptors.NumRotatableBonds(mol),
        }
        
        for i, item in enumerate(tqdm(self.data, desc="Calculating properties")):
            # Skip if all properties are already present
            if all(prop in item for prop in self.property_names):
                continue
            
            # Create RDKit molecule
            mol = Chem.MolFromSmiles(item[self.smiles_column])
            if mol is None:
                logger.warning(f"Could not parse SMILES: {item[self.smiles_column]}")
                # Set missing properties to NaN
                for prop in self.property_names:
                    if prop not in item:
                        item[prop] = float('nan')
                continue
            
            # Calculate missing properties
            for prop in self.property_names:
                if prop not in item and prop in property_calculators:
                    try:
                        item[prop] = property_calculators[prop](mol)
                    except Exception as e:
                        logger.warning(f"Error calculating {prop}: {e}")
                        item[prop] = float('nan')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get SMILES
        smiles = item[self.smiles_column]
        
        # Encode SMILES
        encoding = self.tokenizer.encode(smiles)
        
        # Get properties
        properties = {}
        for prop in self.property_names:
            if prop in item:
                properties[prop] = item.get(prop, float('nan'))
        
        # Convert properties to tensor
        property_tensor = torch.tensor(
            [properties.get(prop, float('nan')) for prop in self.property_names],
            dtype=torch.float
        )
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "properties": property_tensor,
            "smiles": smiles,
        }

class MoleculeCollator:
    """
    Collate function for batching MoleculeDataset samples.
    """
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        # Get batch size
        batch_size = len(batch)
        
        # Get maximum sequence length in this batch
        max_length = max(len(sample["input_ids"]) for sample in batch)
        
        # Initialize tensors
        input_ids = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        
        # Fill tensors
        properties = []
        smiles = []
        
        for i, sample in enumerate(batch):
            seq_length = len(sample["input_ids"])
            input_ids[i, :seq_length] = sample["input_ids"]
            attention_mask[i, :seq_length] = sample["attention_mask"]
            properties.append(sample["properties"])
            smiles.append(sample["smiles"])
        
        # Stack properties
        properties_tensor = torch.stack(properties) if properties else None
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "properties": properties_tensor,
            "smiles": smiles,
        }
