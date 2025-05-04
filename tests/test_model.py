"""
Tests for AnyMol-MoleculeSTM model.
"""

import os
import sys
import pytest
import torch
import numpy as np
from rdkit import Chem

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from anymol import AnyMolSTM
from anymol.utils import (
    smiles_to_mol,
    mol_to_smiles,
    compute_molecular_fingerprints,
    compute_molecular_descriptors,
    validate_molecules,
    filter_valid_molecules,
    calculate_similarity,
    calculate_pairwise_similarities,
)

# Test data
TEST_SMILES = [
    "CCO",  # Ethanol
    "CC(=O)O",  # Acetic acid
    "c1ccccc1",  # Benzene
    "Cc1ccccc1",  # Toluene
    "c1ccccc1O",  # Phenol
]

INVALID_SMILES = [
    "invalid",
    "C1CC",
    "C1CC1C",
]

class TestAnyMolSTM:
    """Tests for AnyMolSTM model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = AnyMolSTM()
    
    def test_initialization(self):
        """Test model initialization."""
        assert isinstance(self.model, AnyMolSTM)
        assert self.model.embedding_dim == 768
        assert self.model.hidden_dim == 512
        assert self.model.num_layers == 6
        assert self.model.num_heads == 8
        assert self.model.dropout == 0.1
    
    def test_load_pretrained(self):
        """Test loading pretrained model."""
        # This is just testing the method doesn't error since we don't have actual weights
        self.model.load_pretrained("default")
    
    def test_encode(self):
        """Test molecule encoding."""
        embeddings = self.model.encode(TEST_SMILES)
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (len(TEST_SMILES), self.model.embedding_dim)
    
    def test_decode(self):
        """Test molecule decoding."""
        embeddings = self.model.encode(TEST_SMILES)
        decoded = self.model.decode(embeddings)
        assert isinstance(decoded, list)
        assert len(decoded) == len(TEST_SMILES)
        assert all(isinstance(s, str) for s in decoded)
    
    def test_generate(self):
        """Test molecule generation."""
        num_molecules = 3
        target_properties = {"LogP": 2.5, "MolWt": 200.0}
        generated = self.model.generate(target_properties, num_molecules=num_molecules)
        assert isinstance(generated, list)
        assert len(generated) == num_molecules
        assert all(isinstance(s, str) for s in generated)
    
    def test_optimize(self):
        """Test molecule optimization."""
        target_properties = {"LogP": 2.5, "MolWt": 200.0}
        optimized = self.model.optimize(TEST_SMILES, target_properties)
        assert isinstance(optimized, list)
        assert len(optimized) == len(TEST_SMILES)
    
    def test_predict_properties(self):
        """Test property prediction."""
        properties = ["LogP", "MolWt"]
        predictions = self.model.predict_properties(TEST_SMILES, properties)
        assert isinstance(predictions, dict)
        assert all(prop in predictions for prop in properties)
        assert all(len(predictions[prop]) == len(TEST_SMILES) for prop in properties)

class TestUtils:
    """Tests for utility functions."""
    
    def test_smiles_to_mol(self):
        """Test SMILES to molecule conversion."""
        for smiles in TEST_SMILES:
            mol = smiles_to_mol(smiles)
            assert mol is not None
            assert isinstance(mol, Chem.Mol)
        
        for smiles in INVALID_SMILES:
            mol = smiles_to_mol(smiles)
            assert mol is None
    
    def test_mol_to_smiles(self):
        """Test molecule to SMILES conversion."""
        for smiles in TEST_SMILES:
            mol = smiles_to_mol(smiles)
            converted = mol_to_smiles(mol)
            assert converted is not None
            assert isinstance(converted, str)
            
            # Convert back to molecule to check validity
            assert smiles_to_mol(converted) is not None
    
    def test_compute_molecular_fingerprints(self):
        """Test fingerprint computation."""
        for fp_type in ["morgan", "maccs", "rdkit"]:
            fps = compute_molecular_fingerprints(TEST_SMILES, fp_type=fp_type)
            assert isinstance(fps, np.ndarray)
            assert fps.shape[0] == len(TEST_SMILES)
    
    def test_compute_molecular_descriptors(self):
        """Test descriptor computation."""
        descriptors = compute_molecular_descriptors(TEST_SMILES)
        assert len(descriptors) == len(TEST_SMILES)
        assert "MolWt" in descriptors.columns
        assert "LogP" in descriptors.columns
    
    def test_validate_molecules(self):
        """Test molecule validation."""
        valid = validate_molecules(TEST_SMILES)
        assert all(valid)
        
        all_smiles = TEST_SMILES + INVALID_SMILES
        all_valid = validate_molecules(all_smiles)
        assert sum(all_valid) == len(TEST_SMILES)
    
    def test_filter_valid_molecules(self):
        """Test filtering valid molecules."""
        all_smiles = TEST_SMILES + INVALID_SMILES
        filtered = filter_valid_molecules(all_smiles)
        assert len(filtered) == len(TEST_SMILES)
    
    def test_calculate_similarity(self):
        """Test similarity calculation."""
        for i in range(len(TEST_SMILES)):
            for j in range(len(TEST_SMILES)):
                sim = calculate_similarity(TEST_SMILES[i], TEST_SMILES[j])
                assert 0.0 <= sim <= 1.0
                if i == j:
                    assert sim == 1.0
    
    def test_calculate_pairwise_similarities(self):
        """Test pairwise similarity calculation."""
        sim_matrix = calculate_pairwise_similarities(TEST_SMILES)
        assert isinstance(sim_matrix, np.ndarray)
        assert sim_matrix.shape == (len(TEST_SMILES), len(TEST_SMILES))
        
        # Check diagonal (self-similarity)
        for i in range(len(TEST_SMILES)):
            assert sim_matrix[i, i] == 1.0
