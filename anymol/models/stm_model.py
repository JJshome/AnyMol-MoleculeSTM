"""
MoleculeSTM model integration with AnyMol framework.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import math
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer architecture.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MoleculeEncoder(nn.Module):
    """
    Encoder for molecular representations based on transformer architecture.
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 256,
        vocabulary_size: int = 1000,
    ):
        super(MoleculeEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.vocabulary_size = vocabulary_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # Final projection layer
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src shape: [seq_len, batch_size]
        
        # Embedding and positional encoding
        src = self.token_embedding(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        
        # Transformer encoder
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        
        # Global pooling (mean over sequence length)
        output = output.mean(dim=0)
        
        # Final projection
        output = self.output_projection(output)
        
        return output

class MoleculeDecoder(nn.Module):
    """
    Decoder for generating molecules from latent representations.
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 256,
        vocabulary_size: int = 1000,
    ):
        super(MoleculeDecoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.vocabulary_size = vocabulary_size
        
        # Initial linear projection for latent representation
        self.latent_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        # GRU for sequential decoding
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_dim, vocabulary_size)
        
    def forward(self, latent, target=None, teacher_forcing_ratio=0.5):
        # latent shape: [batch_size, embedding_dim]
        # target shape (if provided): [batch_size, seq_len]
        
        batch_size = latent.size(0)
        
        # Project latent vector
        latent = self.latent_projection(latent).unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # Initialize decoder input with start token
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=latent.device)
        decoder_input[:, 0] = 1  # Assuming 1 is the start token
        
        # Initialize hidden state with latent representation
        hidden = latent.permute(1, 0, 2)  # [1, batch_size, embedding_dim]
        
        # Prepare outputs tensor
        max_length = self.max_seq_length if target is None else target.size(1)
        outputs = torch.zeros(batch_size, max_length, self.vocabulary_size, device=latent.device)
        
        # Decode sequence token by token
        for t in range(max_length):
            # Embed current input token
            embedded = self.token_embedding(decoder_input).permute(1, 0, 2)  # [1, batch_size, embedding_dim]
            
            # GRU step
            output, hidden = self.gru(embedded, hidden)
            
            # Predict next token
            prediction = self.output_projection(output.squeeze(0))
            outputs[:, t] = prediction
            
            # Teacher forcing: use actual next token as input
            if target is not None and t < max_length - 1:
                # Apply teacher forcing with probability
                teacher_force = torch.rand(1) < teacher_forcing_ratio
                decoder_input = target[:, t+1].unsqueeze(1) if teacher_force else prediction.argmax(dim=1).unsqueeze(1)
            else:
                decoder_input = prediction.argmax(dim=1).unsqueeze(1)
        
        return outputs

class PropertyPredictor(nn.Module):
    """
    Neural network for predicting molecular properties from latent representation.
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dims: List[int] = [512, 256, 128],
        output_dim: int = 1,
        dropout: float = 0.1,
        activation: str = 'relu',
    ):
        super(PropertyPredictor, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Build MLP layers
        layers = []
        input_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(input_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # x shape: [batch_size, embedding_dim]
        return self.network(x)

class MoleculeSTM(nn.Module):
    """
    Main MoleculeSTM model combining encoder, decoder, and property prediction.
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 256,
        vocabulary_size: int = 1000,
        property_hidden_dims: List[int] = [512, 256, 128],
        property_names: List[str] = ["logP", "TPSA", "MolWt", "QED"],
    ):
        super(MoleculeSTM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.property_names = property_names
        
        # Encoder
        self.encoder = MoleculeEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_length=max_seq_length,
            vocabulary_size=vocabulary_size,
        )
        
        # Decoder
        self.decoder = MoleculeDecoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_length=max_seq_length,
            vocabulary_size=vocabulary_size,
        )
        
        # Property predictors (one for each property)
        self.property_predictors = nn.ModuleDict()
        for prop in property_names:
            self.property_predictors[prop] = PropertyPredictor(
                embedding_dim=embedding_dim,
                hidden_dims=property_hidden_dims,
                output_dim=1,
                dropout=dropout,
            )
        
        # Contrastive learning projection head
        self.contrastive_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def encode(self, x, mask=None, padding_mask=None):
        """Encode input sequence to latent representation."""
        return self.encoder(x, mask, padding_mask)
    
    def decode(self, latent, target=None, teacher_forcing_ratio=0.5):
        """Decode latent representation to sequence."""
        return self.decoder(latent, target, teacher_forcing_ratio)
    
    def predict_properties(self, latent):
        """Predict properties from latent representation."""
        predictions = {}
        for prop, predictor in self.property_predictors.items():
            predictions[prop] = predictor(latent)
        return predictions
    
    def get_contrastive_embedding(self, latent):
        """Get normalized embedding for contrastive learning."""
        embedding = self.contrastive_projection(latent)
        return F.normalize(embedding, p=2, dim=1)
    
    def forward(self, x, mask=None, padding_mask=None, target=None, teacher_forcing_ratio=0.5):
        """Full forward pass: encode, decode, and predict properties."""
        # Encode
        latent = self.encode(x, mask, padding_mask)
        
        # Decode (reconstruction)
        decoded = self.decode(latent, target, teacher_forcing_ratio)
        
        # Property prediction
        properties = self.predict_properties(latent)
        
        # Contrastive embedding
        contrastive_embedding = self.get_contrastive_embedding(latent)
        
        return {
            "latent": latent,
            "decoded": decoded,
            "properties": properties,
            "contrastive_embedding": contrastive_embedding
        }

class AnyMolSTMModel(nn.Module):
    """
    Integration of AnyMol methodology with MoleculeSTM for enhanced molecular design.
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 256,
        vocabulary_size: int = 1000,
        property_hidden_dims: List[int] = [512, 256, 128],
        property_names: List[str] = ["logP", "TPSA", "MolWt", "QED"],
        anymol_integration_method: str = "concatenate",
    ):
        super(AnyMolSTMModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.property_names = property_names
        self.anymol_integration_method = anymol_integration_method
        
        # Base MoleculeSTM model
        self.molecule_stm = MoleculeSTM(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_length=max_seq_length,
            vocabulary_size=vocabulary_size,
            property_hidden_dims=property_hidden_dims,
            property_names=property_names,
        )
        
        # AnyMol integration components
        if anymol_integration_method == "concatenate":
            # Concatenate MoleculeSTM embedding with additional features
            self.integration_projection = nn.Linear(embedding_dim * 2, embedding_dim)
        elif anymol_integration_method == "attention":
            # Multi-head attention for integrating MoleculeSTM with AnyMol
            self.integration_attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.layer_norm1 = nn.LayerNorm(embedding_dim)
            self.layer_norm2 = nn.LayerNorm(embedding_dim)
            self.integration_mlp = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim)
            )
        elif anymol_integration_method == "gating":
            # Gating mechanism for adaptive integration
            self.gate_network = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embedding_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown integration method: {anymol_integration_method}")
        
        # Enhanced property predictors with integrated embeddings
        self.enhanced_property_predictors = nn.ModuleDict()
        for prop in property_names:
            self.enhanced_property_predictors[prop] = PropertyPredictor(
                embedding_dim=embedding_dim,
                hidden_dims=property_hidden_dims,
                output_dim=1,
                dropout=dropout,
            )
        
        # Molecular optimization module
        self.optimization_network = nn.Sequential(
            nn.Linear(embedding_dim + len(property_names), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
    def integrate_embeddings(self, stm_embedding, anymol_embedding):
        """Integrate MoleculeSTM and AnyMol embeddings based on the selected method."""
        if self.anymol_integration_method == "concatenate":
            # Concatenate and project
            combined = torch.cat([stm_embedding, anymol_embedding], dim=1)
            return self.integration_projection(combined)
            
        elif self.anymol_integration_method == "attention":
            # Multi-head attention integration
            stm_embedding_norm = self.layer_norm1(stm_embedding)
            anymol_embedding_norm = self.layer_norm1(anymol_embedding)
            
            # Self-attention with STM as query and AnyMol as key/value
            attn_output, _ = self.integration_attention(
                query=stm_embedding_norm.unsqueeze(1),
                key=anymol_embedding_norm.unsqueeze(1),
                value=anymol_embedding_norm.unsqueeze(1)
            )
            attn_output = attn_output.squeeze(1)
            
            # Add & norm
            add1 = stm_embedding + attn_output
            add1_norm = self.layer_norm2(add1)
            
            # MLP
            mlp_output = self.integration_mlp(add1_norm)
            
            # Final add
            return add1 + mlp_output
            
        elif self.anymol_integration_method == "gating":
            # Gating mechanism
            combined = torch.cat([stm_embedding, anymol_embedding], dim=1)
            gate = self.gate_network(combined)
            return gate * stm_embedding + (1 - gate) * anymol_embedding
    
    def forward(
        self,
        x,
        anymol_features,
        mask=None,
        padding_mask=None,
        target=None,
        target_properties=None,
        teacher_forcing_ratio=0.5,
    ):
        """
        Forward pass through the integrated AnyMol-MoleculeSTM model.
        
        Args:
            x: Input token sequence
            anymol_features: Features from AnyMol
            mask: Attention mask
            padding_mask: Padding mask
            target: Target sequence for teacher forcing
            target_properties: Target property values for optimization
            teacher_forcing_ratio: Ratio for teacher forcing
            
        Returns:
            Dictionary of outputs
        """
        # Get MoleculeSTM embeddings and outputs
        stm_outputs = self.molecule_stm(x, mask, padding_mask, target, teacher_forcing_ratio)
        stm_embedding = stm_outputs["latent"]
        
        # Integrate MoleculeSTM and AnyMol embeddings
        integrated_embedding = self.integrate_embeddings(stm_embedding, anymol_features)
        
        # Enhanced property prediction
        enhanced_properties = {}
        for prop, predictor in self.enhanced_property_predictors.items():
            enhanced_properties[prop] = predictor(integrated_embedding)
        
        # If target properties are provided, perform molecular optimization
        optimized_embedding = None
        optimized_decoded = None
        if target_properties is not None:
            # Create property target tensor
            property_targets = torch.zeros(
                integrated_embedding.size(0),
                len(self.property_names),
                device=integrated_embedding.device
            )
            
            for i, prop in enumerate(self.property_names):
                if prop in target_properties:
                    property_targets[:, i] = target_properties[prop]
            
            # Concatenate embedding with property targets
            optimization_input = torch.cat([integrated_embedding, property_targets], dim=1)
            
            # Generate optimized embedding
            optimized_embedding = self.optimization_network(optimization_input)
            
            # Decode optimized molecules
            optimized_decoded = self.molecule_stm.decode(optimized_embedding)
        
        # Return results
        results = {
            "stm_outputs": stm_outputs,
            "integrated_embedding": integrated_embedding,
            "enhanced_properties": enhanced_properties,
        }
        
        if optimized_embedding is not None:
            results["optimized_embedding"] = optimized_embedding
            results["optimized_decoded"] = optimized_decoded
        
        return results
