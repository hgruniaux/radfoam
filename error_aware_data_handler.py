"""
Enhanced data handler with error-based ray sampling support.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from data_loader import DataHandler
from error_sampling import ErrorBasedRaySampler, SpatialErrorSampler, AdaptiveErrorSampler


class ErrorAwareDataHandler(DataHandler):
    """
    Enhanced DataHandler that supports error-based ray sampling.
    """
    
    def __init__(
        self, 
        dataset_args, 
        rays_per_batch, 
        device="cuda",
        update_frequency=100,
        decay=0.9,
        warmup_iterations=1000,
        min_error_weight=0.1,
        debug_constant_error=False,
    ):
        super().__init__(dataset_args, rays_per_batch, device)
        
        self.update_frequency = update_frequency
        self.decay = decay
        self.warmup_iterations = warmup_iterations
        self.min_error_weight = min_error_weight
        self.debug_constant_error = debug_constant_error
        self.current_ray_indices = None
        self.error_sampler = None
        
    def reload(self, split, downsample=None):
        """Reload dataset and initialize error sampler."""
        super().reload(split, downsample)
        
        if split == "train":
            # Move training data to the correct device
            self.train_rays = self.train_rays.to(self.device)
            self.train_rgbs = self.train_rgbs.to(self.device)
            self.train_alphas = self.train_alphas.to(self.device)

            if self.train_normals is not None:
                self.train_normals = self.train_normals.to(self.device)
            if self.train_depths is not None:
                self.train_depths = self.train_depths.to(self.device)
            if self.train_instances is not None:
                self.train_instances = self.train_instances.to(self.device)

            # Initialize error sampler based on strategy
            total_rays = len(self.train_rays)
            
            self.error_sampler = ErrorBasedRaySampler(
                total_rays=total_rays,
                device=self.device,
                update_frequency=self.update_frequency,
                decay=self.decay,
                warmup_iterations=self.warmup_iterations,
                min_error_weight=self.min_error_weight
            )
            
            # Initialize debug constant error pattern if enabled
            if self.debug_constant_error:
                self._init_debug_error_pattern()
    
    def get_iter_with_error_sampling(self, iteration=0):
        """
        Args:
            iteration: Current training iteration for warmup control
            
        Yields:
            Tuple of (ray_batch, rgb_batch, alpha_batch, normal_batch, depth_batch, instance_batch, ray_indices)
        """
        while True:
            # Sample ray indices based on error
            ray_indices = self.error_sampler.sample_ray_indices(
                self.batch_size, iteration
            )
            self.current_ray_indices = ray_indices
            
            ray_batch = self.train_rays[ray_indices]
            rgb_batch = self.train_rgbs[ray_indices] 
            alpha_batch = self.train_alphas[ray_indices]
            
            normal_batch = (
                self.train_normals[ray_indices] 
                if self.train_normals is not None else None
            )
            depth_batch = (
                self.train_depths[ray_indices] 
                if self.train_depths is not None else None
            )
            instance_batch = (
                self.train_instances[ray_indices] 
                if self.train_instances is not None else None
            )
            
            yield (
                ray_batch, rgb_batch, alpha_batch, 
                normal_batch, depth_batch, instance_batch,
                ray_indices
            )
    
    def update_error_map(self, losses: torch.Tensor):
        """
        Update error map with recent losses.
        
        Args:
            losses: Per-ray losses from the last batch
        """
        if self.error_sampler is not None and self.current_ray_indices is not None:
            if self.debug_constant_error:
                # Debug mode: Don't update error map, keep it constant
                print(f"[DEBUG] Error map update skipped (constant error mode). "
                      f"Mean loss: {losses.mean().item():.6f}, "
                      f"Error map mean: {self.error_sampler.error_map.mean().item():.6f}")
            else:
                # Normal mode: Update error map with losses
                self.error_sampler.update_error_map(self.current_ray_indices, losses)
    
    def get_error_statistics(self) -> dict:
        """Get error sampling statistics."""
        if self.error_sampler is None:
            return {}
        
        if hasattr(self.error_sampler, 'get_sampling_info'):
            return self.error_sampler.get_sampling_info()
        else:
            return {}

    def _init_debug_error_pattern(self):
        """Initialize a specific error pattern for debugging purposes."""
        if self.error_sampler is not None:
            total_rays = len(self.error_sampler.error_map)
            
            debug_pattern = torch.full((total_rays,), 0.001, device=self.device)
            debug_pattern[::250] = 10.0  # Every 250th ray gets high error
            
            self.error_sampler.error_map = debug_pattern
            print(f"[DEBUG] Initialized constant error pattern: "
                  f"{(debug_pattern == 10.0).sum().item()} high-error rays out of {total_rays}")

    def get_debug_info(self) -> dict:
        """Get debug information about error sampling."""
        if self.error_sampler is None:
            return {}
            
        info = {
            'debug_constant_error': self.debug_constant_error,
            'current_ray_indices_shape': self.current_ray_indices.shape if self.current_ray_indices is not None else None,
            'error_map_shape': self.error_sampler.error_map.shape,
            'error_map_min': self.error_sampler.error_map.min().item(),
            'error_map_max': self.error_sampler.error_map.max().item(),
            'error_map_mean': self.error_sampler.error_map.mean().item(),
        }
        
        if self.debug_constant_error:
            # Count how many high-error rays are being sampled
            if self.current_ray_indices is not None:
                sampled_errors = self.error_sampler.error_map[self.current_ray_indices]
                high_error_sampled = (sampled_errors > 1.0).sum().item()
                info['high_error_rays_sampled'] = high_error_sampled
                info['high_error_sample_ratio'] = high_error_sampled / len(self.current_ray_indices)
        
        return info


class CustomBatchFetcher:
    """
    Custom batch fetcher that can work with pre-computed indices.
    This replaces the radfoam.BatchFetcher with error-aware sampling.
    """
    
    def __init__(self, data, batch_size, shuffle=True, indices=None):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_idx = 0
        
        if indices is None:
            self.indices = torch.arange(len(data))
        else:
            self.indices = indices
            
        if self.shuffle and self.indices is not None:
            self.indices = self.indices[torch.randperm(len(self.indices))]
    
    def next(self):
        """Get next batch."""
        if self.indices is None or self.current_idx >= len(self.indices):
            self.current_idx = 0
            if self.shuffle and self.indices is not None:
                self.indices = self.indices[torch.randperm(len(self.indices))]
        
        if self.indices is not None:
            end_idx = min(self.current_idx + self.batch_size, len(self.indices))
            batch_indices = self.indices[self.current_idx:end_idx]
            self.current_idx = end_idx
            return self.data[batch_indices]
        else:
            # Fallback to regular indexing
            return self.data[:self.batch_size]
    
    def set_custom_indices(self, indices):
        """Set custom indices for sampling."""
        self.indices = indices
        self.current_idx = 0


def create_error_aware_data_handler(
    dataset_args,
    rays_per_batch=1000000,
    device="cuda", 
    update_frequency=100,
    decay=0.9,
    warmup_iterations=1000,
    min_error_weight=0.1,
    debug_constant_error=False
):
    return ErrorAwareDataHandler(
        dataset_args=dataset_args,
        rays_per_batch=rays_per_batch,
        device=device,
        update_frequency=update_frequency,
        decay=decay,
        warmup_iterations=warmup_iterations,
        min_error_weight=min_error_weight,
        debug_constant_error=debug_constant_error
    )
