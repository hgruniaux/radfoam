"""
Error-based ray sampling for adaptive training in RadFoam.

This module implements several strategies for biasing ray sampling towards
areas with higher rendering error to improve training efficiency.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
import warnings


class ErrorBasedRaySampler:
    """
    Implements error-based ray sampling for adaptive training.
    
    This sampler maintains an error map that tracks rendering losses for different
    rays and uses this information to bias sampling towards high-error regions.
    """
    
    def __init__(
        self, 
        total_rays: int, 
        device: str, 
        update_frequency: int = 100,
        decay: float = 0.9, 
        warmup_iterations: int = 1000,
        min_error_weight: float = 0.1
    ):
        """
        Initialize the error-based ray sampler.
        
        Args:
            total_rays: Total number of rays in the training dataset
            device: PyTorch device
            update_frequency: How often to update the error map
            decay: Exponential moving average decay factor for error tracking
            warmup_iterations: Number of iterations to use uniform sampling
            min_error_weight: Minimum weight to prevent zero probability regions
        """
        self.total_rays = total_rays
        self.device = device
        self.update_frequency = update_frequency
        self.decay = decay
        self.warmup_iterations = warmup_iterations
        self.min_error_weight = min_error_weight
        
        # Error map for each ray in the training set
        self.error_map = torch.ones(total_rays, device=device)
        self.iteration_count = 0
        self.last_sampled_indices = None
        
        # Statistics for monitoring
        self.error_stats = {
            'mean_error': [],
            'std_error': [],
            'max_error': [],
            'min_error': []
        }
        
    def update_error_map(self, ray_indices: torch.Tensor, losses: torch.Tensor):
        """
        Update error map with recent losses using vectorized operations.
        
        Args:
            ray_indices: Indices of the rays that were sampled
            losses: Per-ray losses (should match length of ray_indices)
        """
        if ray_indices is None or len(losses) == 0:
            return
            
        # Ensure everything is on the same device and detached
        if hasattr(losses, 'detach'):
            losses = losses.detach()
        if not isinstance(losses, torch.Tensor):
            losses = torch.tensor(losses, device=self.device)
        losses = losses.to(self.device)
        ray_indices = ray_indices.to(self.device)
        
        # Handle case where losses might be aggregated
        if len(losses) == 1 and len(ray_indices) > 1:
            losses = losses.repeat(len(ray_indices))
        
        # Ensure we don't exceed available indices
        valid_mask = ray_indices < len(self.error_map)
        if not valid_mask.all():
            ray_indices = ray_indices[valid_mask]
            losses = losses[valid_mask]
        
        # Vectorized update using exponential moving average - NO PYTHON LOOPS!
        if len(ray_indices) > 0:
            current_errors = self.error_map[ray_indices]
            # All operations stay on GPU
            self.error_map[ray_indices] = (
                self.decay * current_errors + (1 - self.decay) * losses[:len(ray_indices)]
            )
        
        self.iteration_count += 1
        
        # Update statistics less frequently to avoid overhead
        if self.iteration_count % self.update_frequency == 0:
            self._update_error_statistics()
        
    def sample_ray_indices(self, num_rays: int, iteration: int) -> torch.Tensor:
        """
        Sample ray indices based on error weights or uniformly during warmup.
        
        Args:
            num_rays: Number of rays to sample
            iteration: Current training iteration
            
        Returns:
            Tensor of sampled ray indices
        """
        if iteration < self.warmup_iterations:
            # Uniform sampling during warmup
            indices = torch.randperm(self.total_rays, device=self.device)[:num_rays]
        else:
            # Error-based sampling
            weights = self.error_map + self.min_error_weight
            # Normalize weights
            weights = weights / weights.sum()
            
            try:
                indices = torch.multinomial(weights, num_rays, replacement=True)
            except RuntimeError as e:
                warnings.warn(f"Error in multinomial sampling: {e}. Falling back to uniform sampling.")
                indices = torch.randperm(self.total_rays, device=self.device)[:num_rays]
        
        self.last_sampled_indices = indices
        return indices
    
    def _update_error_statistics(self):
        """Update error statistics for monitoring."""
        errors = self.error_map.cpu().numpy()
        self.error_stats['mean_error'].append(np.mean(errors))
        self.error_stats['std_error'].append(np.std(errors))
        self.error_stats['max_error'].append(np.max(errors))
        self.error_stats['min_error'].append(np.min(errors))
    
    def get_error_statistics(self) -> dict:
        """Get current error statistics."""
        return self.error_stats.copy()
    
    def should_use_error_sampling(self, iteration: int) -> bool:
        """Check if we should use error-based sampling."""
        return iteration >= self.warmup_iterations
    
    def get_sampling_info(self) -> dict:
        """Get information about current sampling state."""
        if self.error_map is None:
            return {}
            
        errors = self.error_map.cpu().numpy()
        return {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(errors)),
            'min_error': float(np.min(errors)),
            'error_range': float(np.max(errors) - np.min(errors)),
            'iteration_count': self.iteration_count
        }


class SpatialErrorSampler:
    """
    Spatial error sampler that maintains error maps based on image coordinates.
    
    This sampler is useful when you want to maintain spatial coherence and 
    sample neighboring pixels with similar error characteristics.
    """
    
    def __init__(
        self, 
        num_images: int,
        img_height: int,
        img_width: int,
        device: torch.device,
        patch_size: int = 8,
        decay: float = 0.9,
        warmup_iterations: int = 1000
    ):
        """
        Initialize spatial error sampler.
        
        Args:
            num_images: Number of training images
            img_height: Image height
            img_width: Image width  
            device: PyTorch device
            patch_size: Size of spatial patches for error aggregation
            decay: Exponential moving average decay factor
            warmup_iterations: Number of iterations for uniform sampling
        """
        self.num_images = num_images
        self.img_height = img_height
        self.img_width = img_width
        self.device = device
        self.patch_size = patch_size
        self.decay = decay
        self.warmup_iterations = warmup_iterations
        
        # Error map per pixel across all training images
        self.error_map = torch.ones(num_images, img_height, img_width, device=device)
        self.iteration_count = 0
    
    def pixel_indices_to_coords(self, ray_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert flat ray indices to image coordinates."""
        pixels_per_image = self.img_height * self.img_width
        img_indices = ray_indices // pixels_per_image
        pixel_indices = ray_indices % pixels_per_image
        h_coords = pixel_indices // self.img_width
        w_coords = pixel_indices % self.img_width
        return img_indices, h_coords, w_coords
    
    def update_error_map(self, ray_indices: torch.Tensor, losses: torch.Tensor):
        """Update spatial error map."""
        img_indices, h_coords, w_coords = self.pixel_indices_to_coords(ray_indices)
        
        if len(losses) == 1 and len(ray_indices) > 1:
            losses = losses.repeat(len(ray_indices))
        
        for i in range(len(ray_indices)):
            if i < len(losses):
                img_idx, h, w = img_indices[i], h_coords[i], w_coords[i]
                current_error = self.error_map[img_idx, h, w]
                new_error = losses[i].item() if hasattr(losses[i], 'item') else float(losses[i])
                self.error_map[img_idx, h, w] = (
                    self.decay * current_error + (1 - self.decay) * new_error
                )
        
        self.iteration_count += 1
    
    def sample_ray_indices(self, num_rays: int, iteration: int) -> torch.Tensor:
        """Sample rays based on spatial error distribution."""
        if iteration < self.warmup_iterations:
            # Uniform sampling
            total_rays = self.num_images * self.img_height * self.img_width
            return torch.randperm(total_rays, device=self.device)[:num_rays]
        else:
            # Spatial error-based sampling
            weights = self.error_map.flatten() + 1e-6
            weights = weights / weights.sum()
            
            try:
                indices = torch.multinomial(weights, num_rays, replacement=True)
                return indices
            except RuntimeError:
                warnings.warn("Error in spatial sampling, falling back to uniform")
                total_rays = self.num_images * self.img_height * self.img_width
                return torch.randperm(total_rays, device=self.device)[:num_rays]
    
    def get_sampling_info(self) -> dict:
        """Get information about current sampling state."""
        if self.error_map is None:
            return {}
            
        errors = self.error_map.cpu().numpy()
        return {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(errors)),
            'min_error': float(np.min(errors)),
            'error_range': float(np.max(errors) - np.min(errors)),
            'iteration_count': self.iteration_count
        }


class AdaptiveErrorSampler:
    """
    Adaptive sampler that combines multiple error-based sampling strategies.
    """
    
    def __init__(
        self,
        total_rays: int,
        device: torch.device,
        strategies: List[str] = ['error_weighted', 'top_k', 'mixed']
    ):
        self.total_rays = total_rays
        self.device = device
        self.strategies = strategies
        self.current_strategy = 0
        
        # Different samplers for different strategies
        self.error_sampler = ErrorBasedRaySampler(total_rays, device)
        
    def update_error_map(self, ray_indices: torch.Tensor, losses: torch.Tensor):
        """Update error maps for all strategies."""
        self.error_sampler.update_error_map(ray_indices, losses)
    
    def sample_ray_indices(self, num_rays: int, iteration: int) -> torch.Tensor:
        """Sample using adaptive strategy selection."""
        strategy = self.strategies[self.current_strategy % len(self.strategies)]
        
        if strategy == 'error_weighted':
            return self.error_sampler.sample_ray_indices(num_rays, iteration)
        elif strategy == 'top_k':
            return self._sample_top_k_errors(num_rays, iteration)
        elif strategy == 'mixed':
            return self._sample_mixed(num_rays, iteration)
        else:
            return self.error_sampler.sample_ray_indices(num_rays, iteration)
    
    def _sample_top_k_errors(self, num_rays: int, iteration: int) -> torch.Tensor:
        """Sample from top-k highest error rays."""
        if iteration < self.error_sampler.warmup_iterations:
            return torch.randperm(self.total_rays, device=self.device)[:num_rays]
        
        # Get top-k error indices
        k = min(num_rays * 4, self.total_rays)  # 4x oversampling
        _, top_indices = torch.topk(self.error_sampler.error_map, k)
        
        # Randomly sample from top-k
        perm = torch.randperm(k, device=self.device)[:num_rays]
        return top_indices[perm]
    
    def _sample_mixed(self, num_rays: int, iteration: int) -> torch.Tensor:
        """Mix uniform and error-based sampling."""
        if iteration < self.error_sampler.warmup_iterations:
            return torch.randperm(self.total_rays, device=self.device)[:num_rays]
        
        # 70% error-based, 30% uniform
        error_rays = int(num_rays * 0.7)
        uniform_rays = num_rays - error_rays
        
        error_indices = self.error_sampler.sample_ray_indices(error_rays, iteration)
        uniform_indices = torch.randperm(self.total_rays, device=self.device)[:uniform_rays]
        
        # Combine and shuffle
        all_indices = torch.cat([error_indices, uniform_indices])
        perm = torch.randperm(len(all_indices), device=self.device)
        return all_indices[perm]
    
    def get_sampling_info(self) -> dict:
        """Get information about current sampling state."""
        return self.error_sampler.get_sampling_info()
