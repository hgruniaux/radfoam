# Error-Based Ray Sampling for RadFoam

## Overview

This document proposes several methods for implementing error-based ray sampling in RadFoam to improve training efficiency by focusing computational resources on areas with higher rendering error.

## Current Approach

RadFoam currently uses uniform random sampling of rays from the training dataset. While this ensures unbiased coverage, it may waste computation on well-fitted regions while under-sampling difficult areas.

## Proposed Methods

### 1. Error-Weighted Sampling (Recommended)

**Core Idea**: Maintain an error map tracking rendering losses for each ray, then sample rays with probability proportional to their recent error.

**Implementation**:
- Track per-ray errors using exponential moving average
- Sample rays using weighted multinomial distribution
- Include warmup period with uniform sampling
- Prevent zero-probability regions with minimum weight floor

**Advantages**:
- ✅ Automatically focuses on hard-to-render regions
- ✅ Continuous adaptation throughout training
- ✅ Simple to implement and understand
- ✅ Minimal memory overhead
- ✅ Compatible with existing data pipeline

**Disadvantages**:
- ❌ May create sampling bias in final stages
- ❌ Could neglect already-learned regions too much
- ❌ Requires hyperparameter tuning (decay, warmup)

**Key Parameters**:
- `decay`: Controls how quickly old errors are forgotten (default: 0.9)
- `warmup_iterations`: Uniform sampling period (default: 1000)
- `min_error_weight`: Prevents zero probability regions (default: 0.1)
- `update_frequency`: How often to update error statistics (default: 100)

### 2. Spatial Error Sampling

**Core Idea**: Maintain error maps based on image coordinates to preserve spatial coherence and sample neighboring pixels with similar error characteristics.

**Implementation**:
- Error maps organized by image and pixel coordinates
- Optional spatial smoothing or patch-based aggregation
- Spatial correlation in sampling to maintain coherence

**Advantages**:
- ✅ Preserves spatial relationships
- ✅ Good for scenes with spatially coherent error patterns
- ✅ Can use patch-based sampling for efficiency
- ✅ Natural for image-based neural rendering

**Disadvantages**:
- ❌ Higher memory requirements (per-pixel storage)
- ❌ More complex coordinate mapping
- ❌ May not work well with irregular error patterns
- ❌ Requires knowledge of image structure

### 3. Adaptive Mixed Sampling

**Core Idea**: Combine multiple sampling strategies and adaptively balance between them based on training progress.

**Implementation**:
- Mix error-weighted, top-k, and uniform sampling
- Adaptive weighting based on training phase
- Strategy switching based on convergence metrics

**Advantages**:
- ✅ Robust to different scene types
- ✅ Can adapt strategy during training
- ✅ Combines benefits of multiple approaches
- ✅ Reduces overfitting to single sampling bias

**Disadvantages**:
- ❌ More complex implementation
- ❌ Additional hyperparameters
- ❌ May be overkill for simple scenes
- ❌ Harder to debug and analyze

### 4. Top-K Error Sampling

**Core Idea**: Sample exclusively from rays with the highest errors.

**Implementation**:
- Maintain sorted error ranking
- Sample from top percentile of errors
- Periodic recomputation of rankings

**Advantages**:
- ✅ Guaranteed focus on hardest regions
- ✅ Simple conceptually
- ✅ Fast sampling from pre-sorted indices

**Disadvantages**:
- ❌ May neglect already-learned regions entirely
- ❌ Can lead to severe overfitting
- ❌ Poor exploration of new regions
- ❌ Sorting overhead

## Implementation Strategy

### Integration Points

1. **Data Handler Enhancement**: Extend `DataHandler` to support error-aware sampling
2. **Error Tracking**: Add error map updates in training loop  
3. **Ray Index Management**: Track which rays correspond to which errors
4. **Batch Construction**: Modify batch fetching to use error-based indices

### Memory Considerations

- **Error-Weighted**: O(N) memory for N rays
- **Spatial**: O(N × H × W) for N images of size H×W  
- **Hybrid approaches**: Moderate overhead with better performance

### Performance Impact

- **Sampling overhead**: ~1-5ms per iteration (negligible)
- **Memory access**: Mostly sequential, cache-friendly
- **Error updates**: Minimal computational cost
- **Overall**: Expected net speedup from better convergence

## Experimental Validation

### Metrics to Track

1. **Convergence Speed**: Iterations to reach target PSNR
2. **Sample Efficiency**: PSNR improvement per ray processed
3. **Error Distribution**: How error varies across scene regions
4. **Sampling Statistics**: Which regions get sampled most/least

### Comparison Baselines

1. **Uniform Random**: Current RadFoam approach
2. **Stratified**: Ensure coverage of all image regions  
3. **Importance Sampling**: Based on image gradients or edge detection
4. **Progressive**: Start uniform, gradually increase error bias

### Expected Results

- **10-30% faster convergence** in complex scenes
- **Better handling of difficult regions** (e.g., fine details, occlusions)
- **Improved final quality** through focused attention
- **More stable training dynamics** with adaptive sampling

## Alternative Approaches

### 1. Curriculum Learning
- Start with easy rays (low variance regions)
- Gradually introduce harder rays
- Schedule based on training progress

### 2. Multi-Scale Error Maps
- Maintain error maps at multiple resolutions
- Coarse-to-fine sampling strategy
- Balance between detail and overview

### 3. Temporal Error Tracking
- For video sequences, track temporal consistency errors
- Sample based on motion and occlusion patterns
- Particularly useful for dynamic scenes

### 4. Semantic-Aware Sampling
- Use semantic segmentation to bias sampling
- Focus on object boundaries and important features
- Combine with perceptual loss functions

## Caveats and Limitations

### Potential Issues

1. **Sampling Bias**: May create distribution shift in final model
2. **Hyperparameter Sensitivity**: Requires careful tuning
3. **Scene Dependence**: Optimal strategy varies by scene type
4. **Memory Usage**: Additional storage for error maps
5. **Initialization Dependence**: Poor initial estimates affect early training

### Mitigation Strategies

1. **Mixed Sampling**: Always include some uniform samples
2. **Adaptive Parameters**: Adjust based on training progress
3. **Multiple Strategies**: Use ensemble of sampling methods
4. **Validation Monitoring**: Track performance on uniform test set
5. **Gradual Transition**: Slowly increase error-bias over time

## Recommended Implementation Plan

### Phase 1: Basic Error-Weighted Sampling
- Implement `ErrorBasedRaySampler` class
- Add error tracking to training loop
- Validate on simple scenes

### Phase 2: Integration and Optimization  
- Integrate with existing data pipeline
- Add proper error aggregation and smoothing
- Performance optimization and memory reduction

### Phase 3: Advanced Features
- Implement spatial sampling
- Add adaptive/mixed strategies
- Comprehensive evaluation and comparison

### Phase 4: Production Ready
- Hyperparameter optimization
- Documentation and examples
- Integration into main codebase

## Usage Example

```python
# Create error-aware data handler
error_config = {
    'strategy': 'error_weighted',
    'args': {
        'decay': 0.9,
        'warmup_iterations': 1000,
        'min_error_weight': 0.1
    }
}

train_data_handler = create_error_aware_data_handler(
    dataset_args=dataset_args,
    rays_per_batch=1_000_000,
    device=device,
    error_sampling_config=error_config
)

# In training loop
for i in range(iterations):
    # Get batch with error-aware sampling
    data_iterator = train_data_handler.get_iter_with_error_sampling(iteration=i)
    ray_batch, rgb_batch, alpha_batch, _, _, _, ray_indices = next(data_iterator)
    
    # Forward pass and loss calculation
    rgba_output, depth, _, _, _ = model(ray_batch, depth_quantiles)
    color_loss = loss_fn(rgb_batch, rgba_output[..., :3])
    
    # Update error map
    per_ray_loss = color_loss.mean(dim=-1)  # Per-ray error
    train_data_handler.update_error_map(per_ray_loss)
    
    # Continue with standard training...
```

## Conclusion

Error-based ray sampling offers a promising approach to improve RadFoam training efficiency. The **error-weighted sampling** method provides the best balance of simplicity, effectiveness, and robustness. Implementation should be incremental, starting with basic error tracking and gradually adding more sophisticated features based on empirical results.

The key insight is that not all rays are equally important for learning—focusing computational resources on difficult regions should lead to faster convergence and better final quality.
