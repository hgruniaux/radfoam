# Error Sampling Control Guide

## How to Disable Error Sampling

There are several ways to disable error sampling and fallback to baseline uniform sampling:

### Method 1: Command Line Arguments (Recommended)

```bash
# Disable error sampling completely
python train.py --error_sampling none

# Use error-weighted sampling (default)
python train.py --error_sampling error_weighted

# Use spatial error sampling
python train.py --error_sampling spatial

# Use adaptive error sampling
python train.py --error_sampling adaptive
```

### Method 2: Custom Parameters

You can also customize the error sampling parameters:

```bash
# Use error sampling with custom parameters
python train.py \
    --error_sampling error_weighted \
    --error_sampling_decay 0.95 \
    --error_sampling_warmup 500 \
    --error_sampling_update_freq 50
```

### Method 3: Configuration File

Add to your YAML config file:
```yaml
error_sampling: "none"  # or "error_weighted", "spatial", "adaptive"
error_sampling_decay: 0.9
error_sampling_warmup: 1000
error_sampling_update_freq: 100
```

### Method 4: Code Modification

In `train.py`, change:
```python
error_sampling_config = {
    'strategy': None,  # None = disabled, 'error_weighted' = enabled
    'args': { ... }
}
```

## Command Line Arguments Reference

| Argument | Default | Options | Description |
|----------|---------|---------|-------------|
| `--error_sampling` | `error_weighted` | `error_weighted`, `spatial`, `adaptive`, `none` | Sampling strategy |
| `--error_sampling_decay` | `0.9` | `0.0-1.0` | Error memory decay factor |
| `--error_sampling_warmup` | `1000` | `> 0` | Uniform sampling iterations |
| `--error_sampling_update_freq` | `100` | `> 0` | Error statistics update frequency |

## Performance Comparison Commands

To compare performance between uniform and error-based sampling:

```bash
# Baseline (uniform sampling)
python train.py --error_sampling none --experiment_name "baseline_uniform"

# Error-based sampling
python train.py --error_sampling error_weighted --experiment_name "error_weighted"

# Quick error sampling (less warmup)
python train.py \
    --error_sampling error_weighted \
    --error_sampling_warmup 100 \
    --experiment_name "error_weighted_fast"
```

## Output Messages

The training will print which method is being used:

- **Error sampling enabled**: `"Using error-based ray sampling: error_weighted"`
- **Error sampling disabled**: `"Using uniform ray sampling (error sampling disabled)"`

## Troubleshooting

If you encounter issues:

1. **Disable error sampling**: Add `--error_sampling none`
2. **Check arguments**: Use `python train.py --help` to see all options
3. **Verify config**: Check your YAML config file doesn't override settings
4. **Debug mode**: Add `--debug` to disable logging overhead

## Example Usage

```bash
# Production training with error sampling
python train.py \
    --config configs/mipnerf360_outdoor.yaml \
    --error_sampling error_weighted \
    --experiment_name "scene_with_error_sampling"

# Baseline comparison without error sampling  
python train.py \
    --config configs/mipnerf360_outdoor.yaml \
    --error_sampling none \
    --experiment_name "scene_baseline"

# Fast error sampling for testing
python train.py \
    --error_sampling error_weighted \
    --error_sampling_warmup 10 \
    --error_sampling_update_freq 10 \
    --iterations 1000 \
    --experiment_name "quick_test"
```
