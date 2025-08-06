import os
import uuid
import yaml
import gc
import numpy as np
from PIL import Image
import configargparse
import tqdm
import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data_loader import DataHandler
from configs import *
from radfoam_model.scene import RadFoamScene
from radfoam_model.utils import psnr
import radfoam

# Import error sampling components
from error_sampling import ErrorBasedRaySampler
from error_aware_data_handler import ErrorAwareDataHandler, create_error_aware_data_handler


seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)

def relu_based_conditional_loss(x, y):
    return torch.where(x >= 0, (x - y) ** 2, torch.zeros_like(x))


def train(args, pipeline_args, model_args, optimizer_args, dataset_args):
    device = torch.device(model_args.device)
    # Setting up output directory
    if not pipeline_args.debug:
        if len(pipeline_args.experiment_name) == 0:
            unique_str = str(uuid.uuid4())[:8]
            experiment_name = f"{dataset_args.scene}@{unique_str}"
        else:
            experiment_name = pipeline_args.experiment_name
        out_dir = f"output/{experiment_name}"
        writer = SummaryWriter(out_dir, purge_step=0)
        os.makedirs(f"{out_dir}/test", exist_ok=True)

        def represent_list_inline(dumper, data):
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )

        yaml.add_representer(list, represent_list_inline)

        # Save the arguments to a YAML file
        with open(f"{out_dir}/config.yaml", "w") as yaml_file:
            yaml.dump(vars(args), yaml_file, default_flow_style=False)

    # Setting up dataset with error-based sampling
    iter2downsample = dict(
        zip(
            dataset_args.downsample_iterations,
            dataset_args.downsample,
        )
    )
    
    # Create error-aware data handler or fallback to standard DataHandler
    if pipeline_args.error_sampling:
        train_data_handler = create_error_aware_data_handler(
            dataset_args=dataset_args,
            rays_per_batch=1_000_000,
            device=str(device),  # Convert to string for compatibility
            decay=pipeline_args.error_sampling_decay,
            warmup_iterations=pipeline_args.error_sampling_warmup,
            update_frequency=pipeline_args.error_sampling_update_freq,
            min_error_weight=pipeline_args.error_sampling_min_weight,
        )
        print(f"Using error-based ray sampling")
    else:
        train_data_handler = DataHandler(
            dataset_args, rays_per_batch=1_000_000, device=str(device)
        )
        print("Using uniform ray sampling (error sampling disabled)")
    
    downsample = iter2downsample[0]
    train_data_handler.reload(split="train", downsample=downsample)

    test_data_handler = DataHandler(
        dataset_args, rays_per_batch=0, device=str(device)
    )
    test_data_handler.reload(
        split="test", downsample=min(dataset_args.downsample)
    )
    
    # Try to use original radfoam.BatchFetcher for test data (faster)
    try:
        test_ray_batch_fetcher = radfoam.BatchFetcher(
            test_data_handler.rays, batch_size=1, shuffle=False
        )
        test_rgb_batch_fetcher = radfoam.BatchFetcher(
            test_data_handler.rgbs, batch_size=1, shuffle=False
        )
    except AttributeError:
        # Fallback if radfoam.BatchFetcher not available
        class SimpleBatchFetcher:
            def __init__(self, data, batch_size=1, shuffle=False):
                self.data = data
                self.batch_size = batch_size
                self.current_idx = 0
            
            def next(self):
                if self.current_idx >= len(self.data):
                    self.current_idx = 0
                
                end_idx = min(self.current_idx + self.batch_size, len(self.data))
                batch = self.data[self.current_idx:end_idx]
                self.current_idx = end_idx
                return [batch]
        
        test_ray_batch_fetcher = SimpleBatchFetcher(
            test_data_handler.rays, batch_size=1, shuffle=False
        )
        test_rgb_batch_fetcher = SimpleBatchFetcher(
            test_data_handler.rgbs, batch_size=1, shuffle=False
        )

    # Define viewer settings
    viewer_options = {
        "camera_pos": train_data_handler.viewer_pos,
        "camera_up": train_data_handler.viewer_up,
        "camera_forward": train_data_handler.viewer_forward,
        "all_positions": train_data_handler.camera_positions,
        "all_forwards": train_data_handler.camera_forwards,
        "all_ups": train_data_handler.camera_ups,
    }

    # Setting up pipeline
    rgb_loss = nn.SmoothL1Loss(reduction="none")

    # Setting up model
    model = RadFoamScene(
        args=model_args,
        device=device,
        points=train_data_handler.points3D,
        points_colors=train_data_handler.points3D_colors,
    )

    # Setting up optimizer
    model.declare_optimizer(
        args=optimizer_args,
        warmup=pipeline_args.densify_from,
        max_iterations=pipeline_args.iterations,
    )

    def test_render(
        test_data_handler, ray_batch_fetcher, rgb_batch_fetcher, suffix="final", debug=False, save_output=False
    ):
        rays = test_data_handler.rays
        points, _, _, _ = model.get_trace_data()
        start_points = model.get_starting_point(
            rays[:, 0, 0].cuda(), points, model.aabb_tree
        )

        psnr_list = []
        with torch.no_grad():
            for i in range(rays.shape[0]):
                ray_batch = ray_batch_fetcher.next()[0]
                rgb_batch = rgb_batch_fetcher.next()[0]
                output, _, _, _, _ = model(ray_batch, start_points[i])

                # White background
                opacity = output[..., -1:]
                rgb_output = output[..., :3] + (1 - opacity)
                rgb_output = rgb_output.reshape(*rgb_batch.shape).clip(0, 1)

                img_psnr = psnr(rgb_output, rgb_batch).mean()
                psnr_list.append(img_psnr)
                torch.cuda.synchronize()

                if not debug and save_output:
                    error = np.uint8((rgb_output - rgb_batch).cpu().abs() * 255)
                    rgb_output = np.uint8(rgb_output.cpu() * 255)
                    rgb_batch = np.uint8(rgb_batch.cpu() * 255)

                    im = Image.fromarray(
                        np.concatenate([rgb_output, rgb_batch, error], axis=1)
                    )
                    im.save(
                        f"{out_dir}/test/rgb_{i:03d}_{suffix}.png"
                    )

        average_psnr = sum(psnr_list) / len(psnr_list)
        if not debug:
            f = open(f"{out_dir}/metrics.txt", "w")
            f.write(f"Average PSNR: {average_psnr}")
            f.close()

        return average_psnr

    def train_loop(viewer):
        print(f"Training (saving to {out_dir})")

        torch.cuda.synchronize()

        if pipeline_args.error_sampling:
            # Error-aware data handler
            data_iterator = train_data_handler.get_iter_with_error_sampling(iteration=0)
            ray_batch, rgb_batch, alpha_batch, normal_batch, depth_batch, instance_batch, ray_indices = next(data_iterator)
        else:
            # Standard data handler
            data_iterator = train_data_handler.get_iter()
            ray_batch, rgb_batch, alpha_batch, normal_batch, depth_batch, instance_batch = next(data_iterator)

        triangulation_update_period = 1
        iters_since_update = 1
        iters_since_densification = 0
        next_densification_after = 1

        loss_over_time = []
        color_loss_over_time = []
        opacity_loss_over_time = []
        depth_loss_over_time = []
        quant_loss_over_time = []
        w_depth_over_time = []
        num_points_over_time = []
        test_psnr_over_time = []
        position_grad_over_time = []
        att_dc_grad_over_time = []
        att_sh_grad_over_time = []
        density_grad_over_time = []

        # Track error sampling statistics
        error_stats_history = []

        with tqdm.trange(pipeline_args.iterations) as train:
            for i in train:
                if viewer is not None:
                    model.update_viewer(viewer)
                    viewer.step(i)

                if i in iter2downsample and i:
                    downsample = iter2downsample[i]
                    train_data_handler.reload(
                        split="train", downsample=downsample
                    )

                    if pipeline_args.error_sampling:
                        data_iterator = train_data_handler.get_iter_with_error_sampling(iteration=i)
                        ray_batch, rgb_batch, alpha_batch, normal_batch, depth_batch, instance_batch, ray_indices = next(data_iterator)
                    else:
                        data_iterator = train_data_handler.get_iter()
                        ray_batch, rgb_batch, alpha_batch, normal_batch, depth_batch, instance_batch = next(data_iterator)

                depth_quantiles = (
                    torch.rand(*ray_batch.shape[:-1], 2, device=device)
                    .sort(dim=-1, descending=True)
                    .values
                )

                rgba_output, depth, _, _, _ = model(
                    ray_batch,
                    depth_quantiles=depth_quantiles,
                )

                # White background
                opacity = rgba_output[..., -1:]
                if pipeline_args.white_background:
                    rgb_output = rgba_output[..., :3] + (1 - opacity)
                else:
                    rgb_output = rgba_output[..., :3]

                color_loss = rgb_loss(rgb_batch, rgb_output)
                opacity_loss = ((alpha_batch - opacity) ** 2).mean()

                valid_depth_mask = (depth > 0).all(dim=-1)
                quant_loss = (depth[..., 0] - depth[..., 1]).abs()
                quant_loss = (quant_loss * valid_depth_mask).mean()
                w_depth = pipeline_args.quantile_weight * min(
                    2 * i / pipeline_args.iterations, 1
                )

                depth_loss_value = relu_based_conditional_loss(
                    depth[..., 0],
                    pipeline_args.depth_scale * depth_batch
                ).mean() if pipeline_args.depth_loss else torch.tensor(0.0, device=device)

                loss = color_loss.mean() + opacity_loss + w_depth * quant_loss + pipeline_args.depth_coeff * depth_loss_value

                # Update error map for error-based sampling
                if pipeline_args.error_sampling:
                    # Calculate per-ray color loss for error map update
                    per_ray_color_loss = color_loss.mean(dim=-1) if len(color_loss.shape) > 1 else color_loss
                    train_data_handler.update_error_map(per_ray_color_loss)
                    
                    # Log error statistics periodically
                    if i % 100 == 0:
                        error_stats = train_data_handler.get_error_statistics()
                        error_stats_history.append(error_stats)
                        if not pipeline_args.debug:
                            writer.add_scalar("error_sampling/mean_error", 
                                            error_stats.get('mean_error', 0), i)
                            writer.add_scalar("error_sampling/error_range", 
                                            error_stats.get('error_range', 0), i)

                model.optimizer.zero_grad(set_to_none=True)

                # Hide latency of data loading behind the backward pass
                event = torch.cuda.Event()
                event.record()
                loss.backward()
                event.synchronize()
                
                if pipeline_args.error_sampling:
                    ray_batch, rgb_batch, alpha_batch, normal_batch, depth_batch, instance_batch, ray_indices = next(data_iterator)
                else:
                    ray_batch, rgb_batch, alpha_batch, normal_batch, depth_batch, instance_batch = next(data_iterator)

                position_grad_over_time.append(model.primal_points.grad.norm().item() if model.primal_points.grad is not None else 0.0)
                att_dc_grad_over_time.append(model.att_dc.grad.norm().item() if model.att_dc.grad is not None else 0.0)
                att_sh_grad_over_time.append(model.att_sh.grad.norm().item() if model.att_sh.grad is not None else 0.0)
                density_grad_over_time.append(model.density.grad.norm().item() if model.density.grad is not None else 0.0)

                model.optimizer.step()
                model.update_learning_rate(i)

                # Probe rays
                if i >= pipeline_args.probe_from and ((i - pipeline_args.probe_from) % pipeline_args.probe_every == 0):
                    k = int(pipeline_args.probe_sample_size * ray_batch.size(0))
                    perm = torch.randperm(ray_batch.size(0))
                    idx = perm[:k]
                    probe_ray_batch = ray_batch[idx]
                    t = model.probe_rays(probe_ray_batch, max_intersections=100)

                color_loss_mean = color_loss.mean().item()
                train.set_postfix(color_loss=f"{color_loss_mean:.5f}")

                loss_over_time.append(loss.item())
                opacity_loss_over_time.append(opacity_loss.item())
                depth_loss_over_time.append(depth_loss_value.item())
                quant_loss_over_time.append(quant_loss.item())
                w_depth_over_time.append(w_depth)
                color_loss_over_time.append(color_loss_mean)
                num_points_over_time.append(model.primal_points.shape[0])

                if i % 100 == 99 and not pipeline_args.debug:
                    writer.add_scalar("train/rgb_loss", color_loss.mean(), i)
                    num_points = model.primal_points.shape[0]
                    writer.add_scalar("test/num_points", num_points, i)

                    test_psnr = test_render(
                        test_data_handler,
                        test_ray_batch_fetcher,
                        test_rgb_batch_fetcher,
                        str(i),
                        pipeline_args.debug,
                        i % 5000 == 0
                    )
                    writer.add_scalar("test/psnr", test_psnr, i)

                    test_psnr_over_time.append(float(test_psnr))
            
                    writer.add_scalar(
                        "lr/points_lr", model.xyz_scheduler_args(i), i
                    )
                    writer.add_scalar(
                        "lr/density_lr", model.den_scheduler_args(i), i
                    )
                    writer.add_scalar(
                        "lr/attr_lr", model.attr_dc_scheduler_args(i), i
                    )
                else:
                    test_psnr_over_time.append(test_psnr_over_time[-1] if test_psnr_over_time else 0)

                if iters_since_update >= triangulation_update_period:
                    model.update_triangulation(incremental=True)
                    iters_since_update = 0

                    if triangulation_update_period < 100:
                        triangulation_update_period += 2

                iters_since_update += 1
                if i + 1 >= pipeline_args.densify_from:
                    iters_since_densification += 1

                if (
                    iters_since_densification == next_densification_after
                    and model.primal_points.shape[0]
                    < 0.9 * model.num_final_points
                ):
                    point_error, point_contribution = model.collect_error_map(
                        train_data_handler, pipeline_args.white_background
                    )
                    model.prune_and_densify(
                        point_error,
                        point_contribution,
                        pipeline_args.densify_factor,
                    )

                    model.update_triangulation(incremental=False)
                    triangulation_update_period = 1
                    gc.collect()

                    # Linear growth
                    iters_since_densification = 0
                    next_densification_after = int(
                        (
                            (pipeline_args.densify_factor - 1)
                            * model.primal_points.shape[0]
                            * (
                                pipeline_args.densify_until
                                - pipeline_args.densify_from
                            )
                        )
                        / (model.num_final_points - model.num_init_points)
                    )
                    next_densification_after = max(
                        next_densification_after, 100
                    )

                if i == optimizer_args.freeze_points:
                    model.update_triangulation(incremental=False)

                if viewer is not None and viewer.is_closed():
                    break

        # Save training statistics including error sampling data
        with open(f"{out_dir}/stats.csv", "w") as f:
            f.write(
                "iteration,loss,color_loss,opacity_loss,depth_loss,quant_loss,w_depth,num_points,test_psnr,position_grad,att_dc_grad,att_sh_grad,density_grad\n"
            )
            for i in range(len(loss_over_time)):
                f.write(
                    f"{i},{loss_over_time[i]},{color_loss_over_time[i]},{opacity_loss_over_time[i]},{depth_loss_over_time[i]},{quant_loss_over_time[i]},{w_depth_over_time[i]},{num_points_over_time[i]},{test_psnr_over_time[i]},{position_grad_over_time[i]},{att_dc_grad_over_time[i]},{att_sh_grad_over_time[i]},{density_grad_over_time[i]}\n"
                )
        
        # Save error sampling statistics if available
        if error_stats_history:
            import json
            with open(f"{out_dir}/error_sampling_stats.json", "w") as f:
                json.dump(error_stats_history, f, indent=2)

        model.save_ply(f"{out_dir}/scene.ply")
        model.save_pt(f"{out_dir}/model.pt")
        del data_iterator

    if pipeline_args.viewer:
        model.show(
            train_loop, iterations=pipeline_args.iterations, **viewer_options
        )
    else:
        train_loop(viewer=None)
    if not pipeline_args.debug:
        writer.close()

    test_render(
        test_data_handler,
        test_ray_batch_fetcher,
        test_rgb_batch_fetcher,
        "final",
        pipeline_args.debug,
        True # Save final output images
    )


def main():
    parser = configargparse.ArgParser(
        default_config_files=["arguments/mipnerf360_outdoor_config.yaml"]
    )

    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser)

    # Add argument to specify a custom config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    # Parse arguments
    args = parser.parse_args()

    train(
        args,
        pipeline_params.extract(args),
        model_params.extract(args),
        optimization_params.extract(args),
        dataset_params.extract(args),
    )


if __name__ == "__main__":
    main()
