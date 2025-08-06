import csv
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import sys

@dataclass
class Stats:
    iteration: np.ndarray[tuple[int], np.dtype[np.int_]]
    loss: np.ndarray[tuple[int], np.dtype[np.float32]]
    color_loss: np.ndarray[tuple[int], np.dtype[np.float32]]
    opacity_loss: np.ndarray[tuple[int], np.dtype[np.float32]]
    depth_loss: np.ndarray[tuple[int], np.dtype[np.float32]]
    quant_loss: np.ndarray[tuple[int], np.dtype[np.float32]]
    w_depth: np.ndarray[tuple[int], np.dtype[np.float32]]
    num_points: np.ndarray[tuple[int], np.dtype[np.int_]]
    test_psnr: np.ndarray[tuple[int], np.dtype[np.float32]]
    position_grad: np.ndarray[tuple[int], np.dtype[np.float32]]
    att_dc_grad: np.ndarray[tuple[int], np.dtype[np.float32]]
    att_sh_grad: np.ndarray[tuple[int], np.dtype[np.float32]]
    density_grad: np.ndarray[tuple[int], np.dtype[np.float32]]


def load_stats(file_path) -> Stats:
    iteration = []
    loss = []
    color_loss = []
    opacity_loss = []
    depth_loss = []
    quant_loss = []
    w_depth = []
    num_points = []
    test_psnr = []
    position_grad = []
    att_dc_grad = []
    att_sh_grad = []
    density_grad = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            

            iteration.append(int(row[0]))
            loss.append(float(row[1]))
            color_loss.append(float(row[2]))
            opacity_loss.append(float(row[3]))
            depth_loss.append(float(row[4]))
            quant_loss.append(float(row[5]))
            w_depth.append(float(row[6]))
            num_points.append(int(row[7]))
            test_psnr.append(float(row[8]))
            position_grad.append(float(row[9]))
            att_dc_grad.append(float(row[10]))
            att_sh_grad.append(float(row[11]))
            density_grad.append(float(row[12]))

    iteration = np.array(iteration, dtype=np.int_)
    loss = np.array(loss, dtype=np.float32)
    color_loss = np.array(color_loss, dtype=np.float32)
    opacity_loss = np.array(opacity_loss, dtype=np.float32)
    depth_loss = np.array(depth_loss, dtype=np.float32)
    quant_loss = np.array(quant_loss, dtype=np.float32)
    w_depth = np.array(w_depth, dtype=np.float32)
    num_points = np.array(num_points, dtype=np.int_)
    test_psnr = np.array(test_psnr, dtype=np.float32)
    position_grad = np.array(position_grad, dtype=np.float32) 
    att_dc_grad = np.array(att_dc_grad, dtype=np.float32)
    att_sh_grad = np.array(att_sh_grad, dtype=np.float32)
    density_grad = np.array(density_grad, dtype=np.float32)

    return Stats(
        iteration=iteration,
        loss=loss,
        color_loss=color_loss,
        opacity_loss=opacity_loss,
        depth_loss=depth_loss,
        quant_loss=quant_loss,
        w_depth=w_depth,
        num_points=num_points,
        test_psnr=test_psnr,
        position_grad=position_grad,
        att_dc_grad=att_dc_grad,
        att_sh_grad=att_sh_grad,
        density_grad=density_grad
    )

def plot_stats(stats: Stats, save_path=None):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 3, 1)
    plt.plot(stats.iteration, stats.loss, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over Iterations')
    plt.legend()

    plt.subplot(3, 3, 2)
    plt.plot(stats.iteration, stats.test_psnr, label='Test PSNR', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Test PSNR')
    plt.title('Test PSNR over Iterations')
    plt.legend()

    plt.subplot(3, 3, 3)
    plt.plot(stats.iteration, stats.num_points, label='Number of Points', color='purple')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Points')
    plt.title('Number of Points over Iterations')
    plt.legend()

    plt.subplot(3, 3, 4)
    plt.plot(stats.iteration, stats.color_loss, label='Color Loss', color='green')
    plt.plot(stats.iteration, stats.opacity_loss, label='Opacity Loss', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Components')
    plt.title('Color and Opacity Loss over Iterations')
    plt.legend()

    plt.subplot(3, 3, 5)
    plt.plot(stats.iteration, stats.depth_loss, label='Depth Loss', color='brown')
    plt.xlabel('Iteration')
    plt.ylabel('Depth Loss')
    plt.title('Depth Loss over Iterations')
    plt.legend()

    plt.subplot(3, 3, 6)
    plt.plot(stats.iteration, stats.quant_loss, label='Quant Loss', color='pink')
    plt.xlabel('Iteration')
    plt.ylabel('Quant Loss')
    plt.title('Quant Loss over Iterations')
    plt.legend()

    plt.subplot(3, 3, 7)
    plt.plot(stats.iteration, stats.w_depth, label='Weighted Depth', color='cyan')
    plt.xlabel('Iteration')
    plt.ylabel('Weighted Depth')
    plt.title('Weighted Depth over Iterations')
    plt.legend()

    plt.subplot(3, 3, 8)
    plt.plot(stats.iteration, stats.position_grad, label='Position Grad', color='magenta')
    plt.xlabel('Iteration')
    plt.ylabel('Position Grad')
    plt.title('Position Gradient over Iterations')
    plt.legend()

    plt.subplot(3, 3, 9)
    plt.plot(stats.iteration, stats.att_dc_grad, label='Att DC Grad', color='teal')
    plt.plot(stats.iteration, stats.att_sh_grad, label='Att SH Grad', color='lime')
    plt.plot(stats.iteration, stats.density_grad, label='Density Grad', color='gray')
    plt.xlabel('Iteration')
    plt.ylabel('Gradients')
    plt.title('SH Coefficients and Density Gradients over Iterations')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

stats = load_stats("{}/stats.csv".format(sys.argv[1]))
plot_stats(stats, save_path="{}/stats_plot.png".format(sys.argv[1]))
plot_stats(stats, save_path="{}/stats_plot.svg".format(sys.argv[1]))
