import csv
import numpy as np
from dataclasses import dataclass

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