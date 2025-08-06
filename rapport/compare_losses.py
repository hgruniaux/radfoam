from stats import load_stats
import matplotlib.pyplot as plt

# experiments = ["classroom_depth_loss_scale_0", "classroom_depth_loss_scale_0.01", "classroom_depth_loss_scale_0.001", "classroom_depth_loss_scale_0.0001"]
experiments = {"depth_loss_scale_0", "depth_loss_scale_0.001", "depth_loss_scale_0.01", "depth_loss_scale_0.05", "depth_loss_scale_0.1"}
experiments_stats = []

for experiment in experiments:
    stats_file = f"tests/{experiment}/stats.csv"
    stats = load_stats(stats_file)
    stats.experiment_name = experiment
    stats.depth_scale = experiment.split('_')[-1]
    experiments_stats.append(stats)

for stats in experiments_stats:
    plt.plot(stats.iteration, stats.depth_loss, label=f"Depth loss with $\\lambda_{{depth}}={stats.depth_scale}$")

# plt.xlim(0, 1000)

plt.ylim(0, 30)
plt.xlabel("Iteration")
plt.ylabel("Depth Loss")
plt.title('Depth Loss Over Iterations For Different Depth Loss Scales')
plt.legend()
plt.savefig("depth_loss_over_iterations.svg")
plt.close()

for stats in experiments_stats:
    plt.plot(stats.iteration, stats.color_loss, label=f"Color Loss with $\\lambda_{{depth}}={stats.depth_scale}$")

plt.xlabel("Iteration")
plt.ylabel("Color Loss")
plt.title('Color Loss Over Iterations For Different Depth Loss Scales')
plt.legend()
plt.savefig("color_loss_over_iterations.svg")
plt.close()

for stats in experiments_stats:
    plt.plot(stats.iteration, stats.loss, label=f"Loss with $\\lambda_{{depth}}={stats.depth_scale}$")

plt.ylim(0, 0.1)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title('Loss Over Iterations For Different Depth Loss Scales')
plt.legend()
plt.savefig("loss_over_iterations.svg")
plt.close()
