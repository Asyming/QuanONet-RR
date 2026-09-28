"""Render the five report figures from the released numerical source data."""
import argparse
import csv
import math
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = Path(__file__).resolve().parents[2]
WIDTH = 130.8 / 25.4
SHORT = ['Anti', 'Hom', 'Nonlin', 'D–R', 'Adv', 'Darcy']
PAIRED_MODELS = ['TFQuanONet', 'DeepONet']
BUDGETS = [3, 1, 2]
ORIGINAL = ["Antideriv", "Homogeneous", "Nonlinear", "RDiffusion", "Advection", "Darcy"]
EXTENSION = ["Burgers", "AllenCahn", "Wave"]
MODELS = ["TFQuanONet", "FFQuanONet", "TFHEA", "DeepONet"]
LABELS = {"TFQuanONet": "TF-QuanONet", "FFQuanONet": "QuanONet", "TFHEA": "TF-HEA", "FFHEA": "HEA", "DeepONet": "DeepONet", "FNN": "FNN"}
TITLES = {"Antideriv": "Antiderivative", "Homogeneous": "Homogeneous", "Nonlinear": "Nonlinear", "RDiffusion": "Diffusion–reaction", "Advection": "Advection", "Darcy": "Darcy", "Burgers": "Burgers", "AllenCahn": "Allen–Cahn", "Wave": "Wave"}
COLORS = {"TFQuanONet": "#0072B2", "FFQuanONet": "#56B4E9", "TFHEA": "#CC79A7", "FFHEA": "#999999", "DeepONet": "#D55E00", "FNN": "#E69F00"}
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]

def read_csv(path):
    def value(text):
        if re.fullmatch(r'-?\d+', text):
            return int(text)
        if re.fullmatch(r'-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', text):
            return float(text)
        return text if text else None
    with path.open(newline='') as handle:
        return [{key: value(item) for key, item in row.items()} for row in csv.DictReader(handle)]

def stats(values):
    return float(np.mean(values)), float(np.std(values, ddof=1))

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
def base_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9.5, "axes.titlesize": 10, "axes.titleweight": "normal",
        "axes.linewidth": 0.65, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.fontsize": 8.5, "legend.frameon": False,
        "lines.linewidth": 1.35, "lines.markersize": 3.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
    })
def style():
    base_style()
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.2,
                         "lines.linewidth": 1.0, "lines.markersize": 2.5, "xtick.major.size": 2,
                         "ytick.major.size": 2, "axes.linewidth": 0.55})
def letter(axis, label):
    axis.text(-0.21, 1.10, label, transform=axis.transAxes, fontsize=9, fontweight="bold")
def save(fig, name, args):
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figures_dir / f"{name}.pdf")
    fig.savefig(args.figures_dir / f"{name}.png", dpi=300)
    plt.close(fig)
def model_handles(variants=MODELS):
    return [Line2D([], [], color=COLORS[variant], label=LABELS[variant], marker="o", markersize=3) for variant in variants]
def fig1(args):
    variants = ["TFQuanONet", "FFQuanONet", "TFHEA", "FFHEA"]
    pairs = read_csv(args.input / 'fig1_paired_runs.csv')
    summaries, paired_rows = [], []
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 99 / 25.4))
    fig.subplots_adjust(left=0.11, right=0.985, top=0.86, bottom=0.18, wspace=0.40, hspace=0.40)
    for axis, operator in zip(axes.flat, ORIGINAL):
        limits = [0.0]
        for model_index, variant in enumerate(variants):
            for scale, offset, hatch in [(0.001, -0.17, "///"), (0.01, 0.17, "")]:
                group = [r for r in pairs if r['operator'] == operator and r['variant'] == variant and r['scale'] == scale]
                differences = [r['paired_difference_pp'] for r in group]
                paired_rows.extend(group)
                mean, sd = stats(differences)
                axis.bar(model_index + offset, mean, width=0.31, color=COLORS[variant], edgecolor="#333333", linewidth=0.3, hatch=hatch, yerr=sd, capsize=1.1, error_kw=dict(elinewidth=0.6, capthick=0.6, zorder=6))
                axis.scatter(model_index + offset, mean, s=11, facecolors="white", edgecolors=COLORS[variant], linewidths=0.75, zorder=5)
                limits.extend([mean - sd, mean + sd])
                summaries.append(dict(operator=operator, variant=variant, scale=scale, reference_scale=0.1, mean_difference_pp=mean, sample_sd_difference_pp=sd, repetitions=5))
        padding = max(1.0, (max(limits) - min(limits)) * 0.10)
        axis.set(title=TITLES[operator], xlim=(-0.65, 3.65), ylim=(min(limits) - padding, max(limits) + padding))
        axis.set_xticks(range(4), ["TFQ", "FFQ", "TFH", "FFH"])
        axis.axhline(0, color="#555555", linewidth=0.65)
        axis.grid(axis="y", color="#E8E8E8", linewidth=0.4)
        axis.set_axisbelow(True)
    for axis, label in zip(axes[:, 0], "ab"):
        letter(axis, label)
    for axis in axes[:, 0]:
        axis.set_ylabel("Change in error (pp)")
    fig.legend(handles=[Patch(facecolor="white", edgecolor="#333333", hatch=hatch, label=f"{scale:g}") for scale, hatch in [(0.001, "///"), (0.01, "")]], title=r"Encoding scale $\omega_0$", title_fontsize=7.5, loc="upper center", ncol=2, bbox_to_anchor=(0.54, 1.005), handlelength=1.2, columnspacing=1.4)
    fig.legend(handles=model_handles(variants), loc="lower center", ncol=2, bbox_to_anchor=(0.54, 0.005), handlelength=1.2, columnspacing=1.5, labelspacing=0.4)
    save(fig, "fig1_encoding_scale", args)
    write_csv(args.source_dir / "fig1.csv", summaries)
    write_csv(args.source_dir / "fig1_paired_runs.csv", paired_rows)

def fig2(args):
    histories = read_csv(args.input / 'fig2_training_runs.csv')
    summaries = []
    scaling = (read_csv(args.input / 'fig2_scaling_runs.csv'), read_csv(args.input / 'fig2_scaling_summary.csv'))
    fig = plt.figure(figsize=(WIDTH, 140 / 25.4))
    grid = fig.add_gridspec(3, 6, left=0.105, right=0.97, top=0.94, bottom=0.185,
                          hspace=0.55, wspace=1.00)
    axes = [fig.add_subplot(grid[row, 2 * column:2 * column + 2]) for row in range(2) for column in range(3)]
    axes.extend((fig.add_subplot(grid[2, :3]), fig.add_subplot(grid[2, 3:])))
    for axis, operator in zip(axes[:6], ORIGINAL):
        observed = []
        for variant in MODELS:
            by_seed = {seed: {row["step"]: row for row in histories if row["operator"] == operator
                        and row["variant"] == variant and row["seed"] == seed} for seed in range(5)}
            expected_steps = set(range(0, 100001, 5000))
            steps = sorted(expected_steps)
            values = np.asarray([[by_seed[seed][step]["val_mse"] for step in steps] for seed in range(5)])
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            observed.extend(values.ravel())
            axis.plot(np.asarray(steps) / 1000, mean, color=COLORS[variant])
            axis.scatter(0, mean[0], color=COLORS[variant], s=7, zorder=4)
            axis.fill_between(np.asarray(steps) / 1000, np.maximum(mean - sd, np.finfo(float).tiny), mean + sd,
                              color=COLORS[variant], alpha=0.15, linewidth=0)
            summaries.extend(dict(operator=operator, variant=variant, step=step, mean_val_mse=value,
                                  sample_sd_val_mse=deviation, repetitions=5) for step, value, deviation in zip(steps, mean, sd))
        axis.set(title=TITLES[operator], yscale="log", xlim=(0, 100), xticks=[0, 50, 100],
                 ylim=(min(observed) * 0.6, max(observed) * 1.6))
    for index in (0, 3):
        axes[index].set_ylabel("Validation MSE")
    for axis in axes[3:6]:
        axis.set_xlabel("Updates (×10³)")
    scaling_runs, scaling_summary = scaling
    dimensions = [4, 8, 16, 32, 64, 128, 256]
    error, size = axes[6:]
    bounds = []
    for variant in PAIRED_MODELS:
        group = sorted((row for row in scaling_summary if row["variant"] == variant), key=lambda row: row["latent_dimension"])
        for split, linestyle, facecolor in (("train", "--", "white"), ("test", "-", COLORS[variant])):
            means = np.asarray([row[f"{split}_mse_mean"] for row in group])
            deviations = np.asarray([row[f"{split}_mse_sample_sd"] for row in group])
            bounds.extend((means - deviations).tolist() + (means + deviations).tolist())
            error.errorbar(dimensions, means, yerr=deviations, color=COLORS[variant], linestyle=linestyle,
                           marker="o", markerfacecolor=facecolor, markersize=2.5, capsize=1.1,
                           elinewidth=0.55, barsabove=True)
        size.plot(dimensions, [row["parameters"] for row in group], color=COLORS[variant], marker="o", markersize=2.5)
    error_scale = "log" if min(bounds) > 0 and max(bounds) / min(bounds) >= 10 else "linear"
    error.set(title="Antiderivative", ylabel="MSE", yscale=error_scale)
    if error_scale == "linear":
        error.set_ylim(bottom=0)
        error.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    size.set(title="Model size", ylabel="Parameters", yscale="log")
    for axis in (error, size):
        axis.set_xscale("log", base=2)
        axis.set_xticks(dimensions, [str(p) for p in dimensions])
        axis.set(xlabel="Latent dimension $p$", xlim=(3.5, 290))
    for axis, label in zip((axes[0], error, size), "abc"):
        letter(axis, label)
    for axis in axes:
        axis.grid(axis="y", color="#E8E8E8", linewidth=0.4)
        axis.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], color="#444444", linestyle=linestyle, marker="o", markersize=2.5,
                                    markerfacecolor=face, label=label)
                        for linestyle, face, label in (("--", "white", "Train"), ("-", "#444444", "Test"))],
               loc="center", bbox_to_anchor=(0.55, 0.105), ncol=2, handlelength=1.7, columnspacing=1.5)
    fig.legend(handles=model_handles(), loc="lower center", ncol=2, bbox_to_anchor=(0.55, 0.005),
               columnspacing=1.5, handlelength=1.3)
    save(fig, "fig2_training", args)
    write_csv(args.source_dir / "fig2_training_runs.csv", histories)
    write_csv(args.source_dir / "fig2_training_summary.csv", summaries)
    write_csv(args.source_dir / "fig2_scaling_runs.csv", scaling_runs)
    write_csv(args.source_dir / "fig2_scaling_summary.csv", scaling_summary)

def fig3(args):
    data_rows = read_csv(args.input / 'fig3_training_runs.csv')
    noise_rows = read_csv(args.input / 'fig3_noise_runs.csv')
    augmentation = read_csv(args.input / 'fig3_augmentation_runs.csv')
    fig = plt.figure(figsize=(WIDTH, 146 / 25.4))
    axes = [fig.add_axes((0.13, 0.79, 0.855, 0.135))]
    axes.extend(fig.add_axes((left, bottom, 0.365, 0.135)) for bottom in (0.54, 0.32, 0.085) for left in (0.13, 0.62))
    summaries = []
    sizes, markers = [100, 250, 500, 1000], ["o", "s", "^", "D"]
    for task, operator in enumerate(ORIGINAL):
        for variant, model_offset in (("TFQuanONet", -0.22), ("DeepONet", 0.22)):
            points, means = [], []
            for size, marker, offset in zip(sizes, markers, (-0.13, -0.045, 0.045, 0.13)):
                group = [r for r in data_rows if r["operator"] == operator and r["variant"] == variant and r["train_functions"] == size]
                mean, sd = stats([r["relative_l2_percent"] for r in group])
                position = task + model_offset + offset
                axes[0].errorbar(position, mean, yerr=sd, fmt=marker, color=COLORS[variant], markersize=2.4,
                                capsize=0.8, elinewidth=0.55, barsabove=True)
                points.append(position)
                means.append(mean)
                summaries.append(dict(experiment="training_functions", operator=operator, variant=variant,
                                      condition=size, mean_percent=mean, sample_sd_percent=sd, repetitions=5))
            axes[0].plot(points, means, color=COLORS[variant], linewidth=0.55)
    axes[0].set_xticks(range(6), SHORT)
    axes[0].set(xlim=(-0.6, 5.6), ylim=(0, 24), yticks=[0, 10, 20], ylabel="Test error (%)")
    handles = model_handles(PAIRED_MODELS) + [Line2D([], [], color="#555555", marker=marker,
              linestyle="none", markersize=3, label=f"{size:,}") for size, marker in zip(sizes, markers)]
    fig.legend(handles=handles, ncol=6, loc="upper center", bbox_to_anchor=(0.535, 0.995),
               handlelength=1.0, handletextpad=0.25, columnspacing=0.55)
    shades = ["#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]
    for axis, variant in zip(axes[1:5], MODELS):
        for task, operator in enumerate(ORIGINAL):
            for level, color, offset in zip(NOISE_LEVELS, shades, (-0.3, -0.1, 0.1, 0.3)):
                group = [r for r in noise_rows if r["operator"] == operator and r["variant"] == variant and r["noise_percent"] == 100 * level]
                mean, sd = stats([r["relative_l2_percent"] for r in group])
                axis.bar(task + offset, mean, width=0.18, color=color, yerr=sd, capsize=0.65,
                         error_kw=dict(elinewidth=0.5, capthick=0.5))
                summaries.append(dict(experiment="input_noise", operator=operator, variant=variant,
                                      condition=100 * level, mean_percent=mean, sample_sd_percent=sd, repetitions=5))
        axis.set(title=LABELS[variant], xlim=(-0.6, 5.6), ylim=(0, 60), yticks=[0, 20, 40, 60])
        axis.set_xticks(range(6), SHORT)
    fig.legend(handles=[Patch(facecolor=color, label=f"{100 * level:g}%") for color, level in zip(shades, NOISE_LEVELS)],
               title="Input noise", title_fontsize=7, ncol=4, loc="center", bbox_to_anchor=(0.56, 0.73),
               handlelength=1.0, columnspacing=1.1)
    augmented_summaries = []
    limits = []
    for axis, operator in zip(axes[5:], ("Advection", "Darcy")):
        for variant in PAIRED_MODELS:
            for training_noise, linestyle in ((0.0, "-"), (0.05, "--")):
                means, deviations = [], []
                for level in NOISE_LEVELS:
                    group = [r for r in augmentation if (r["operator"], r["variant"], r["training_noise"], r["testing_noise"])
                             == (operator, variant, training_noise, level)]
                    mean, sd = stats([r["relative_l2_percent"] for r in group])
                    means.append(mean)
                    deviations.append(sd)
                    limits.append(mean + sd)
                    augmented_summaries.append(dict(operator=operator, variant=variant, training_noise=training_noise,
                                              testing_noise=level, mean_percent=mean, sample_sd_percent=sd, repetitions=5))
                if operator == "Darcy":
                    axis.errorbar(np.asarray(NOISE_LEVELS) * 100, means, yerr=deviations, linestyle=linestyle,
                                  color=COLORS[variant], marker="o", markersize=2.2, capsize=1.0,
                                  elinewidth=0.55, barsabove=True)
        axis.set(title=TITLES[operator], xlabel="Input noise (%)", xlim=(-0.4, 10.4), xticks=[0, 1, 5, 10])
    upper = 10 * math.ceil(max(limits) * 1.05 / 10)
    for axis in axes[5:]:
        axis.set_ylim(0, upper)
    references = read_csv(args.input / 'robustness_reference_runs.csv')
    reference_names = ['original_labels', 'nominal_reference', 'scheme_aligned_reference']
    reference_summary = []
    for reference_index, reference in enumerate(reference_names):
        for variant, model_offset in (("TFQuanONet", -0.2), ("DeepONet", 0.2)):
            for training_noise, offset, hatch in ((0.0, -0.1, ""), (0.05, 0.1, "///")):
                group = [r for r in references if r['experiment'] == 'noise_training'
                         and r['operator'] == 'Advection' and r['variant'] == variant
                         and r['reference'] == reference and r['training_noise'] == training_noise
                         and r['testing_noise'] == 0.1]
                mean, sd = stats([r['relative_l2_percent'] for r in group])
                axes[5].bar(reference_index + model_offset + offset, mean, width=0.18,
                            color=COLORS[variant], edgecolor="#333333", linewidth=0.3,
                            hatch=hatch, yerr=sd, capsize=0.8,
                            error_kw=dict(elinewidth=0.55, capthick=0.55))
                reference_summary.append(dict(reference=reference, variant=variant, training_noise=training_noise,
                                              testing_noise=0.1, mean_percent=mean, sample_sd_percent=sd, repetitions=5))
    axes[5].set(title="Advection (10% noise)", xlabel="Reference", xlim=(-0.55, 2.55),
                xticks=[0, 1, 2], xticklabels=["Released", "Ref. (i)", "Ref. (ii)"])
    fig.legend(handles=[(Patch(facecolor="white", edgecolor="#444444", linewidth=0.5, hatch=hatch),
                         Line2D([], [], color="#444444", linestyle=linestyle))
                        for linestyle, hatch in (("-", ""), ("--", "///"))],
               labels=["Clean training", "5% noise training"], handler_map={tuple: HandlerTuple(ndivide=None, pad=0.45)},
               ncol=2, loc="upper center", bbox_to_anchor=(0.565, 0.29), handlelength=2.4, columnspacing=1.1)
    write_csv(args.source_dir / "fig3_reference_augmentation_summary.csv", reference_summary)
    for axis in (axes[1], axes[3], axes[5]):
        axis.set_ylabel("Test error (%)")
    for axis in axes:
        axis.grid(axis="y", color="#E8E8E8", linewidth=0.4)
        axis.set_axisbelow(True)
    for axis, label in zip((axes[0], axes[1], axes[5]), "abc"):
        axis.text(-0.085 if label == "a" else -0.22, 1.12, label, transform=axis.transAxes, fontsize=9, fontweight="bold")
    save(fig, "fig3_robustness", args)
    write_csv(args.source_dir / "fig3_original_summary.csv", summaries)
    write_csv(args.source_dir / "fig3_training_runs.csv", data_rows)
    write_csv(args.source_dir / "fig3_noise_runs.csv", noise_rows)
    write_csv(args.source_dir / "fig3_augmentation_runs.csv", augmentation)
    write_csv(args.source_dir / "fig3_augmentation_summary.csv", augmented_summaries)

def fig4(args):
    rows = read_csv(args.input / 'fig4_global_runs.csv')
    features = read_csv(args.input / 'fig4_local_repetitions.csv')
    regional = read_csv(args.input / 'fig4_local_summary.csv')
    individual = read_csv(args.input / 'fig4_local_functions.csv')
    for row in features:
        row.update(equation=row['operator'], local_rmse_mean=row['local_rmse'], global_rmse_mean=row['full_grid_rmse'])
        keys = ('operator', 'variant', 'budget')
        row['region'] = next(r['region'] for r in regional if all(r[k] == row[k] for k in keys))
        row['per_function'] = [{k: v for k, v in r.items() if k in ('function_id', 'global_rmse', 'local_rmse', 'local_points')}
                               for r in individual if all(r[k] == row[k] for k in (*keys, 'seed'))]
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH, 122 / 25.4))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.93, bottom=0.15, wspace=0.40, hspace=0.42)
    feature_summaries, feature_repetitions, feature_functions = [], [], []
    region_titles = {"Burgers": "High-gradient region", "AllenCahn": "Interface region", "Wave": "Time trace"}
    for column, operator in enumerate(EXTENSION):
        upper, middle, lower = axes[:, column]
        upper.set_title(TITLES[operator])
        for index, variant in enumerate(MODELS):
            group = [r for r in rows if r["operator"] == operator and r["variant"] == variant and r["budget"] == 1]
            values = [r["relative_l2_percent"] for r in group]
            mean, sd = stats(values)
            upper.bar(index, mean, color=COLORS[variant], width=0.65, alpha=0.8, yerr=sd, capsize=1.4,
                      error_kw=dict(elinewidth=0.65, capthick=0.6, zorder=5))
            upper.scatter(index + np.linspace(-0.13, 0.13, 5), values, facecolors="white",
                          edgecolors=COLORS[variant], s=6, linewidths=0.5, zorder=4)
        upper.set(xticks=[], xlim=(-0.6, 3.6), ylim=(0, 110), yticks=[0, 50, 100])
        for variant in PAIRED_MODELS:
            x, means, deviations, local_means, local_deviations = [], [], [], [], []
            for budget in BUDGETS:
                group = [r for r in rows if r["operator"] == operator and r["variant"] == variant and r["budget"] == budget]
                local = [r for r in features if r["equation"] == operator and r["variant"] == variant and r["budget"] == budget]
                counts = {r["parameters"] for r in group + local}
                parameters = counts.pop()
                x.append(parameters / 1000)
                mean, sd = stats([r["relative_l2_percent"] for r in group])
                means.append(mean)
                deviations.append(sd)
                local_mean, local_sd = stats([r["local_rmse_mean"] for r in local])
                local_means.append(local_mean)
                local_deviations.append(local_sd)
                header = dict(operator=operator, variant=variant, budget=budget, parameters=parameters)
                feature_summaries.append(dict(**header, mean_local_rmse=local_mean, sample_sd_local_rmse=local_sd,
                                         repetitions=5, functions_per_repetition=16, region=local[0]["region"],
                                         excluded_empty_regions=sum(r["excluded_empty_regions"] for r in local)))
                for item in local:
                    feature_repetitions.append(dict(**header, seed=item["seed"], local_rmse=item["local_rmse_mean"],
                                               full_grid_rmse=item["global_rmse_mean"],
                                               excluded_empty_regions=item["excluded_empty_regions"],
                                               checkpoint=item["checkpoint"]))
                    feature_functions.extend(dict(**header, seed=item["seed"], **function) for function in item["per_function"])
            middle.errorbar(x, means, yerr=deviations, color=COLORS[variant], marker="o", markersize=2.5,
                            capsize=1.5, elinewidth=0.6, barsabove=True)
            lower.errorbar(x, local_means, yerr=local_deviations, color=COLORS[variant], marker="o",
                           markersize=2.5, capsize=1.5, elinewidth=0.6, barsabove=True)
        for axis in (middle, lower):
            axis.set(xlim=(1.0, 5.2), xticks=[1.25, 2.5, 5])
        lower.set(xlabel="Parameters (×10³)", title=region_titles[operator])
        lower.title.set_fontsize(7.2)
    axes[0, 0].set_ylabel("Test error (%)")
    axes[1, 0].set_ylabel("Test error (%)")
    axes[2, 0].set_ylabel("Local RMSE")
    for axis, label in zip(axes[:, 0], "abc"):
        letter(axis, label)
    for axis in axes.flat:
        axis.grid(axis="y", color="#E8E8E8", linewidth=0.4)
        axis.set_axisbelow(True)
    fig.legend(handles=model_handles(), loc="lower center", ncol=2, bbox_to_anchor=(0.55, 0.00),
               columnspacing=1.5, handlelength=1.3)
    save(fig, "fig4_new_equations", args)
    write_csv(args.source_dir / "fig4_global_runs.csv", rows)
    write_csv(args.source_dir / "fig4_local_summary.csv", feature_summaries)
    write_csv(args.source_dir / "fig4_local_repetitions.csv", feature_repetitions)
    write_csv(args.source_dir / "fig4_local_functions.csv", feature_functions)

def fig5(args):
    profiles_source = read_csv(args.input / 'fig5_profiles.csv')
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH, 132 / 25.4))
    fig.subplots_adjust(left=0.085, right=0.98, top=0.94, bottom=0.12, wspace=0.45, hspace=0.50)
    source = []
    for index, operator in enumerate(EXTENSION):
        with np.load(args.input / f'fig5_{operator}.npz', allow_pickle=False) as fields:
            x, t, truth, quantum, classical = [fields[k] for k in ('x', 't', 'reference', 'TFQuanONet', 'DeepONet')]
        function_id = next(r['function_id'] for r in profiles_source if r['operator'] == operator)
        limit = max(float(abs(truth).max()), float(abs(quantum).max()))
        for column, field in enumerate((truth, quantum)):
            axis = axes[index, column]
            mesh = axis.pcolormesh(t, x, field, cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True, shading="auto")
            axis.set(xlabel="$t$", xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], xlim=(0, 1), ylim=(0, 1))
            axis.set_xticklabels(["0", "0.5", "1"])
            axis.set_yticklabels(["0", "0.5", "1"])
            if column == 0:
                axis.set_ylabel("$x$")
            else:
                axis.set_yticklabels([])
            if index == 0:
                axis.set_title("Reference" if column == 0 else "TF-QuanONet")
            color_axis = make_axes_locatable(axis).append_axes("right", size="4%", pad=0.018)
            colorbar = fig.colorbar(mesh, cax=color_axis, ticks=[-limit, 0, limit], format="%.1f")
            colorbar.ax.tick_params(labelsize=7.2, pad=1, length=2)
        axis = axes[index, 2]
        if operator == "Wave":
            at = int(np.argmin(abs(x - 0.2)))
            coordinate, profiles = t, [field[at, :] for field in (truth, quantum, classical)]
            axis.set_xlabel("$t$")
            annotation, slice_value = f"$x={x[at]:.1f}$", float(x[at])
        else:
            at = int(np.argmin(abs(t - (0.5 if operator == "Burgers" else 1))))
            coordinate, profiles = x, [field[:, at] for field in (truth, quantum, classical)]
            axis.set_xlabel("$x$")
            annotation, slice_value = (f"$t={t[at]:.2f}$" if operator == "Burgers" else "$t=1$"), float(t[at])
        axis.plot(coordinate, profiles[0], color="#333333", linestyle="--", linewidth=0.85)
        axis.plot(coordinate, profiles[1], color=COLORS["TFQuanONet"])
        axis.plot(coordinate, profiles[2], color=COLORS["DeepONet"])
        axis.set(xticks=[0, 0.5, 1], title=TITLES[operator])
        axis.set_xticklabels(["0", "0.5", "1"])
        axis.text(-0.08, 1.025, "$u$", transform=axis.transAxes, fontsize=8)
        axis.text(0.04 if operator == "Burgers" else 0.96, 0.95, annotation, ha="left" if operator == "Burgers" else "right", va="top", transform=axis.transAxes, fontsize=7.2, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.3))
        source.extend(dict(operator=operator, seed=0, function_id=function_id, coordinate=float(value), reference=float(profiles[0][k]), TFQuanONet=float(profiles[1][k]), DeepONet=float(profiles[2][k]), slice_coordinate=slice_value) for k, value in enumerate(coordinate))
        np.savez_compressed(args.source_dir / f"fig5_{operator}.npz", x=x, t=t, reference=truth, TFQuanONet=quantum, DeepONet=classical)
    for axis, label in zip(axes[:, 0], "abc"):
        letter(axis, label)
    fig.legend(handles=[Line2D([], [], color="#333333", linestyle="--", label="Reference"), *model_handles(["TFQuanONet", "DeepONet"])], ncol=3, loc="lower center", bbox_to_anchor=(0.54, 0.01), handlelength=1.4, columnspacing=1.1)
    save(fig, "fig5_solution_profiles", args)
    write_csv(args.source_dir / "fig5_profiles.csv", source)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/paper/source_data')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/generated')
    parser.add_argument('--figures', type=int, nargs='+', choices=range(1, 6), default=list(range(1, 6)))
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()
    if args.num_workers != 1:
        parser.error('The figure renderer uses one process.')
    args.figures_dir = args.out / 'figures'
    args.source_dir = args.out / 'source_data'
    if args.source_dir.resolve() == args.input.resolve():
        parser.error('Keep generated summaries separate from input source data.')
    args.source_dir.mkdir(parents=True, exist_ok=True)
    style()
    for number in args.figures:
        globals()[f'fig{number}'](args)
        print(f'Figure {number} complete', flush=True)

if __name__ == '__main__':
    main()
