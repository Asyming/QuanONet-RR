#!/usr/bin/env python3
"""Run released QuanONet architectures on frozen operator datasets."""
import argparse
import json
import logging
import math
import os
import pickle
import platform
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "QuanONet"))
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("DDE_BACKEND", "pytorch")
STOP = False


def stop_requested(signum, frame):
    global STOP
    STOP = True


def atomic_pickle(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_completion_metadata(args, metadata, metrics):
    if args.evaluate_only:
        metadata = dict(metadata, requested_noise_levels=list(args.noise_levels),
                        test_noise_levels=sorted({value["input_noise_fraction_of_training_std"]
                                                  for value in metrics.values() if value["split"] == "test"}),
                        metric_keys=sorted(metrics))
        atomic_json(args.out / "evaluation_completed.json", metadata)
    else:
        atomic_json(args.out / "completed.json", metadata)








def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--variant", choices=["TFQuanONet", "FFQuanONet", "TFHEA", "FFHEA", "DeepONet", "FNN"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--budget", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mq-workers", type=int, default=0, help="MindQuantum batch workers; 0 preserves released automatic setting")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--train-functions", type=int)
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=1000)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=100)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--evaluate-only", action="store_true")
    phase.add_argument("--train-only", action="store_true")
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0])
    parser.add_argument("--noise-seed", type=int, default=20260923)
    parser.add_argument("--wall-minutes", type=float, default=0)
    parser.add_argument("--grid", type=Path)
    parser.add_argument("--grid-functions", type=int, default=3)
    return parser.parse_args()


def load_arrays(args):
    with np.load(args.data, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files if key.startswith(("train_", "test_", "val_")) and key not in ("train_input", "test_input", "val_input")}
    for split, number in [("train", args.num_train), ("test", args.num_test), ("val", args.num_val)]:
        if split + "_output" not in data:
            continue
        for suffix in ("branch_input", "trunk_input", "output"):
            key = split + "_" + suffix
            data[key] = np.asarray(data[key], dtype=np.float32)
            if data[key].ndim != 2 or not np.isfinite(data[key]).all():
                raise ValueError(f"Invalid array: {key}")
        rows = len(data[split + "_output"])
        if split + "_function_id" not in data:
            if rows % number:
                raise ValueError(f"Cannot infer function IDs for {split}; provide explicit IDs")
            data[split + "_function_id"] = np.repeat(np.arange(number), rows // number)
        if len(data[split + "_function_id"]) != rows:
            raise ValueError(f"Function IDs do not match {split} outputs")
    if args.train_functions:
        ids = np.unique(data["train_function_id"])
        if args.train_functions > len(ids):
            raise ValueError("Requested more training functions than available")
        selected = np.isin(data["train_function_id"], ids[:args.train_functions])
        for key in list(data):
            if key.startswith("train_") and len(data[key]) == len(selected):
                data[key] = data[key][selected]
    if args.variant in ("TFHEA", "FFHEA", "FNN"):
        for split in ("train", "test", "val"):
            if split + "_output" in data:
                data[split + "_input"] = np.concatenate((data[split + "_branch_input"], data[split + "_trunk_input"]), axis=1)
    return data


def configuration(args, data):
    ode = args.operator in ("Antideriv", "Homogeneous", "Nonlinear")
    models = {"TFQuanONet": "QuanONet", "FFQuanONet": "QuanONet", "TFHEA": "HEAQNN", "FFHEA": "HEAQNN", "DeepONet": "DeepONet", "FNN": "FNN"}
    nets_ode = {"TFQuanONet": [20, 2, 10, 2], "FFQuanONet": [20, 2, 20, 2], "TFHEA": [32, 2], "FFHEA": [40, 2], "DeepONet": [2, 10, 2, 10], "FNN": [2, 10]}
    nets_pde = {"TFQuanONet": [40, 2, 20, 2], "FFQuanONet": [40, 2, 40, 2], "TFHEA": [64, 2], "FFHEA": [80, 2], "DeepONet": [3, 15, 3, 15], "FNN": [3, 16]}
    if args.budget == 2:
        if ode or args.variant not in ("TFQuanONet", "DeepONet"):
            raise ValueError("Second budget is defined only for TFQuanONet/DeepONet PDE extensions")
        nets_pde["TFQuanONet"] = [80, 2, 40, 2]
        nets_pde["DeepONet"] = [3, 24, 3, 24]
    if args.budget == 3:
        if ode or args.variant not in ("TFQuanONet", "DeepONet"):
            raise ValueError("Third budget is defined only for TFQuanONet/DeepONet PDE extensions")
        nets_pde["TFQuanONet"] = [20, 2, 10, 2]
        nets_pde["DeepONet"] = [2, 10, 2, 10]
    quantum = args.variant.startswith(("TF", "FF"))
    if quantum and args.device != "cpu":
        raise ValueError("Released MindQuantum mqvector runs on CPU; use --device cpu")
    batches = math.ceil(len(data["train_output"]) / 100)
    if args.steps % batches:
        raise ValueError("Steps must end at an epoch boundary to preserve released MS selection")
    return dict(operator=args.operator, variant=args.variant, budget=args.budget, model_type=models[args.variant],
                net_size=(nets_ode if ode else nets_pde)[args.variant], num_qubits=5,
                if_trainable_freq=str(args.variant.startswith("TF")).lower(), scale_coeff=args.scale,
                ham_bound=[-5, 5], ham_pauli="Z", quantum_backend="mindquantum", classical_backend="pytorch",
                seed=args.seed, learning_rate=1e-4, optimizer="adam", batch_size=100,
                num_epochs=args.steps // batches, optimizer_steps=args.steps,
                num_train=int(len(np.unique(data["train_function_id"]))),
                num_test=int(len(np.unique(data["test_function_id"]))), train_query_pairs=len(data["train_output"]),
                test_query_pairs=len(data["test_output"]), normalization="none",
                data_path=os.path.relpath(args.data.resolve(), ROOT),
                device=args.device, simulator="mqvector" if quantum else None,
                checkpoint_selection="minimum_epoch_mean_training_mse" if quantum else "released_deepxde_logged_minibatch_training_mse")


def bound_mq_workers(workers):
    if workers == 0:
        return
    from mindquantum.simulator import Simulator
    original = Simulator.get_expectation_with_grad
    def bounded(simulator, *args, **kwargs):
        kwargs.setdefault("parallel_worker", workers)
        return original(simulator, *args, **kwargs)
    Simulator.get_expectation_with_grad = bounded


def ms_model(config, data, workers, mq_workers):
    import mindspore as ms
    from solvers.solver_ms import MSSolver
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU", runtime_num_threads=workers)
    bound_mq_workers(mq_workers)
    holder = SimpleNamespace(config=config, data=data, model_type=config["model_type"], logger=logging.getLogger("rr"))
    model = MSSolver._create_model(holder)
    optimizer = MSSolver._build_optimizer(holder, model.trainable_params(), config["optimizer_steps"])
    return model, optimizer


def validation_curve(args, config, model, data, state):
    if "val_output" not in data:
        return
    records = state.setdefault("validation_curve", [])
    if records and records[-1]["step"] == state["step"]:
        return
    start = time.perf_counter()
    ids = data["val_function_id"]
    selected = np.unique(ids)[:32]
    indices = np.concatenate([np.flatnonzero(ids == item)[:100] for item in selected])
    quantum = config["simulator"] == "mqvector"
    if quantum:
        import mindspore as ms
        training_mode = model.training
        framework_rng = ms.get_rng_state()
        model.set_train(False)
    else:
        import torch
        training_mode = model.training
        framework_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if args.device == "cuda" else None
        model.eval()
    predictions = []
    rng = np.random.get_state()
    for offset in range(0, len(indices), 100):
        index = indices[offset:offset + 100]
        branch, trunk = data["val_branch_input"][index], data["val_trunk_input"][index]
        arrays = (branch, trunk) if config["model_type"] in ("QuanONet", "DeepONet") else np.concatenate((branch, trunk), axis=1)
        if quantum:
            values = tuple(ms.Tensor(a) for a in arrays) if isinstance(arrays, tuple) else ms.Tensor(arrays)
            predictions.append(model(values).asnumpy())
        else:
            values = tuple(torch.as_tensor(a, device=args.device) for a in arrays) if isinstance(arrays, tuple) else torch.as_tensor(arrays, device=args.device)
            with torch.no_grad():
                predictions.append(model(values).cpu().numpy())
    if quantum:
        model.set_train(training_mode)
        ms.set_rng_state(framework_rng)
    else:
        model.train(training_mode)
        torch.set_rng_state(framework_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    np.random.set_state(rng)
    target = data["val_output"][indices].astype(np.float64)
    residual = np.concatenate(predictions).astype(np.float64) - target
    records.append(dict(step=state["step"], val_mse=float(np.mean(residual ** 2)),
                        val_relative_l2=float(np.linalg.norm(residual) / (np.linalg.norm(target) + 1e-8)),
                        functions=len(selected), queries=len(indices), seconds=time.perf_counter() - start))
    atomic_json(args.out / "validation_curve.json", dict(protocol="First 32 sorted validation function IDs, first 100 stored queries per function; every 5000 updates; never used for checkpoint selection", selected_function_ids=selected.tolist(), first_step=records[0]["step"], records=records))


def train_ms(args, config, data, deadline):
    import mindspore as ms
    import mindspore.nn as nn
    model, optimizer = ms_model(config, data, args.num_workers, args.mq_workers)
    train_net = nn.TrainOneStepCell(nn.WithLossCell(model, nn.MSELoss()), optimizer)
    train_net.set_train()
    total_rows = len(data["train_output"])
    batches = math.ceil(total_rows / 100)
    state = dict(step=0, epoch=0, batch=0, epoch_loss=0.0, best_loss=float("inf"), best_step=None,
                 permutation=None, history=[], training_seconds=0.0)
    checkpoint = args.out / "resume.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            saved = pickle.load(handle)
        state = saved["state"]
        if state.get("best_parameters"):
            ms.save_checkpoint([{"name": key, "data": ms.Tensor(value.copy())}
                                for key, value in state["best_parameters"].items()], str(args.out / "best.ckpt"))
        missing = ms.load_param_into_net(train_net, {key: ms.Parameter(ms.Tensor(value), name=key) for key, value in saved["parameters"].items()}, strict_load=True)
        if any(missing):
            raise RuntimeError(f"Incomplete optimizer/model restoration: {missing}")
        np.random.set_state(saved["numpy_rng"])
        if "mindspore_rng" in saved:
            ms.set_rng_state(ms.Tensor(saved["mindspore_rng"]))
    last_wall = time.perf_counter()

    def save():
        nonlocal last_wall
        now = time.perf_counter()
        state["training_seconds"] += now - last_wall
        atomic_pickle(checkpoint, dict(state=state, numpy_rng=np.random.get_state(), mindspore_rng=ms.get_rng_state().asnumpy(), parameters={p.name: p.asnumpy() for p in train_net.get_parameters()}))
        last_wall = time.perf_counter()
        atomic_json(args.out / "progress.json", dict(step=state["step"], target_steps=args.steps, epoch=state["epoch"], best_train_loss=state["best_loss"] if np.isfinite(state["best_loss"]) else None, training_seconds=state["training_seconds"], host=platform.node()))

    if not args.evaluate_only:
        validation_curve(args, config, model, data, state)
    while state["step"] < args.steps and not args.evaluate_only:
        if state["batch"] == 0:
            state["permutation"] = np.random.permutation(total_rows)
        batch = state["batch"]
        index = state["permutation"][batch * 100:(batch + 1) * 100]
        if config["model_type"] == "QuanONet":
            inputs = (ms.Tensor(data["train_branch_input"][index]), ms.Tensor(data["train_trunk_input"][index]))
        else:
            inputs = ms.Tensor(data["train_input"][index])
        loss = float(train_net(inputs, ms.Tensor(data["train_output"][index])).asnumpy())
        if not np.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {state['step'] + 1}")
        state["epoch_loss"] += loss
        state["step"] += 1
        state["batch"] += 1
        if state["batch"] == batches:
            average = state["epoch_loss"] / batches
            state["history"].append(dict(step=state["step"], epoch=state["epoch"], train_mse=average))
            if average < state["best_loss"]:
                state["best_loss"] = average
                state["best_step"] = state["step"]
                state["best_parameters"] = {p.name: p.asnumpy().copy() for p in model.get_parameters()}
                ms.save_checkpoint(model, str(args.out / "best.tmp.ckpt"))
                os.replace(args.out / "best.tmp.ckpt", args.out / "best.ckpt")
            state.update(epoch=state["epoch"] + 1, batch=0, epoch_loss=0.0, permutation=None)
        if state["step"] % 5000 == 0 or state["step"] == args.steps:
            validation_curve(args, config, model, data, state)
        if state["step"] % args.checkpoint_every == 0 or STOP or time.monotonic() > deadline:
            save()
            print(json.dumps(dict(step=state["step"], train_mse=loss, elapsed_seconds=state["training_seconds"])), flush=True)
        if STOP or time.monotonic() > deadline:
            return model, state, False
    if not args.evaluate_only:
        save()
        ms.save_checkpoint(model, str(args.out / "final.ckpt"))
    if state["step"] != args.steps:
        raise RuntimeError("Cannot evaluate an unfinished run as a completed experiment")
    missing = ms.load_param_into_net(model, ms.load_checkpoint(str(args.out / "best.ckpt")), strict_load=True)
    if any(missing):
        raise RuntimeError(f"Incomplete best-model loading: {missing}")
    model.set_train(False)
    return model, state, True


def train_classical(args, config, data, deadline):
    import torch
    import deepxde as dde
    from solvers.solver_dde import DDESolver, BestModelCheckpoint
    torch.set_num_threads(args.num_workers)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dde.config.set_default_float("float32")
    holder = SimpleNamespace(config=config, data_dict=data, model_type=config["model_type"], logger=logging.getLogger("rr"))
    model = DDESolver._build_model(holder)
    model.net.to(args.device)
    model.compile("adam", lr=config["learning_rate"], metrics=["l2 relative error"])
    checkpoint = args.out / "resume.pt"
    state = dict(step=0, best_loss=float("inf"), best_step=None, history=[], training_seconds=0.0)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
        state = saved["state"]
        if state.get("best_parameters"):
            torch.save(state["best_parameters"], args.out / "best.pt")
        model.net.load_state_dict(saved["model"])
        model.opt.load_state_dict(saved["optimizer"])
        model.data.train_sampler.__dict__.update(saved["sampler"])
        model.train_state.__dict__.update(saved["train_state"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if args.device == "cuda":
            torch.cuda.set_rng_state_all([rng.cpu() for rng in saved["cuda_rng"]])
    last_wall = time.perf_counter()

    def save():
        nonlocal last_wall
        now = time.perf_counter()
        state["training_seconds"] += now - last_wall
        retained = {key: value for key, value in model.train_state.__dict__.items() if key not in ("X_test", "y_test", "y_pred_test", "best_y", "best_ystd")}
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(dict(state=state, model=model.net.state_dict(), optimizer=model.opt.state_dict(), sampler=model.data.train_sampler.__dict__, train_state=retained, numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if args.device == "cuda" else None), temporary)
        os.replace(temporary, checkpoint)
        last_wall = time.perf_counter()
        atomic_json(args.out / "progress.json", dict(step=state["step"], target_steps=args.steps, best_train_loss=state["best_loss"] if np.isfinite(state["best_loss"]) else None, training_seconds=state["training_seconds"], host=platform.node()))

    class Recorder(dde.callbacks.Callback):
        def on_epoch_end(self):
            state["step"] = self.model.train_state.step
            value = float(sum(self.model.train_state.loss_train))
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite classical training loss")
            if value < state["best_loss"]:
                state["best_loss"] = value
                state["best_step"] = state["step"]
                state["best_parameters"] = {key: value.detach().cpu().clone() for key, value in self.model.net.state_dict().items()}
                torch.save(self.model.net.state_dict(), args.out / "best.pt.tmp")
                os.replace(args.out / "best.pt.tmp", args.out / "best.pt")
            if state["step"] % 1000 == 0 or state["step"] == args.steps:
                state["history"].append(dict(step=state["step"], train_mse=value, test_mse=float(sum(self.model.train_state.loss_test))))
            if state["step"] % 5000 == 0 or state["step"] == args.steps:
                validation_curve(args, config, self.model.net, data, state)
            if state["step"] % args.checkpoint_every == 0 or STOP or time.monotonic() > deadline:
                save()
            if STOP or time.monotonic() > deadline:
                self.model.stop_training = True

    if not args.evaluate_only and state["step"] < args.steps:
        validation_curve(args, config, model.net, data, state)
        remaining = args.steps - state["step"]
        if state["step"]:
            model.batch_size = 100
            model.callbacks = dde.callbacks.CallbackList(callbacks=[Recorder()])
            model.callbacks.set_model(model)
            model.train_state.set_data_test(*model.data.test())
            model.stop_training = False
            model._train_sgd(remaining, 1000)
        else:
            model.train(iterations=remaining, batch_size=100, callbacks=[Recorder()], display_every=1000)
        save()
    if state["step"] != args.steps:
        return model.net, state, False
    if not args.evaluate_only:
        torch.save(model.net.state_dict(), args.out / "final.pt")
    model.net.load_state_dict(torch.load(args.out / "best.pt", map_location=args.device, weights_only=True))
    model.net.eval()
    return model.net, state, True


def evaluate(args, config, model, data, split, noise):
    quantum = config["simulator"] == "mqvector"
    branch = data[split + "_branch_input"]
    trunk = data[split + "_trunk_input"]
    truth = data[split + "_output"]
    function_ids, inverse = np.unique(data[split + "_function_id"], return_inverse=True)
    first = np.unique(inverse, return_index=True)[1]
    if noise:
        clean_functions = branch[first]
        generator = np.random.RandomState(args.noise_seed)
        perturbation = generator.standard_normal(clean_functions.shape).astype(np.float32)
        train_first = np.unique(data["train_function_id"], return_index=True)[1]
        reference_std = float(np.std(data["train_branch_input"][train_first], dtype=np.float64))
        perturbation *= noise * reference_std
    prediction = np.empty_like(truth)
    start_time = time.perf_counter()
    if quantum:
        import mindspore as ms
    else:
        import torch
    for start in range(0, len(truth), args.eval_batch_size):
        end = min(start + args.eval_batch_size, len(truth))
        values = branch[start:end]
        if noise:
            values = values + perturbation[inverse[start:end]]
        if config["model_type"] in ("QuanONet", "DeepONet"):
            arrays = (values, trunk[start:end])
        else:
            arrays = np.concatenate((values, trunk[start:end]), axis=1)
        if quantum:
            inputs = tuple(ms.Tensor(a) for a in arrays) if isinstance(arrays, tuple) else ms.Tensor(arrays)
            prediction[start:end] = model(inputs).asnumpy()
        else:
            inputs = tuple(torch.as_tensor(a, device=args.device) for a in arrays) if isinstance(arrays, tuple) else torch.as_tensor(arrays, device=args.device)
            with torch.no_grad():
                prediction[start:end] = model(inputs).cpu().numpy()
    residual2 = (prediction.astype(np.float64) - truth) ** 2
    truth2 = truth.astype(np.float64) ** 2
    sse = np.bincount(inverse, weights=residual2.ravel())
    norm2 = np.bincount(inverse, weights=truth2.ravel())
    relative = np.sqrt(sse) / (np.sqrt(norm2) + 1e-8)
    tag = split if noise == 0 else f"{split}_noise{noise:g}"
    np.savez_compressed(args.out / f"predictions_{tag}.npz", truth=truth, prediction=prediction, function_id=data[split + "_function_id"], trunk_input=trunk, unique_function_id=function_ids, function_rel_l2=relative, function_sse=sse, function_norm2=norm2)
    return dict(split=split, input_noise_fraction_of_training_std=noise, noise_seed=args.noise_seed, input_noise_std=float(noise * reference_std) if noise else 0.0, noise_reference="global_clean_training_branch_std", relative_l2=float(np.sqrt(residual2.sum()) / (np.sqrt(truth2.sum()) + 1e-8)), mse=float(residual2.mean()), mae=float(np.abs(prediction - truth).mean()), max_error=float(np.abs(prediction - truth).max()), per_function_relative_l2_mean=float(relative.mean()), per_function_relative_l2_std=float(relative.std(ddof=1)) if len(relative) > 1 else 0.0, per_function_relative_l2_median=float(np.median(relative)), functions=len(function_ids), queries=len(truth), seconds=time.perf_counter() - start_time)


def main():
    args = arguments()
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_arrays(args)
    config = configuration(args, data)
    if config["simulator"] == "mqvector":
        config["quantum_parallel_workers"] = args.mq_workers or "released_default"
    config_path = args.out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output directory contains a different experiment configuration")
    atomic_json(config_path, config)
    from utils.common import set_random_seed
    set_random_seed(args.seed)
    deadline = time.monotonic() + args.wall_minutes * 60 if args.wall_minutes else float("inf")
    quantum = config["simulator"] == "mqvector"
    model, state, completed = (train_ms if quantum else train_classical)(args, config, data, deadline)
    if not completed:
        print("Training checkpoint saved; target steps not yet reached", flush=True)
        return
    atomic_json(args.out / "history.json", state["history"])
    if args.train_only:
        print("Training target reached; full evaluation deferred to a long allocation", flush=True)
        return
    metrics_path = args.out / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    for split in ("test", "val"):
        if split + "_output" not in data:
            continue
        for noise in (args.noise_levels if split == "test" else [0.0]):
            key = split if noise == 0 else f"{split}_noise{noise:g}"
            if key not in metrics:
                metrics[key] = evaluate(args, config, model, data, split, noise)
                atomic_json(metrics_path, metrics)
                print(json.dumps(metrics[key]), flush=True)
    if args.grid:
        with np.load(args.grid, allow_pickle=False) as archive:
            grid = {key: archive[key] for key in archive.files}
        if "u0" in grid or "branch" in grid:
            branch = grid.get("u0", grid.get("branch"))
            selected = np.argsort(grid["function_id"])[:args.grid_functions]
            queries = int(np.prod(grid["solution"].shape[1:]))
            if "coordinates" in grid:
                coordinates = grid["coordinates"]
            else:
                coordinates = np.column_stack((np.repeat(grid["x"], len(grid["t"])), np.tile(grid["t"], len(grid["x"]))))
            if coordinates.ndim == 3:
                trunk = coordinates[selected].reshape(-1, coordinates.shape[-1])
            else:
                trunk = np.tile(coordinates, (len(selected), 1))
            grid = dict(grid_branch_input=np.repeat(branch[selected], queries, axis=0).astype(np.float32),
                        grid_trunk_input=trunk.astype(np.float32),
                        grid_output=grid["solution"][selected].reshape(-1, 1).astype(np.float32),
                        grid_function_id=np.repeat(grid["function_id"][selected], queries))
        metrics["grid"] = evaluate(args, config, model, grid, "grid", 0.0)
        metrics["grid"].update(data_path=os.path.relpath(args.grid.resolve(), ROOT))
        atomic_json(metrics_path, metrics)
    parameter_count = (sum(int(p.size) for p in model.trainable_params()) if quantum
                       else sum(p.numel() for p in model.parameters()))
    write_completion_metadata(args, dict(optimizer_steps=state["step"], parameters=parameter_count,
        best_step=state["best_step"], best_training_loss=state["best_loss"],
        training_seconds=state["training_seconds"]), metrics)
    print("Completed training and full evaluation", flush=True)


if __name__ == "__main__":
    main()
