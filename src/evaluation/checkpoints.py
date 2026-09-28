"""Load the original model architectures and their selected report weights."""
import logging
from pathlib import Path
from types import SimpleNamespace

from src.training import run as training

ROOT = Path(__file__).resolve().parents[2]


def load_model(directory, config, data, num_workers=1, mq_workers=1, device='cpu'):
    quantum = config['simulator'] == 'mqvector'
    if quantum:
        import mindspore as ms
        from solvers.solver_ms import MSSolver
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU', runtime_num_threads=num_workers)
        training.bound_mq_workers(mq_workers)
        holder = SimpleNamespace(config=config, data=data, model_type=config['model_type'], logger=logging.getLogger('evaluate'))
        model = MSSolver._create_model(holder)
        ms.load_param_into_net(model, ms.load_checkpoint(str(directory / 'best.ckpt')), strict_load=True)
        model.set_train(False)
    else:
        import torch
        import deepxde as dde
        from solvers.solver_dde import DDESolver
        torch.set_num_threads(num_workers)
        dde.config.set_default_float('float32')
        holder = SimpleNamespace(config=config, data_dict=data, model_type=config['model_type'], logger=logging.getLogger('evaluate'))
        model = DDESolver._build_model(holder).net.to(device)
        model.load_state_dict(torch.load(directory / 'best.pt', map_location=device, weights_only=True), strict=True)
        model.eval()
    return model
