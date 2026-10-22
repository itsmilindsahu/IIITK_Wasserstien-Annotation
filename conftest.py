import sys
import types

try:
    import torch
except ModuleNotFoundError:
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    F = types.ModuleType("torch.nn.functional")
    utils = types.ModuleType("torch.nn.utils")
    rnn = types.ModuleType("torch.nn.utils.rnn")
    optim = types.ModuleType("torch.optim")

    class DummyModule:
        def __init__(self, *args, **kwargs):
            pass

        def parameters(self):
            return []

        def eval(self):
            return self

        def train(self, mode=True):
            return self

        def __call__(self, *args, **kwargs):
            return args[0] if args else None

    class DummyTensor:
        def __init__(self, *args, **kwargs):
            pass

    nn.Module = DummyModule
    nn.Linear = DummyModule
    nn.Conv1d = DummyModule
    nn.Dropout = DummyModule
    nn.LSTM = DummyModule
    nn.functional = F
    rnn.pad_sequence = lambda *args, **kwargs: None
    utils.rnn = rnn
    nn.utils = utils
    optim.AdamW = lambda *args, **kwargs: None

    torch.nn = nn
    torch.optim = optim
    torch.Tensor = DummyTensor
    torch.from_numpy = lambda arr: arr
    torch.save = lambda *args, **kwargs: None
    torch.load = lambda *args, **kwargs: {}
    torch.manual_seed = lambda seed: None

    for n, m in [
        ("torch", torch),
        ("torch.nn", nn),
        ("torch.nn.functional", F),
        ("torch.nn.utils", utils),
        ("torch.nn.utils.rnn", rnn),
        ("torch.optim", optim),
    ]:
        sys.modules[n] = m

try:
    import cv2
except ModuleNotFoundError:
    cv2 = types.ModuleType("cv2")
    sys.modules["cv2"] = cv2
