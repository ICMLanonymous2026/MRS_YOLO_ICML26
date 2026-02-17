# utils/feature_hooks.py
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn

class FeatureHooks:
    """
    Gère des forward/backward hooks sur des modules donnés.
    - features[name] contiendra la dernière sortie forward du module `name`
    - grads[name] contiendra le gradient d'output (utile pour Grad-CAM-like)
    """
    def __init__(self, detach: bool = True, cpu: bool = False, keep_last_only: bool = True):
        self.detach = detach
        self.cpu = cpu
        self.keep_last_only = keep_last_only
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self.features: Dict[str, torch.Tensor] = OrderedDict()
        self.grads: Dict[str, torch.Tensor] = OrderedDict()

    def _maybe_store(self, store: Dict[str, torch.Tensor], key: str, value: torch.Tensor):
        t = value
        if self.detach: t = t.detach()
        if self.cpu:    t = t.cpu()
        if self.keep_last_only:
            store[key] = t
        else:
            store.setdefault(key, []).append(t)

    def add(self, module: nn.Module, name: str, with_backward: bool = False):
        # Forward hook → enregistre la sortie du module
        def fwd_hook(_m, _inp, out):
            self._maybe_store(self.features, name, out)
            # attache un hook gradient sur out pour backward si demandé
            if with_backward and isinstance(out, torch.Tensor) and out.requires_grad:
                def _save_grad(g):
                    self._maybe_store(self.grads, name, g)
                out.register_hook(_save_grad)

        h1 = module.register_forward_hook(fwd_hook)
        self._handles.append(h1)

        # Backward hook au niveau module (optionnel si on préfère grad d'output côté module)
        if with_backward:
            # bwd_hook reçoit (grad_input, grad_output). On prend grad_output[0]
            def bwd_hook(_m, gin, gout):
                if isinstance(gout, (list, tuple)) and len(gout) > 0 and isinstance(gout[0], torch.Tensor):
                    self._maybe_store(self.grads, name, gout[0])
            h2 = module.register_backward_hook(bwd_hook)
            self._handles.append(h2)

    def clear(self):
        self.features.clear()
        self.grads.clear()

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.clear()


def attach_sequential_hooks(seq: nn.Sequential, prefix: str, hooks: FeatureHooks, types=None):
    """
    - seq: nn.Sequential (ex. model.backbone.branches[k].model)
    - types: list de classes à garder (ex. [Conv, TFSepBlock]); None => tout
    """
    for idx, m in enumerate(seq):
        if (types is None) or any(isinstance(m, t) for t in types):
            name = f"{prefix}_{idx:02d}_{m.__class__.__name__}"
            hooks.add(m, name=name)
