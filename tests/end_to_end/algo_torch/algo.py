import torch

import numpy as np

from pathlib import Path
from typing import Optional, Tuple, Dict

from connectlib.algorithms import Algo, register


@register
class MyAlgo(Algo):
    def __init__(self):
        self.module = torch.nn.Linear(256, 1)
        self.optimizer = torch.optim.SGD(self.module.parameters(), 1e-3)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def preprocessing(
        self, x: np.array, y: Optional[np.array] = None
    ) -> Tuple[np.array, np.array]:
        return x, y

    def perform_update(self, x: np.array, y: np.array):
        self.optimizer.zero_grad()

        y_hat = self.module(x)
        loss = self.criterion(y_hat, y)
        loss.backward()
        self.optimizer.step()

    def test(self, x: np.array):
        return self.module(x)

    @property
    def weights(self) -> Dict[str, np.array]:
        return {k: v.numpy() for k, v in self.module.state_dict().items()}

    @weights.setter
    def weights(self, weights: Dict[str, np.array]):
        state_dict = self.module.state_dict()

        for k in state_dict.keys():
            state_dict[k] = torch.from_numpy(weights[k])

        self.module.load_state_dict(state_dict)

    def load(self, path: Path):
        state = torch.load(path)

        self.module.load_state_dict(state["module"])
        self.optimizer.load_state_dict(state["optimizer"])

    def save(self, path: Path):
        torch.save(
            {
                "module": self.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
