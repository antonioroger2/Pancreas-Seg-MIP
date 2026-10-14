"""nnU-Net v2 trainer with a wall-clock budget (scripts/nnunet_run.sh links it into nnunetv2's trainer folder).

After every epoch the number of epochs is re-planned from the measured epoch time so that training ends within
NNUNET_BUDGET_MIN minutes of epoch time; the poly LR schedule is stretched/compressed to that end point, so the LR
still decays to ~0 at the last epoch (the full nnU-Net recipe, compressed). Never more than NNUNET_MAX_EPOCHS,
never past the budget: if epochs get slower, the run ends sooner. Resume (--c) keeps the time already spent.
"""
import os

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerBudget(nnUNetTrainer):
    budget_min = float(os.environ.get("NNUNET_BUDGET_MIN", "210"))
    max_epochs = int(os.environ.get("NNUNET_MAX_EPOCHS", "1000"))
    warmup = 2                                      # epoch 0 includes torch.compile; plan only from epoch 2 on

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):   # explicit: nnU-Net introspects this signature
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.max_epochs

    def _replan(self):
        log = self.logger.local_logger.my_fantastic_logging
        start, end = log["epoch_start_timestamps"], log["epoch_end_timestamps"]
        done = len(end)
        if done <= self.warmup:
            return
        t_epoch = float(np.median(np.diff(start)[1:][-20:]))   # start-to-start: includes checkpoint/plot overhead
        used = end[-1] - start[0]                               # wall clock (a resume gap counts: conservative)
        left = self.budget_min * 60 - used
        self.num_epochs = int(min(self.max_epochs, done + max(0, left // t_epoch)))   # 0 left -> stop now
        self.lr_scheduler.max_steps = self.num_epochs
        self.print_to_log_file(f"budget: {used / 60:.1f}/{self.budget_min:.0f} min used, {t_epoch:.1f} s/epoch "
                               f"-> {self.num_epochs} epochs, ETA {(self.num_epochs - done) * t_epoch / 60:.1f} min")

    def on_train_start(self):
        super().on_train_start()
        self._replan()                              # resumed run: re-plan before the first LR step

    def on_epoch_end(self):
        super().on_epoch_end()
        self._replan()

    def run_training(self):                         # same loop as nnUNetTrainer, but num_epochs may change
        self.on_train_start()
        while self.current_epoch < self.num_epochs:
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = [self.train_step(next(self.dataloader_train)) for _ in range(self.num_iterations_per_epoch)]
            self.on_train_epoch_end(train_outputs)
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = [self.validation_step(next(self.dataloader_val))
                               for _ in range(self.num_val_iterations_per_epoch)]
                self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()
        self.on_train_end()


class nnUNetTrainerBudget_smoke(nnUNetTrainerBudget):
    """End-to-end check of the pipeline (train -> validation -> post-processing -> test): 4 epochs."""
    budget_min, max_epochs = 6.0, 4


if __name__ == "__main__":                          # self-check of the budget logic (CPU, no data): python -m src.nnunet_trainer
    import types

    def simulate(budget_min, slow_after=None):
        t = nnUNetTrainerBudget.__new__(nnUNetTrainerBudget)
        t.budget_min, t.max_epochs, t.num_epochs = budget_min, 1000, 1000
        t.lr_scheduler, t.print_to_log_file = types.SimpleNamespace(max_steps=1000), lambda *_: None
        log = {"epoch_start_timestamps": [], "epoch_end_timestamps": []}
        t.logger = types.SimpleNamespace(local_logger=types.SimpleNamespace(my_fantastic_logging=log))
        s, e = 0.0, 0
        while e < t.num_epochs:                     # same condition as run_training
            d = 66 if e == 0 else (90 if slow_after and e >= slow_after else 59)   # measured on the RTX 6000 Ada
            log["epoch_start_timestamps"].append(s); log["epoch_end_timestamps"].append(s + d); s += d + 3
            e += 1
            t._replan()
            assert t.lr_scheduler.max_steps == t.num_epochs >= e
        return e, log["epoch_end_timestamps"][-1] / 60

    for b, slow in [(210, None), (210, 100), (6, None)]:
        n, used = simulate(b, slow)
        assert used <= b + 0.1, (b, slow, n, used)
        print(f"budget {b} min, slow_after {slow}: {n} epochs, {used:.1f} min")
