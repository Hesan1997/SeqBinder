import torch
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm


class Trainer:
    """
    A flexible training class for PyTorch models.
    """
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, primaryLoader, secondaryLoader, scoreFuncs=None, saveAddress=None, progressBar=True):
        """
        Initializes the Trainer.

        Args:
            primaryLoader: DataLoader for training data.
            secondaryLoader: DataLoader for validation data.
            scoreFuncs: Dictionary of scoring functions for metrics (default: None).
            saveAddress: Path to save model state (default: None).
            progressBar: Whether to display progress bars (default: True).
        """
        self.primaryLoader = primaryLoader
        self.secondaryLoader = secondaryLoader
        self.scoreFuncs = scoreFuncs or {}
        self.saveAddress = saveAddress
        self.progressBar = progressBar

        keys = ["Epoch", "Training Loss", "Validation Loss"]
        for scoreFunc in self.scoreFuncs:
            keys.append(f"Training {scoreFunc}")
            keys.append(f"Validation {scoreFunc}")

        self.records = {key: [] for key in keys}

    def run(self, module, errorFunc, scheduler=None, parsUpdater=None, numEpochs=16):
        """
        Runs the training and validation process.

        Args:
            module: The PyTorch model to train.
            errorFunc: The loss function.
            scheduler: Learning rate scheduler (default: None).
            parsUpdater: Optimizer (default: AdamW optimizer).
            numEpochs: Number of epochs to train (default: 16).

        Returns:
            DataFrame: Training and validation records.
        """
        self.errorFunc = errorFunc
        self.parsUpdater = parsUpdater or torch.optim.AdamW(module.parameters())
        cleaner = parsUpdater is None  # Track if we created the optimizer

        module.to(Trainer.DEVICE)
        for epoch in tqdm(range(numEpochs), desc="Epoch", disable=not self.progressBar):
            self.records["Epoch"].append(epoch)

            # Training phase
            module.train()
            self.run_epoch(module, self.primaryLoader, prefix="Training", desc="Training")

            # Validation phase
            module.eval()
            with torch.no_grad():
                self.run_epoch(module, self.secondaryLoader, prefix="Validation", desc="Validating")

            # Scheduler step
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(self.records["Validation Loss"][-1])
                else:
                    scheduler.step()

        # Save the model state if needed
        if self.saveAddress:
            torch.save({
                'epoch': epoch,
                'model_state_dict': module.state_dict(),
                'optimizer_state_dict': self.parsUpdater.state_dict(),
                'results': self.records
            }, self.saveAddress)

        if cleaner:
            del self.parsUpdater

        return pd.DataFrame.from_dict(self.records)

    def run_epoch(self, module, currLoader, prefix="", desc=None):
        """
        Runs a single epoch.

        Args:
            module: The PyTorch model.
            currLoader: The DataLoader for the current phase.
            prefix: Prefix for logging (e.g., "Training" or "Validation").
            desc: Description for the progress bar.
        """
        runningLoss = []
        yTrue, yPred = [], []

        with torch.enable_grad() if module.training else torch.no_grad():
            for XP, XL, y in tqdm(currLoader, desc=desc, leave=False):
                XP, XL, y = moveTo(XP, Trainer.DEVICE), moveTo(XL, Trainer.DEVICE), moveTo(y, Trainer.DEVICE)

                y_hat = module(XP, XL)
                error = self.errorFunc(y_hat, y)
                runningLoss.append(error.item())

                if module.training:
                    error.backward()
                    self.parsUpdater.step()
                    self.parsUpdater.zero_grad()

                if self.scoreFuncs and isinstance(y, torch.Tensor):
                    yTrue.append(y.detach().cpu())
                    yPred.append(y_hat.detach().cpu())

        self.records[f"{prefix} Loss"].append(np.mean(runningLoss))

        # Compute metrics
        if self.scoreFuncs:
            yTrue = torch.cat(yTrue).numpy()
            yPred = torch.cat(yPred).numpy()
            for name, scoreFunc in self.scoreFuncs.items():
                try:
                    self.records[f"{prefix} {name}"].append(scoreFunc(yTrue, yPred))
                except Exception as e:
                    print(f"Error computing {name}: {e}")
                    self.records[f"{prefix} {name}"].append(float("NaN"))

def moveTo(obj, device):
    """
    Moves a Python object or its contents to a specified device.

    Args:
        obj: The Python object to move.
        device: The compute device (e.g., "cuda" or "cpu").

    Returns:
        The object moved to the specified device.
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    else:
        return obj
