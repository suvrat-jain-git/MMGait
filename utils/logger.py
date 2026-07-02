"""
logger.py — Training Logger

Logs per-epoch losses and metrics to:
    1. Console (already handled by trainer)
    2. CSV file — one row per epoch, easy to load into pandas/Excel
    3. JSON file — full history for programmatic access

Usage:
    from utils.logger import TrainingLogger
    logger = TrainingLogger(log_dir='experiments/logs')

    # Each epoch:
    logger.log_epoch(epoch, train_losses, val_losses)

    # After training:
    logger.save()
    df = logger.to_dataframe()
"""

import os
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class TrainingLogger:
    """
    Logs training and validation metrics per epoch.

    Writes to:
        {log_dir}/training_log.csv   — flat CSV, one row per epoch
        {log_dir}/training_log.json  — full history dict
    """

    def __init__(self, log_dir: str = 'experiments/logs'):
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history  = []
        self.csv_path = self.log_dir / 'training_log.csv'
        self.json_path= self.log_dir / 'training_log.json'

        # Timestamp this run
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    def log_epoch(
        self,
        epoch: int,
        train_losses: Dict[str, Any],
        val_losses:   Dict[str, Any],
        lr: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log one epoch.

        Args:
            epoch:        epoch number (1-indexed)
            train_losses: dict from trainer.train_epoch()
            val_losses:   dict from trainer.val_epoch()
            lr:           current learning rate
            extra:        any additional metrics to log
        """
        row = {'epoch': epoch, 'lr': lr or 0.0}

        # Flatten train losses with 'train_' prefix
        for k, v in train_losses.items():
            val = v.item() if hasattr(v, 'item') else float(v)
            row[f'train_{k}'] = round(val, 6)

        # Flatten val losses with 'val_' prefix
        for k, v in val_losses.items():
            val = v.item() if hasattr(v, 'item') else float(v)
            row[f'val_{k}'] = round(val, 6)

        if extra:
            for k, v in extra.items():
                row[k] = v

        self.history.append(row)
        self._append_csv(row)

    def _append_csv(self, row: dict) -> None:
        """Append one row to the CSV file."""
        file_exists = self.csv_path.exists()
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def save(self) -> None:
        """Save full history to JSON."""
        with open(self.json_path, 'w') as f:
            json.dump({
                'run_id':  self.run_id,
                'history': self.history,
            }, f, indent=2)
        print(f"Training log saved to {self.json_path}")

    def to_dataframe(self):
        """Return history as a pandas DataFrame (requires pandas)."""
        if not HAS_PANDAS:
            raise ImportError("pandas not installed. pip install pandas")
        import pandas as pd
        return pd.DataFrame(self.history)

    def print_summary(self) -> None:
        """Print best epoch for each key metric."""
        if not self.history:
            return

        metrics = ['train_total', 'train_identity', 'train_triplet',
                   'val_total', 'val_gender_acc']

        print("\n=== Training Summary ===")
        for metric in metrics:
            if metric not in self.history[0]:
                continue
            values = [(row['epoch'], row[metric]) for row in self.history
                      if metric in row]
            if not values:
                continue
            # Lower is better for losses, higher for accuracy
            if 'acc' in metric:
                best_epoch, best_val = max(values, key=lambda x: x[1])
            else:
                best_epoch, best_val = min(values, key=lambda x: x[1])
            print(f"  {metric:<25} best={best_val:.4f} at epoch {best_epoch}")
