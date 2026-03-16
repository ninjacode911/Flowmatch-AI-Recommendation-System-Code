"""
Experiment Tracker -- MLflow integration for tracking training runs.

Tracks:
  - Hyperparameters (learning rate, batch size, model architecture)
  - Training metrics per epoch (loss, NDCG, recall)
  - Model artifacts (checkpoints, embeddings)
  - Dataset metadata (size, split ratios)

Works in two modes:
  1. MLflow mode: logs to a real MLflow tracking server
  2. Local mode: logs to console and local files (for dev without MLflow)
"""

import json
import time
from pathlib import Path

import structlog

logger = structlog.get_logger()


class ExperimentTracker:
    """
    Unified experiment tracking interface.

    Wraps MLflow when available, falls back to local JSON logging.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "http://localhost:5000",
        local_mode: bool = True,
    ) -> None:
        self.experiment_name = experiment_name
        self.local_mode = local_mode
        self._mlflow = None
        self._run = None
        self._local_log: dict = {
            "experiment": experiment_name,
            "params": {},
            "metrics": {},
            "artifacts": [],
            "tags": {},
        }
        self._start_time = time.time()

        if not local_mode:
            self._init_mlflow(tracking_uri)
        else:
            logger.info("tracker.local_mode", experiment=experiment_name)

    def _init_mlflow(self, tracking_uri: str) -> None:
        """Initialize MLflow connection."""
        try:
            import mlflow

            self._mlflow = mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info("tracker.mlflow_connected", uri=tracking_uri, experiment=self.experiment_name)
        except ImportError:
            logger.warning("tracker.mlflow_not_installed", msg="Falling back to local mode")
            self.local_mode = True
        except Exception as e:
            logger.error("tracker.mlflow_failed", error=str(e))
            self.local_mode = True

    def start_run(self, run_name: str = "") -> None:
        """Start a new experiment run."""
        self._start_time = time.time()
        self._local_log["run_name"] = run_name
        self._local_log["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        if not self.local_mode and self._mlflow:
            self._run = self._mlflow.start_run(run_name=run_name)
            logger.info("tracker.run_started", run_name=run_name)

    def log_params(self, params: dict) -> None:
        """Log hyperparameters."""
        self._local_log["params"].update(params)

        if not self.local_mode and self._mlflow:
            self._mlflow.log_params(params)

        logger.info("tracker.params_logged", count=len(params))

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        """Log a single metric value."""
        if key not in self._local_log["metrics"]:
            self._local_log["metrics"][key] = []
        self._local_log["metrics"][key].append({"step": step, "value": value})

        if not self.local_mode and self._mlflow:
            self._mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """Log multiple metrics at once."""
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    def log_artifact(self, file_path: str) -> None:
        """Log a file as an artifact."""
        self._local_log["artifacts"].append(file_path)

        if not self.local_mode and self._mlflow:
            self._mlflow.log_artifact(file_path)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run."""
        self._local_log["tags"][key] = value

        if not self.local_mode and self._mlflow:
            self._mlflow.set_tag(key, value)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run and save local log."""
        elapsed = time.time() - self._start_time
        self._local_log["duration_seconds"] = round(elapsed, 1)
        self._local_log["status"] = status

        if not self.local_mode and self._mlflow:
            self._mlflow.end_run(status=status)
            logger.info("tracker.run_ended", status=status, duration=elapsed)

        # Always save local log as backup
        self._save_local_log()

    def _save_local_log(self) -> None:
        """Save experiment log to a local JSON file."""
        log_dir = Path(__file__).resolve().parents[3] / "models" / "experiments"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = self.experiment_name.replace(" ", "_").replace("/", "_")
        log_path = log_dir / f"{safe_name}_{timestamp}.json"

        with open(log_path, "w") as f:
            json.dump(self._local_log, f, indent=2, default=str)

        logger.info("tracker.local_log_saved", path=str(log_path))
