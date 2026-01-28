"""GPU statistics callback for logging peak memory and utilization to wandb."""

import os

import lightning.pytorch as pl
import torch


class GPUStatsCallback(pl.Callback):
    """Tracks and logs peak GPU memory usage and utilization to wandb.

    This callback monitors GPU memory allocation and utilization during training,
    then logs the peak values as summary metrics to wandb at the end of training.

    Metrics logged:
        - GPU/peak_memory_allocated_GB: Peak memory allocated by PyTorch tensors
        - GPU/peak_memory_reserved_GB: Peak memory reserved by caching allocator
        - GPU/peak_utilization_percent: Peak GPU compute utilization (if pynvml available)
    """

    def __init__(self):
        """Initialize the GPU stats callback."""
        self.peak_utilization = 0.0
        self.pynvml_available = False
        self.nvml_handle = None
        self.gpu_available = torch.cuda.is_available()

    def _init_pynvml(self):
        """Initialize pynvml for GPU utilization tracking."""
        try:
            import pynvml

            pynvml.nvmlInit()

            # Get the actual GPU index from CUDA_VISIBLE_DEVICES
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if cuda_visible is not None and cuda_visible != '':
                # Use the first visible GPU (maps to cuda:0 in PyTorch)
                gpu_indices = [int(idx.strip()) for idx in cuda_visible.split(',')]
                device_index = gpu_indices[0]
            else:
                # Use the current CUDA device
                device_index = torch.cuda.current_device()

            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.pynvml_available = True
        except (ImportError, Exception):
            # pynvml not available or failed to initialize
            self.pynvml_available = False

    def _shutdown_pynvml(self):
        """Shutdown pynvml."""
        if self.pynvml_available:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _sample_gpu_utilization(self):
        """Sample current GPU utilization and update peak if higher."""
        if not self.pynvml_available or self.nvml_handle is None:
            return

        try:
            import pynvml

            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
            current_util = utilization.gpu
            if current_util > self.peak_utilization:
                self.peak_utilization = current_util
        except Exception:
            pass

    def on_train_start(self, trainer, pl_module):
        """Reset GPU memory stats and initialize utilization tracking.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            The Lightning module being trained.
        """
        if not self.gpu_available:
            return

        # Reset peak memory statistics
        torch.cuda.reset_peak_memory_stats()

        # Initialize pynvml for utilization tracking
        self._init_pynvml()
        self.peak_utilization = 0.0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Sample GPU utilization after each batch.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            The Lightning module being trained.
        outputs : Any
            Outputs from the training step.
        batch : Any
            The current batch.
        batch_idx : int
            Index of the current batch.
        """
        if not self.gpu_available:
            return

        # Sample GPU utilization periodically
        self._sample_gpu_utilization()

    def on_train_end(self, trainer, pl_module):
        """Log peak GPU stats to wandb at the end of training.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            The Lightning module being trained.
        """
        if not self.gpu_available:
            return

        # Get peak memory stats
        peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024**3)  # GB

        # Prepare metrics dict
        gpu_metrics = {
            'GPU/peak_memory_allocated_GB': peak_memory_allocated,
            'GPU/peak_memory_reserved_GB': peak_memory_reserved,
        }

        # Add utilization if available
        if self.pynvml_available:
            gpu_metrics['GPU/peak_utilization_percent'] = self.peak_utilization

        # Log to trainer's logger
        if trainer.logger:
            trainer.logger.log_metrics(gpu_metrics)

        # Shutdown pynvml
        self._shutdown_pynvml()
