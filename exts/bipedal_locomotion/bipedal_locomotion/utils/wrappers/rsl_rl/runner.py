"""On-policy runner for the wheeled-legged velocity tasks.

Flattens what the mjlab task package spread over three classes
(``MjlabOnPolicyRunner`` -> ``VelocityOnPolicyRunner`` ->
``WheeledLeggedVelocityOnPolicyRunner``) into one, since only the wheeled-legged
behaviour is needed here.

On top of :class:`rsl_rl.runners.OnPolicyRunner` it:

* drops ``None``-valued optional model config keys before they reach the model
  constructors (``asdict`` emits them, ``MLPModel`` rejects them),
* persists ``common_step_counter`` in the checkpoint so curricula resume where
  they left off rather than restarting their ramps,
* exports ONNX on every save and attaches the deployment metadata contract.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from .exporter import attach_metadata_to_onnx, get_wheeled_legged_metadata
from .vecenv_wrapper import RslRlVecEnvWrapper


class WheeledLeggedVelocityOnPolicyRunner(OnPolicyRunner):
    """Velocity runner with checkpointed env state and ONNX deployment metadata."""

    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        # `dataclasses.asdict` materialises every optional field, including the ones left
        # at None. The model constructors treat "key present" as "feature requested", so
        # the Nones have to be stripped rather than passed through.
        for key in ("actor", "critic"):
            model_cfg = train_cfg.get(key)
            if not isinstance(model_cfg, dict):
                continue
            for opt in ("cnn_cfg", "distribution_cfg"):
                if model_cfg.get(opt) is None:
                    model_cfg.pop(opt, None)
            if model_cfg.get("rnn_type") is None:
                for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                    model_cfg.pop(opt, None)
        super().__init__(env, train_cfg, log_dir, device)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the policy to ONNX via the legacy (non-dynamo) tracer.

        ``torch>=2.9`` defaults ``torch.onnx.export`` to the TorchDynamo path, which
        warns that ``dynamic_axes`` is deprecated and changes the emitted graph. The
        deployment runtime consumes fixed-shape graphs, so pin the legacy path.
        """
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()
        os.makedirs(path, exist_ok=True)
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,
            output_names=onnx_model.output_names,
            dynamic_axes={},
            dynamo=False,
        )

    def export_onnx_with_metadata(self, path: str, filename: str = "policy.onnx") -> str:
        """Export the policy to ``path/filename`` and attach the deployment metadata; returns the file path."""
        self.export_policy_to_onnx(path, filename)
        run_name = "local"
        if getattr(self.logger, "logger_type", None) == "wandb":  # only set once learn() starts logging
            import wandb

            if wandb.run is not None:
                run_name = wandb.run.name
        onnx_path = os.path.join(path, filename)
        metadata = get_wheeled_legged_metadata(self.env.unwrapped, run_name, self.alg.get_policy())
        attach_metadata_to_onnx(onnx_path, metadata)
        return onnx_path

    @staticmethod
    def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
        """Resolve the ONNX export directory and file name from a checkpoint path."""
        export_dir = Path(checkpoint_path).parent
        filename = f"{export_dir.name}.onnx"
        return export_dir, filename, export_dir / filename

    def save(self, path: str, infos=None) -> None:
        env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
        infos = {**(infos or {}), "env_state": env_state}

        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        if self.cfg.get("upload_model"):
            self.logger.save_model(path, self.current_learning_iteration)

        # ONNX export is best-effort: a broken export must not lose a training run.
        policy_dir, filename, onnx_path = self._get_export_paths(path)
        try:
            self.export_onnx_with_metadata(str(policy_dir), filename)
            if getattr(self.logger, "logger_type", None) == "wandb" and self.cfg.get("upload_model"):
                import wandb

                wandb.save(str(onnx_path), base_path=str(policy_dir))
        except Exception as exc:  # noqa: BLE001 - export failure must not abort training
            print(f"[WARN] ONNX export failed (training continues): {exc}")

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Load a checkpoint and restore ``common_step_counter``.

        Restoring the step counter matters because the roughness-gate ramp and the
        fall-over angle curriculum are both scheduled off it; resuming without it would
        silently restart those ramps at zero.
        """
        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        infos = loaded_dict["infos"]
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
        return infos
