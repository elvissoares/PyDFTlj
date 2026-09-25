"""
Optimization algorithms for equilibrium and minimization problems.

The optimizers operate on a system object that must provide:

    system.state
    system.allowed
    system.compute_force()
    system.compute_energy()
    system.apply_update()

The system may optionally provide:

    system.Ngrideff

Available optimizers:

    Picard
    Anderson
    FIRE
    ABC-FIRE

New algorithms can be added by subclassing Optimizer.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TextIO

import torch


# ================================================================
# Base Optimizer
# ================================================================
class Optimizer(ABC):
    """
    Abstract base class for equilibrium and minimization algorithms.

    Parameters
    ----------
    system
        Object containing the state and the functions required by
        the optimizer.

    atol
        Absolute tolerance used to scale the residual.

    rtol
        Relative tolerance used to scale the residual.

    max_iter
        Maximum number of optimization iterations.

    output_log
        If True, write the complete optimization log to a file.

    log_filename
        Name or path of the optimization log file.
    """

    _registry: dict[str, type["Optimizer"]] = {}

    def __init_subclass__(cls,*, optimizer_name: str | None = None, **kwargs: Any) -> None:
        """
        Automatically register optimizer subclasses.

        Example
        -------
        class DIIS(
            Optimizer,
            optimizer_name="diis",
        ):
            ...
        """

        super().__init_subclass__(**kwargs)

        if optimizer_name is not None:
            Optimizer._registry[optimizer_name.lower()] = cls

    def __init__(self, system, *, atol: float = 1.0e-5, rtol: float = 1.0e-3, max_iter: int = 9999, output_log: bool = False, log_filename: str | Path = "optimizer.log") -> None:
        if atol <= 0.0 and rtol <= 0.0:
            raise ValueError("atol and rtol must be positive non-zero values.")

        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")

        self.system = system

        self.atol = float(atol)
        self.rtol = float(rtol)
        self.max_iter = int(max_iter)

        self.output_log = bool(output_log)
        self.log_filename = Path(log_filename)

        self.error = float("inf")
        self.Niter = 0
        self.converged = False
        self.stop_reason: str | None = None

        self.energy_log: list[float] = []
        self.error_log: list[float] = []

        self._log_file: TextIO | None = None

    # ============================================================
    # Optimizer registry and factory
    # ============================================================
    @classmethod
    def register(cls, name: str, optimizer_class: type["Optimizer"]) -> None:
        """
        Manually register an optimizer class.

        Parameters
        ----------
        name
            Name used by Optimizer.create().

        optimizer_class
            Class derived from Optimizer.
        """

        if not issubclass(optimizer_class, Optimizer):
            raise TypeError(
                "optimizer_class must inherit "
                "from Optimizer."
            )

        cls._registry[name.lower()] = optimizer_class

    @classmethod
    def create(
        cls,
        name: str,
        system,
        **kwargs: Any,
    ) -> "Optimizer":
        """
        Create an optimizer from its registered name.

        Examples
        --------
        optimizer = Optimizer.create(
            "anderson",
            system,
            alpha=0.1,
            history_size=8,
        )
        """

        optimizer_name = name.lower()

        if optimizer_name not in cls._registry:
            available = ", ".join(
                sorted(cls._registry)
            )

            raise ValueError(
                f"Unknown optimizer '{name}'. "
                f"Available optimizers: {available}"
            )

        optimizer_class = cls._registry[
            optimizer_name
        ]

        return optimizer_class(
            system,
            **kwargs,
        )

    # ============================================================
    # Parameter handling
    # ============================================================
    def set_solver_parameters(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Change existing optimizer parameters.

        Unknown parameter names raise an exception. This prevents
        spelling errors from silently creating new attributes.
        """

        unknown_parameters = [
            name
            for name in kwargs
            if not hasattr(self, name)
        ]

        if unknown_parameters:
            names = ", ".join(
                unknown_parameters
            )

            raise AttributeError(
                f"Unknown optimizer parameters: "
                f"{names}"
            )

        for name, value in kwargs.items():
            setattr(
                self,
                name,
                value,
            )

    # ============================================================
    # State manipulation
    # ============================================================
    def _get_active_state(
        self,
    ) -> torch.Tensor:
        """
        Return the state variables allowed to change.
        """

        return self.system.state[
            self.system.allowed
        ]

    def _set_active_state(
        self,
        value: torch.Tensor,
    ) -> None:
        """
        Replace the active state variables.
        """

        with torch.no_grad():
            self.system.state[
                self.system.allowed
            ] = value

    def _add_to_active_state(
        self,
        increment: torch.Tensor,
    ) -> None:
        """
        Add an increment to the active state variables.
        """

        with torch.no_grad():
            active_state = self.system.state[
                self.system.allowed
            ]

            self.system.state[
                self.system.allowed
            ] = active_state + increment

    # ============================================================
    # Residual and convergence
    # ============================================================
    def compute_error(
        self,
        force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the scaled root-mean-square residual.

        Each residual component is scaled according to

            scale_i = atol + rtol * abs(state_i)

        and the total error is

            error = ||force / scale|| / sqrt(N)

        The convergence criterion is

            error < 1.
        """

        if force is None:
            force = self.system.compute_force()

        active_force = force[self.system.allowed]

        active_state = self._get_active_state()

        scale = self.atol + self.rtol* torch.abs(active_state)

        scaled_force = active_force / scale

        if hasattr(self.system,"Ngrideff",):
            normalization_size = float(self.system.Ngrideff)
        else:
            normalization_size = float(scaled_force.numel())

        if normalization_size <= 0.0:
            raise ValueError("The number of active degrees of freedom must be positive.")

        error = torch.linalg.vector_norm(scaled_force.reshape(-1))

        error = (error/ math.sqrt(normalization_size))

        return error

    # ============================================================
    # Logging
    # ============================================================
    def _open_log(
        self,
    ) -> None:
        """
        Open the optimization log.

        The log file is overwritten at the beginning of each run.
        """

        if not self.output_log:
            return

        parent = self.log_filename.parent

        parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._log_file = (
            self.log_filename.open(
                mode="w",
                encoding="utf-8",
            )
        )

    def _close_log(
        self,
    ) -> None:
        """
        Close the optimization log.
        """

        if self._log_file is None:
            return

        self._log_file.close()
        self._log_file = None

    def _write_message(
        self,
        message: str,
        *,
        terminal: bool = False,
    ) -> None:
        """
        Write a message to the log file and optionally to stdout.
        """

        if terminal:
            print(message)

        if self._log_file is not None:
            self._log_file.write(
                message + "\n"
            )

            self._log_file.flush()

    def _write_log_header(
        self,
        *,
        terminal: bool,
    ) -> None:
        """
        Write general optimizer information and the table header.
        """

        optimizer_name = type(self).__name__

        information = [
            f"Optimizer: {optimizer_name}",
            (
                f"Maximum iterations: "
                f"{self.max_iter}"
            ),
            (
                f"Absolute tolerance: "
                f"{self.atol:.8e}"
            ),
            (
                f"Relative tolerance: "
                f"{self.rtol:.8e}"
            ),
            "",
        ]

        for message in information:
            self._write_message(
                message,
                terminal=terminal,
            )

        header = (
            f"{'iter':>8s} | "
            f"{'energy':>18s} | "
            f"{'error':>14s} | "
            f"parameters"
        )

        separator = "-" * 96

        self._write_message(
            header,
            terminal=terminal,
        )

        self._write_message(
            separator,
            terminal=terminal,
        )

    def _status_string(
        self,
    ) -> str:
        """
        Return method-specific information for logging.
        """

        return ""

    @staticmethod
    def _to_float(value: torch.Tensor | float) -> float:
        """
        Convert a scalar tensor or Python number to float.
        """

        if isinstance(value,torch.Tensor,):
            return float(value.detach().cpu().item())

        return float(value)

    # ============================================================
    # Optimizer control
    # ============================================================
    @abstractmethod
    def step(self) -> None:
        """
        Perform one optimization iteration.
        """

    def _reset_method(self,) -> None:
        """
        Reset method-specific variables.

        Derived classes may override this method.
        """

    def reset(self) -> None:
        """
        Reset counters, logs and method-specific variables.

        The state stored in system.state is not modified.
        """

        self.error = float("inf")
        self.Niter = 0
        self.converged = False
        self.stop_reason = None

        self.energy_log.clear()
        self.error_log.clear()

        self._reset_method()

    def run(self,*,logoutput: bool = False,log_interval: int = 50) -> tuple[float, int]:
        """
        Run the optimizer until convergence or max_iter.

        Parameters
        ----------
        logoutput
            If True, print optimization information to the terminal.

        log_interval
            Number of iterations between terminal messages.

            When output_log=True, every iteration is written to
            optimizer.log, independently of log_interval.

        Returns
        -------
        error
            Final normalized residual.

        Niter
            Number of completed iterations.
        """

        if log_interval < 1:
            raise ValueError("log_interval must be at least 1.")

        self.error = float("inf")
        self.Niter = 0
        self.converged = False
        self.stop_reason = None

        self.energy_log.clear()
        self.error_log.clear()

        self._open_log()

        try:
            self._write_log_header(terminal=logoutput)

            for iteration in range(1,self.max_iter + 1):
                self.step()

                energy = self.system.compute_energy()

                force = self.system.compute_force()

                error_tensor = self.compute_error(force)

                energy_value = self._to_float(energy)

                error_value = self._to_float(error_tensor)  

                self.Niter = iteration
                self.error = error_value

                self.energy_log.append(energy_value)

                self.error_log.append(error_value)

                status = (
                    f"{iteration:8d} | "
                    f"{energy_value:18.10e} | "
                    f"{error_value:14.6e}"
                    f"{self._status_string()}"
                )

                print_iteration = (
                    logoutput
                    and (
                        iteration <= 5
                        or iteration
                        % log_interval == 0
                        or error_value < 1.0
                        or not math.isfinite(
                            error_value
                        )
                        or not math.isfinite(
                            energy_value
                        )
                        or iteration
                        == self.max_iter
                    )
                )

                # Every iteration is written to the file when
                # output_log=True.
                self._write_message(
                    status,
                    terminal=print_iteration,
                )

                finite_error = math.isfinite(
                    error_value
                )

                finite_energy = math.isfinite(
                    energy_value
                )

                if (
                    not finite_error
                    or not finite_energy
                ):
                    self.stop_reason = (
                        "Non-finite energy or residual"
                    )

                    message = (
                        "Optimization stopped at "
                        f"iteration {iteration}: "
                        f"energy={energy_value}, "
                        f"error={error_value}."
                    )

                    self._write_message(
                        message,
                        terminal=logoutput,
                    )

                    break

                if error_value < 1.0:
                    self.converged = True

                    self.stop_reason = (
                        "Convergence criterion reached"
                    )

                    message = (
                        "Converged at iteration "
                        f"{iteration}: "
                        f"error={error_value:.6e}."
                    )

                    self._write_message(
                        message,
                        terminal=logoutput,
                    )

                    break

            else:
                self.stop_reason = (
                    "Maximum number of iterations reached"
                )

                message = (
                    "Maximum number of iterations "
                    f"reached: {self.max_iter}."
                )

                self._write_message(
                    message,
                    terminal=logoutput,
                )

            self._write_message(
                "",
                terminal=False,
            )

            self._write_message(
                (
                    f"Final energy: "
                    f"{self.energy_log[-1]:.10e}"
                    if self.energy_log
                    else "Final energy: unavailable"
                ),
                terminal=False,
            )

            self._write_message(
                (
                    f"Final error: "
                    f"{self.error:.6e}"
                ),
                terminal=False,
            )

            self._write_message(
                (
                    f"Completed iterations: "
                    f"{self.Niter}"
                ),
                terminal=False,
            )

            self._write_message(
                (
                    f"Converged: "
                    f"{self.converged}"
                ),
                terminal=False,
            )

            self._write_message(
                (
                    f"Stop reason: "
                    f"{self.stop_reason}"
                ),
                terminal=False,
            )

        except Exception as exception:
            self.stop_reason = (
                f"Exception: {type(exception).__name__}"
            )

            self._write_message(
                (
                    "Optimization interrupted by "
                    f"{type(exception).__name__}: "
                    f"{exception}"
                ),
                terminal=logoutput,
            )

            raise

        finally:
            self._close_log()

        return self.error, self.Niter


# ================================================================
# Picard Optimizer
# ================================================================
class Picard(
    Optimizer,
    optimizer_name="picard",
):
    """
    Picard fixed-point iteration.

    The update is

        state_{k+1}
            = state_k + alpha * force(state_k)

    The value returned by system.compute_force() is interpreted as
    the fixed-point residual.
    """

    def __init__(
        self,
        system,
        *,
        alpha: float = 0.15,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            system,
            **kwargs,
        )

        if alpha <= 0.0:
            raise ValueError(
                "alpha must be positive."
            )

        self.alpha = float(alpha)

    def step(
        self,
    ) -> None:
        force = self.system.compute_force()

        active_force = force[
            self.system.allowed
        ]

        increment = (
            self.alpha * active_force
        )

        self._add_to_active_state(
            increment
        )

        self.system.apply_update()

    def _status_string(
        self,
    ) -> str:
        return (
            f" | alpha={self.alpha:.6e}"
        )


# ================================================================
# Anderson Optimizer
# ================================================================
class Anderson(Optimizer,optimizer_name="anderson"):
    """
    Anderson acceleration for fixed-point iterations.

    The unaccelerated Picard iteration is assumed to be

        state_{k+1}
            = state_k + alpha * force(state_k)

    Anderson acceleration uses previous states and residuals to
    construct a corrected update.

    Parameters
    ----------
    alpha
        Picard mixing parameter.

    history_size
        Maximum number of Anderson difference vectors.

    regularization
        Relative regularization added to the normal matrix.

    start_iteration
        Number of completed Picard iterations before Anderson
        acceleration is activated.
    """

    def __init__(self,system,*,alpha: float = 0.15, history_size: int = 6, regularization: float = 1.0e-8, start_iteration: int = 1,**kwargs: Any,) -> None:
        super().__init__(system,**kwargs)

        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")

        if history_size < 1:
            raise ValueError("history_size must be at least 1.")

        if regularization < 0.0:
            raise ValueError("regularization must be non-negative.")

        if start_iteration < 0:
            raise ValueError("start_iteration must be non-negative.")

        self.alpha = float(alpha)
        self.history_size = int(history_size)

        self.regularization = float(regularization)

        self.start_iteration = int(start_iteration)

        self._x_history: list[torch.Tensor] = []

        self._force_history: list[torch.Tensor] = []

    def _reset_method(self) -> None: 
        self._x_history.clear()
        self._force_history.clear()

    def _append_history(self, state: torch.Tensor, force: torch.Tensor) -> None:
        """
        Add a state and its residual to the Anderson history.
        """

        self._x_history.append(state.detach().clone())

        self._force_history.append(force.detach().clone())

        # m Anderson differences require m + 1 states.
        maximum_history = (self.history_size + 1)

        if (len(self._x_history) > maximum_history):
            self._x_history.pop(0)
            self._force_history.pop(0)

    def _compute_anderson_increment(self, force: torch.Tensor) -> torch.Tensor:
        """
        Compute the Anderson correction using a regularized
        least-squares problem.

        This implementation avoids explicitly forming the normal
        equations delta_force.T @ delta_force.
        """

        number_of_differences = len(self._x_history) - 1

        delta_state = torch.stack(
            [
                self._x_history[index + 1]
                - self._x_history[index]
                for index in range(
                    number_of_differences
                )
            ],
            dim=1,
        )

        delta_force = torch.stack(
            [
                self._force_history[index + 1]
                - self._force_history[index]
                for index in range(
                    number_of_differences
                )
            ],
            dim=1,
        )

        identity = torch.eye(number_of_differences, dtype=force.dtype, device=force.device)

        # Scale the regularization according to the magnitude
        # of the residual-difference matrix.
        matrix_scale = torch.mean(delta_force.square())

        matrix_scale = torch.clamp(matrix_scale, min=torch.finfo(force.dtype).eps)

        lambda_value = self.regularization * matrix_scale

        regularization_matrix = torch.sqrt(lambda_value) * identity

        augmented_matrix = torch.cat(
            [
                delta_force,
                regularization_matrix,
            ],
            dim=0,
        )

        augmented_rhs = torch.cat(
            [
                force,
                torch.zeros(
                    number_of_differences,
                    dtype=force.dtype,
                    device=force.device,
                ),
            ],
            dim=0,
        )

        coefficients = torch.linalg.lstsq(
            augmented_matrix,
            augmented_rhs.unsqueeze(1),
        ).solution[:, 0]

        increment = (
            self.alpha * force
            - (
                delta_state
                + self.alpha * delta_force
            )
            @ coefficients
        )

        return increment

    def step(self) -> None:
        active_state = self._get_active_state().detach().clone()

        energy = self.system.compute_energy()
        force = self.system.compute_force()

        active_force = force[self.system.allowed].detach().clone()

        original_shape = active_state.shape

        state_flat = active_state.reshape(-1)
        force_flat = active_force.reshape(-1)

        self._append_history(state_flat, force_flat)

        # Check if we have enough history to perform Anderson acceleration. If not, fall back to Picard iteration.
        if (len(self._x_history) >= 2 and self.Niter >= self.start_iteration):
            increment_flat = self._compute_anderson_increment(force_flat)

        else:
            increment_flat = self.alpha * force_flat

        updated_state = (state_flat + increment_flat).reshape(original_shape)

        self._set_active_state(updated_state)

        self.system.apply_update()

        # # store the current state if the energy is lower than the previous state
        # if self.system.compute_energy() < energy:
        #     self.system.best_state = self.system.state.detach().clone()

    def _status_string(self) -> str:
        number_of_differences = max(0,len(self._x_history) - 1)

        return (
            f" | alpha={self.alpha:.6e}"
            f" | history={number_of_differences:d}"
        )


# ================================================================
# FIRE Optimizer
# ================================================================
class FIRE(
    Optimizer,
    optimizer_name="fire",
):
    """
    Fast Inertial Relaxation Engine.

    The value returned by system.compute_force() must point in the
    direction of decreasing energy.

    Parameters
    ----------
    mode
        Either "fire" or "abc-fire".

    alpha
        Initial FIRE mixing parameter.

    dt
        Initial integration timestep.

    dt_max
        Maximum allowed timestep. If None, dt_max = 10 * dt.

    dt_min
        Minimum allowed timestep. If None, dt_min = 0.02 * dt.

    Ndelay
        Number of positive-power iterations required before
        increasing the timestep.

    Nnegmax
        Maximum number of consecutive negative-power iterations.

    finc
        Timestep increase factor.

    fdec
        Timestep decrease factor.

    fa
        Alpha decrease factor.
    """

    def __init__(
        self,
        system,
        *,
        mode: str = "fire",
        alpha: float = 0.15,
        dt: float = 0.02,
        dt_max: float | None = None,
        dt_min: float | None = None,
        Ndelay: int = 5,
        Nnegmax: int = 2000,
        finc: float = 1.1,
        fdec: float = 0.5,
        fa: float = 0.99,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            system,
            **kwargs,
        )

        mode = mode.lower()

        if mode not in {
            "fire",
            "abc-fire",
        }:
            raise ValueError(
                "mode must be 'fire' or 'abc-fire'."
            )

        if alpha <= 0.0:
            raise ValueError(
                "alpha must be positive."
            )

        if dt <= 0.0:
            raise ValueError(
                "dt must be positive."
            )

        if Ndelay < 0:
            raise ValueError(
                "Ndelay must be non-negative."
            )

        if Nnegmax < 1:
            raise ValueError(
                "Nnegmax must be at least 1."
            )

        if finc <= 1.0:
            raise ValueError(
                "finc must be greater than 1."
            )

        if not 0.0 < fdec < 1.0:
            raise ValueError(
                "fdec must be between 0 and 1."
            )

        if not 0.0 < fa <= 1.0:
            raise ValueError(
                "fa must be between 0 and 1."
            )

        self.mode = mode

        self.alpha0 = float(alpha)
        self.alpha = float(alpha)

        self.dt0 = float(dt)
        self.dt = float(dt)

        if dt_max is None:
            self.dt_max = (
                10.0 * self.dt0
            )
        else:
            self.dt_max = float(dt_max)

        if dt_min is None:
            self.dt_min = (
                0.02 * self.dt0
            )
        else:
            self.dt_min = float(dt_min)

        if self.dt_min <= 0.0:
            raise ValueError(
                "dt_min must be positive."
            )

        if self.dt_max < self.dt_min:
            raise ValueError(
                "dt_max must be greater than "
                "or equal to dt_min."
            )

        self.Ndelay = int(Ndelay)
        self.Nnegmax = int(Nnegmax)

        self.finc = float(finc)
        self.fdec = float(fdec)
        self.fa = float(fa)

        self.Npos = 0
        self.Nneg = 0

        self.V = torch.zeros_like(
            self.system.state
        )

    def _reset_method(
        self,
    ) -> None:
        self.alpha = self.alpha0
        self.dt = self.dt0

        self.Npos = 0
        self.Nneg = 0

        self.V = torch.zeros_like(
            self.system.state
        )

    def step(
        self,
    ) -> None:
        force = self.system.compute_force()

        allowed = self.system.allowed

        active_force = force[allowed]

        with torch.no_grad():
            active_velocity = (
                self.V[allowed]
                + self.dt * active_force
            )

            power = torch.sum(
                active_force
                * active_velocity
            )

            power_value = float(
                power.detach().cpu().item()
            )

            if power_value > 0.0:
                self.Npos += 1
                self.Nneg = 0

                if self.Npos > self.Ndelay:
                    self.dt = min(
                        self.dt * self.finc,
                        self.dt_max,
                    )

                    self.alpha *= self.fa

            else:
                self.Npos = 0
                self.Nneg += 1

                if self.Nneg > self.Nnegmax:
                    raise RuntimeError(
                        "FIRE reached the maximum "
                        "number of consecutive "
                        "negative-power iterations."
                    )

                self.dt = max(
                    self.dt * self.fdec,
                    self.dt_min,
                )

                self.alpha = self.alpha0

                active_velocity = (
                    torch.zeros_like(
                        active_velocity
                    )
                )

            velocity_norm = (
                torch.linalg.vector_norm(
                    active_velocity.reshape(-1)
                )
            )

            force_norm = (
                torch.linalg.vector_norm(
                    active_force.reshape(-1)
                )
            )

            velocity_norm_value = float(
                velocity_norm.detach().cpu().item()
            )

            force_norm_value = float(
                force_norm.detach().cpu().item()
            )

            if (
                velocity_norm_value > 0.0
                and force_norm_value > 0.0
            ):
                active_velocity = (
                    (1.0 - self.alpha)
                    * active_velocity
                    + self.alpha
                    * active_force
                    * velocity_norm
                    / force_norm
                )

                if (
                    self.mode == "abc-fire"
                    and power_value > 0.0
                    and self.Npos > 0
                ):
                    denominator = (
                        1.0
                        - (
                            1.0 - self.alpha
                        ) ** float(self.Npos)
                    )

                    if denominator > 1.0e-14:
                        active_velocity = (
                            active_velocity
                            / denominator
                        )

            self.V[allowed] = (
                active_velocity
            )

            active_state = (
                self.system.state[allowed]
            )

            updated_state = (
                active_state
                + self.dt * active_velocity
            )

            self.system.state[allowed] = updated_state

        self.system.apply_update()

    def _status_string(
        self,
    ) -> str:
        return (
            f" | mode={self.mode}"
            f" | alpha={self.alpha:.6e}"
            f" | dt={self.dt:.6e}"
            f" | Npos={self.Npos:d}"
            f" | Nneg={self.Nneg:d}"
        )


# ================================================================
# Public interface
# ================================================================
__all__ = [
    "Optimizer",
    "Picard",
    "Anderson",
    "FIRE",
]