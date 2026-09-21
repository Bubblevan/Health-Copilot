"""Process containment contracts and Bubblewrap backends for M9."""

from __future__ import annotations

import platform
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol


class SandboxNetworkPolicy(StrEnum):
    DENY_ALL = "deny_all"
    ALLOWLIST = "allowlist"


@dataclass(frozen=True)
class SandboxFilesystemPolicy:
    read_roots: tuple[Path, ...] = ()
    write_roots: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        for name in ("read_roots", "write_roots"):
            values = tuple(Path(item).resolve() for item in getattr(self, name))
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class SandboxPolicy:
    filesystem: SandboxFilesystemPolicy
    network: SandboxNetworkPolicy = SandboxNetworkPolicy.DENY_ALL
    network_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "network", SandboxNetworkPolicy(self.network))
        object.__setattr__(self, "network_origins", tuple(self.network_origins))
        if self.network == SandboxNetworkPolicy.ALLOWLIST and not self.network_origins:
            raise ValueError("network allowlist must not be empty")


@dataclass(frozen=True)
class SandboxProfile:
    profile_id: str
    policy: SandboxPolicy
    backend_id: str = "bubblewrap-sandbox-v1"
    requires_real_enforcement: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("sandbox profile_id must be non-empty")
        if not isinstance(self.policy, SandboxPolicy):
            raise TypeError("sandbox profile policy must be SandboxPolicy")


class SandboxFailureKind(StrEnum):
    UNAVAILABLE = "sandbox_unavailable"
    FILESYSTEM_DENIED = "filesystem_denied"
    NETWORK_DENIED = "network_denied"
    PROCESS_ERROR = "sandbox_process_error"


class SandboxFailure(RuntimeError):
    def __init__(self, kind: SandboxFailureKind, message: str = "sandbox execution failed") -> None:
        self.kind = SandboxFailureKind(kind)
        super().__init__(message)


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    contained: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class SandboxBackend(Protocol):
    backend_id: str
    contained: bool

    def is_available(self) -> bool:
        ...

    def run(
        self,
        command: Sequence[str],
        *,
        profile: SandboxProfile,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        ...

    def wrap_command(
        self,
        command: Sequence[str],
        *,
        profile: SandboxProfile,
        cwd: Path | None = None,
    ) -> list[str]:
        """Return a command that is subject to this backend's boundary."""
        ...


def _path_args(paths: Sequence[Path]) -> list[str]:
    result: list[str] = []
    for path in paths:
        result.extend(("--dir", str(path)))
    return result


def _parent_dirs(path: Path) -> list[Path]:
    values: list[Path] = []
    current = path.parent
    while current != current.parent:
        values.append(current)
        current = current.parent
    return list(reversed(values))


class BubblewrapSandboxBackend:
    """Real Linux Bubblewrap execution; no Python monkeypatching is involved."""

    backend_id = "bubblewrap-sandbox-v1"
    contained = True

    def __init__(self, executable: str = "bwrap") -> None:
        self.executable = executable

    def is_available(self) -> bool:
        return shutil.which(self.executable) is not None

    def _command(
        self,
        command: Sequence[str],
        profile: SandboxProfile,
        cwd: Path | None,
        *,
        check_paths: bool = True,
    ) -> list[str]:
        if not self.is_available():
            raise SandboxFailure(SandboxFailureKind.UNAVAILABLE, "bubblewrap is unavailable")
        policy = profile.policy
        args = [self.executable, "--die-with-parent", "--new-session"]
        for system_path in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
            path = Path(system_path)
            if not check_paths or path.exists():
                args.extend(("--ro-bind", system_path, system_path))
        args.extend(("--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"))
        if policy.network == SandboxNetworkPolicy.DENY_ALL:
            args.append("--unshare-net")
        elif policy.network == SandboxNetworkPolicy.ALLOWLIST:
            raise SandboxFailure(
                SandboxFailureKind.NETWORK_DENIED,
                "Bubblewrap network allowlist enforcement is not implemented",
            )
        mount_roots = tuple(dict.fromkeys(policy.filesystem.read_roots + policy.filesystem.write_roots))
        for root in mount_roots:
            if check_paths and not root.exists():
                raise SandboxFailure(SandboxFailureKind.FILESYSTEM_DENIED, f"sandbox root missing: {root}")
            args.extend(_path_args(_parent_dirs(root)))
            if root in policy.filesystem.write_roots:
                args.extend(("--bind", str(root), str(root)))
            else:
                args.extend(("--ro-bind", str(root), str(root)))
        working_dir = cwd or (policy.filesystem.write_roots[0] if policy.filesystem.write_roots else Path("/tmp"))
        if check_paths and not working_dir.exists():
            raise SandboxFailure(SandboxFailureKind.FILESYSTEM_DENIED, f"sandbox cwd missing: {working_dir}")
        args.extend(("--chdir", str(working_dir), "--"))
        args.extend(str(item) for item in command)
        return args

    def wrap_command(
        self,
        command: Sequence[str],
        *,
        profile: SandboxProfile,
        cwd: Path | None = None,
    ) -> list[str]:
        return self._command(command, profile, cwd)

    def run(
        self,
        command: Sequence[str],
        *,
        profile: SandboxProfile,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        args = self.wrap_command(command, profile=profile, cwd=cwd)
        try:
            completed = subprocess.run(
                args,
                cwd=None,
                env=dict(env) if env is not None else None,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxFailure(SandboxFailureKind.PROCESS_ERROR, "sandbox process timed out") from exc
        return SandboxResult(completed.returncode, completed.stdout, completed.stderr, contained=True)


class WslBubblewrapSandboxBackend(BubblewrapSandboxBackend):
    """Windows development backend that delegates real containment to WSL2."""

    backend_id = "bubblewrap-sandbox-wsl-v1"

    def __init__(self, distribution: str = "Ubuntu-24.04") -> None:
        super().__init__("/usr/bin/bwrap")
        self.distribution = distribution

    def is_available(self) -> bool:
        if platform.system() != "Windows" or shutil.which("wsl.exe") is None:
            return False
        try:
            result = subprocess.run(
                ["wsl.exe", "-d", self.distribution, "--", "command", "-v", "bwrap"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except OSError:
            return False
        return result.returncode == 0 and bool(result.stdout.strip())

    @staticmethod
    def _wsl_path(value: str | Path) -> str:
        raw = str(value)
        if not (len(raw) >= 2 and raw[1] == ":"):
            return raw
        windows = PureWindowsPath(raw)
        return "/mnt/" + windows.drive[0].lower() + "/" + "/".join(windows.parts[1:])

    def _command(self, command: Sequence[str], profile: SandboxProfile, cwd: Path | None) -> list[str]:
        if not self.is_available():
            raise SandboxFailure(SandboxFailureKind.UNAVAILABLE, "WSL Bubblewrap is unavailable")
        translated = [self._wsl_path(item) for item in command]
        policy = SandboxPolicy(
            filesystem=SandboxFilesystemPolicy(
                read_roots=tuple(Path(self._wsl_path(path)) for path in profile.policy.filesystem.read_roots),
                write_roots=tuple(Path(self._wsl_path(path)) for path in profile.policy.filesystem.write_roots),
            ),
            network=profile.policy.network,
            network_origins=profile.policy.network_origins,
        )
        translated_fs = object.__new__(SandboxFilesystemPolicy)
        object.__setattr__(
            translated_fs,
            "read_roots",
            tuple(PurePosixPath(self._wsl_path(path)) for path in profile.policy.filesystem.read_roots),
        )
        object.__setattr__(
            translated_fs,
            "write_roots",
            tuple(PurePosixPath(self._wsl_path(path)) for path in profile.policy.filesystem.write_roots),
        )
        policy = SandboxPolicy(
            filesystem=translated_fs,
            network=profile.policy.network,
            network_origins=profile.policy.network_origins,
        )
        translated_profile = SandboxProfile(profile.profile_id, policy, profile.backend_id, profile.requires_real_enforcement)
        translated_cwd = PurePosixPath(self._wsl_path(cwd)) if cwd else None
        args = super()._command(translated, translated_profile, translated_cwd, check_paths=False)
        return ["wsl.exe", "-d", self.distribution, "--", *args]


class NoSandboxDevBackend:
    """Explicit trusted-development escape hatch; it is never a containment claim."""

    backend_id = "no-sandbox-dev-v1"
    contained = False

    def is_available(self) -> bool:
        return True

    def wrap_command(
        self,
        command: Sequence[str],
        *,
        profile: SandboxProfile,
        cwd: Path | None = None,
    ) -> list[str]:
        return list(command)

    def run(
        self,
        command: Sequence[str],
        *,
        profile: SandboxProfile,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return SandboxResult(completed.returncode, completed.stdout, completed.stderr, contained=False)


def ensure_sandbox_backend(profile: SandboxProfile, backend: SandboxBackend) -> None:
    if profile.requires_real_enforcement and not backend.is_available():
        raise SandboxFailure(SandboxFailureKind.UNAVAILABLE, f"required backend unavailable: {backend.backend_id}")
    if profile.requires_real_enforcement and not backend.contained:
        raise SandboxFailure(SandboxFailureKind.UNAVAILABLE, f"backend is not containing: {backend.backend_id}")


__all__ = [
    "BubblewrapSandboxBackend",
    "NoSandboxDevBackend",
    "SandboxBackend",
    "SandboxFailure",
    "SandboxFailureKind",
    "SandboxFilesystemPolicy",
    "SandboxNetworkPolicy",
    "SandboxPolicy",
    "SandboxProfile",
    "SandboxResult",
    "WslBubblewrapSandboxBackend",
    "ensure_sandbox_backend",
]
