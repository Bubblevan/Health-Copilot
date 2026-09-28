"""Typed component identity and deterministic runtime provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Any


class ComponentKind(StrEnum):
    """Runtime subsystems that M6 makes explicitly replaceable."""

    PROVIDER = "provider"
    RETRIEVER = "retriever"
    POLICY = "policy"
    VERIFIER = "verifier"
    TOOL = "tool"
    TRACE = "trace"
    ORCHESTRATION = "orchestration"
    MCP_CLIENT = "mcp_client"
    PERMISSION = "permission"
    SANDBOX = "sandbox"
    SESSION_STORE = "session_store"
    CONTEXT_MANAGER = "context_manager"
    MEMORY_STORE = "memory_store"
    MEMORY_POLICY = "memory_policy"


@dataclass(frozen=True)
class LearnedArtifactIdentity:
    """Provenance for a learned local artifact, never a security credential."""

    provider: str
    family: str
    model_id: str
    revision: str | None
    local_files_only: bool = True
    dimension: int | None = None

    def __post_init__(self) -> None:
        for name in ("provider", "family", "model_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"learned artifact {name} must be non-empty")
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ValueError("learned artifact revision must be non-empty or null")
        if not isinstance(self.local_files_only, bool):
            raise TypeError("local_files_only must be boolean")
        if self.dimension is not None and self.dimension <= 0:
            raise ValueError("learned artifact dimension must be positive or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "family": self.family,
            "model_id": self.model_id,
            "revision": self.revision,
            "local_files_only": self.local_files_only,
            "dimension": self.dimension,
        }


@dataclass(frozen=True)
class ComponentIdentity:
    """Stable identity for one constructed runtime component."""

    kind: ComponentKind
    component_id: str
    implementation: str
    version: str
    artifact_revision: str | None = None
    config_hash: str | None = None
    learned_artifacts: tuple[LearnedArtifactIdentity, ...] = ()

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, ComponentKind) else ComponentKind(self.kind)
        object.__setattr__(self, "kind", kind)
        for name in ("component_id", "implementation", "version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"component {name} must be non-empty")
        if self.artifact_revision is not None and not isinstance(self.artifact_revision, str):
            raise TypeError("artifact_revision must be a string or null")
        if self.config_hash is not None and (
            not isinstance(self.config_hash, str) or len(self.config_hash) != 64
        ):
            raise ValueError("config_hash must be a SHA-256 hex digest or null")
        if not all(isinstance(item, LearnedArtifactIdentity) for item in self.learned_artifacts):
            raise TypeError("learned_artifacts must contain LearnedArtifactIdentity values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "component_id": self.component_id,
            "implementation": self.implementation,
            "version": self.version,
            "artifact_revision": self.artifact_revision,
            "config_hash": self.config_hash,
            "learned_artifacts": [item.to_dict() for item in self.learned_artifacts],
        }

    @property
    def identity_key(self) -> tuple[str, str]:
        return self.kind.value, self.component_id


@dataclass(frozen=True)
class CapabilityContractRef:
    """Immutable manifest reference to a canonical worker capability contract."""

    capability_id: str
    contract_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.capability_id, str) or not self.capability_id.strip():
            raise ValueError("capability_id must be non-empty")
        if (
            not isinstance(self.contract_hash, str)
            or len(self.contract_hash) != 64
            or any(char not in "0123456789abcdef" for char in self.contract_hash)
        ):
            raise ValueError("contract_hash must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "capability_id": self.capability_id,
            "contract_hash": self.contract_hash,
        }


@dataclass(frozen=True)
class ComponentManifest:
    """Canonical provenance for one built runtime profile."""

    profile_id: str
    components: tuple[ComponentIdentity, ...]
    knowledge_pack_version: str | None = None
    knowledge_scope_version: str | None = None
    profile_config_hash: str | None = None
    code_commit: str | None = None
    capability_contracts: tuple[CapabilityContractRef, ...] = ()
    source_catalog_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("manifest profile_id must be non-empty")
        if self.profile_config_hash is not None and (
            not isinstance(self.profile_config_hash, str) or len(self.profile_config_hash) != 64
        ):
            raise ValueError("profile_config_hash must be a SHA-256 hex digest or null")
        if self.code_commit is not None and (
            not isinstance(self.code_commit, str) or not self.code_commit.strip()
        ):
            raise ValueError("code_commit must be a non-empty string or null")
        if self.source_catalog_hash is not None and (
            not isinstance(self.source_catalog_hash, str)
            or len(self.source_catalog_hash) != 64
            or any(char not in "0123456789abcdef" for char in self.source_catalog_hash)
        ):
            raise ValueError("source_catalog_hash must be a lowercase SHA-256 digest or null")
        ordered = tuple(sorted(self.components, key=lambda item: item.identity_key))
        if len({item.identity_key for item in ordered}) != len(ordered):
            raise ValueError("manifest contains duplicate component identities")
        object.__setattr__(self, "components", ordered)
        capabilities = tuple(
            sorted(self.capability_contracts, key=lambda item: item.capability_id)
        )
        if not all(isinstance(item, CapabilityContractRef) for item in capabilities):
            raise TypeError("capability_contracts must contain CapabilityContractRef values")
        if len({item.capability_id for item in capabilities}) != len(capabilities):
            raise ValueError("manifest contains duplicate capability contracts")
        object.__setattr__(self, "capability_contracts", capabilities)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "profile_id": self.profile_id,
            "components": [item.to_dict() for item in self.components],
            "knowledge_pack_version": self.knowledge_pack_version,
            "knowledge_scope_version": self.knowledge_scope_version,
            "profile_config_hash": self.profile_config_hash,
            "code_commit": self.code_commit,
        }
        # Omitting this key for legacy profiles preserves their canonical bytes
        # and frozen manifest hashes exactly.
        if self.capability_contracts:
            result["capability_contracts"] = [
                item.to_dict() for item in self.capability_contracts
            ]
        if self.source_catalog_hash is not None:
            result["source_catalog_hash"] = self.source_catalog_hash
        return result

    @property
    def build_commit(self) -> str | None:
        """Compatibility alias for callers that call the field build_commit."""

        return self.code_commit

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @property
    def manifest_hash(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def component_manifest_hash(self) -> str:
        return self.manifest_hash

    def write(self, path) -> None:
        """Persist the manifest and its hash as a deterministic JSON artifact."""

        payload = {**self.to_dict(), "manifest_hash": self.manifest_hash}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def config_hash(value: object) -> str:
    """Hash JSON-compatible component configuration without object identity."""

    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("component configuration must be JSON-compatible") from exc
    return sha256(encoded.encode("utf-8")).hexdigest()


def implementation_name(value: object) -> str:
    """Return a stable source implementation name for provenance."""

    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


@dataclass(frozen=True)
class BuiltComponent:
    """Factory result accepted by ComponentRegistry."""

    instance: Any
    identity: ComponentIdentity | None = None
