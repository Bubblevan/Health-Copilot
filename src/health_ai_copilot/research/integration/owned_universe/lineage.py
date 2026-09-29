"""Complete deterministic lineage rows for owned U2-E pilot artifacts."""

from __future__ import annotations

from typing import Any

from .schema import CONTENT_ORIGIN, WORLD_NOTICE, LatentWorld


def lineage_row(world: LatentWorld, spec_hash: str, generator_version: str) -> dict[str, Any]:
    parent_artifacts = sorted({fact.source_artifact_id for fact in world.graph.facts})
    return {
        "artifact_id": world.world_id,
        "episode_id": world.world_id.removeprefix("LW-"),
        "generator_version": generator_version,
        "spec_hash": spec_hash,
        "seed": world.scenario_seed,
        "split_role": world.split_role,
        "subject_id": world.subject_id,
        "persona_seed": world.persona_seed,
        "scenario_family": world.scenario_family,
        "template_family": world.template_family,
        "surface_variant": world.surface_variant,
        "counterfactual_family_id": world.counterfactual_family_id,
        "parent_latent_world_id": world.world_id,
        "parent_artifact_ids": parent_artifacts,
        "parent_fact_ids": list(world.graph.required_fact_ids),
        "creation_time": world.creation_time,
        "content_origin": CONTENT_ORIGIN,
        "notice": WORLD_NOTICE,
    }
