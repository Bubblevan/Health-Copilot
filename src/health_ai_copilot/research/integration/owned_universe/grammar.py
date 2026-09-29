"""Finite, project-authored query grammar with capability-blind surface forms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryTemplate:
    template_family: str
    surface_variant: str
    text: str

    def render(self, key: str, answer_value: str | None = None) -> str:
        if "{value}" in self.text and not answer_value:
            raise ValueError(f"query template {self.template_family}/{self.surface_variant} requires an answer value")
        values = {"key": key, "value": answer_value or ""}
        return self.text.format(**values)


QUERY_TEMPLATES: dict[str, tuple[QueryTemplate, ...]] = {
    "COMMON_A": (
        QueryTemplate("COMMON_A", "01", "What token is attached to {key}?"),
        QueryTemplate("COMMON_A", "02", "Give the value linked with {key}."),
    ),
    "COMMON_B": (
        QueryTemplate("COMMON_B", "01", "Please return one field for {key}."),
        QueryTemplate("COMMON_B", "02", "Which synthetic item belongs to key {key}?"),
    ),
    "TRAIN_C": (
        QueryTemplate("TRAIN_C", "01", "For code {key}, state its associated value."),
        QueryTemplate("TRAIN_C", "02", "Identify the value associated with {key}."),
    ),
    "TRAIN_D": (
        QueryTemplate("TRAIN_D", "01", "Return the field indexed by {key}, please."),
        QueryTemplate("TRAIN_D", "02", "Report the synthetic entry for key {key}."),
    ),
    "STRUCTURAL_E": (
        QueryTemplate("STRUCTURAL_E", "01", "Select the synthetic entry whose key is {key}."),
        QueryTemplate("STRUCTURAL_E", "02", "Which value belongs to the identifier {key}?"),
    ),
    "STRUCTURAL_F": (
        QueryTemplate("STRUCTURAL_F", "01", "Read the item keyed {key} and return its value."),
        QueryTemplate("STRUCTURAL_F", "02", "Resolve the synthetic field designated by {key}."),
    ),
    "TRAIN_LOOKUP": (
        QueryTemplate("TRAIN_LOOKUP", "01", "Return the field indexed by {key}."),
        QueryTemplate("TRAIN_LOOKUP", "02", "What value is associated with {key}?"),
    ),
    "TRAIN_TEMPORAL": (
        QueryTemplate("TRAIN_TEMPORAL", "01", "At the query time, return the supported result for {key}."),
        QueryTemplate("TRAIN_TEMPORAL", "02", "What result is supported for {key} at the requested time?"),
    ),
    "TRAIN_UPDATE": (
        QueryTemplate("TRAIN_UPDATE", "01", "What synthetic result is supported for {key}?"),
        QueryTemplate("TRAIN_UPDATE", "02", "Report the supported value associated with {key}."),
    ),
    "TRAIN_AGGREGATE": (
        QueryTemplate("TRAIN_AGGREGATE", "01", "Resolve the requested result from the synthetic entries indexed by {key}."),
        QueryTemplate("TRAIN_AGGREGATE", "02", "Return the result supported by the matching records for {key}."),
    ),
    "TRAIN_SOURCE": (
        QueryTemplate("TRAIN_SOURCE", "01", "Using the available evidence, return the entry for {key}."),
        QueryTemplate("TRAIN_SOURCE", "02", "What result is supported by the records for {key}?"),
    ),
    "TRAIN_BOOLEAN": (
        QueryTemplate("TRAIN_BOOLEAN", "01", "Check the available records for {key} and return the supported result."),
        QueryTemplate("TRAIN_BOOLEAN", "02", "What result is supported by the evidence for {key}?"),
    ),
    "TRAIN_SEQUENCE": (
        QueryTemplate("TRAIN_SEQUENCE", "01", "Return the requested information associated with {key}."),
        QueryTemplate("TRAIN_SEQUENCE", "02", "What result is supported for {key}?"),
    ),
    "TRAIN_NUMERIC": (
        QueryTemplate("TRAIN_NUMERIC", "01", "Resolve the requested result from the synthetic entries indexed by {key}."),
        QueryTemplate("TRAIN_NUMERIC", "02", "Return the supported result associated with {key}."),
    ),
    "TRAIN_ABSTAIN": (
        QueryTemplate("TRAIN_ABSTAIN", "01", "Using evidence valid at the requested time, report the result for {key}."),
        QueryTemplate("TRAIN_ABSTAIN", "02", "What synthetic result is supported for {key}?"),
    ),
    "TRAIN_COMPOSE": (
        QueryTemplate("TRAIN_COMPOSE", "01", "Resolve the relevant synthetic evidence associated with {key}."),
        QueryTemplate("TRAIN_COMPOSE", "02", "Return the requested result from the available records for {key}."),
    ),
    "TRAIN_CONVERSATION": (
        QueryTemplate("TRAIN_CONVERSATION", "01", "Return the synthetic item associated with {key}."),
        QueryTemplate("TRAIN_CONVERSATION", "02", "What result is linked to {key}?"),
    ),
    "TRAIN_TREND": (
        QueryTemplate("TRAIN_TREND", "01", "Determine the supported result for the synthetic entries associated with {key}."),
        QueryTemplate("TRAIN_TREND", "02", "Summarize the synthetic evidence associated with {key}."),
    ),
    "TRAIN_CURRENT": (
        QueryTemplate("TRAIN_CURRENT", "01", "Return the synthetic value associated with {key}."),
        QueryTemplate("TRAIN_CURRENT", "02", "What result is associated with {key}?"),
    ),
    "DEV_STRUCTURAL_ORDER": (
        QueryTemplate("DEV_STRUCTURAL_ORDER", "01", "Resolve the synthetic records linked to {key} and return the requested result."),
        QueryTemplate("DEV_STRUCTURAL_ORDER", "02", "What result is supported by the linked records for {key}?"),
    ),
    "DEV_STRUCTURAL_COMPOSE": (
        QueryTemplate("DEV_STRUCTURAL_COMPOSE", "01", "Resolve the synthetic records that refer to {key} and report the supported result."),
        QueryTemplate("DEV_STRUCTURAL_COMPOSE", "02", "Return the requested result from the available synthetic evidence for {key}."),
    ),
    "DEV_STRUCTURAL_CONDITION": (
        QueryTemplate("DEV_STRUCTURAL_CONDITION", "01", "Check the linked synthetic evidence for {key} and report the supported result."),
        QueryTemplate("DEV_STRUCTURAL_CONDITION", "02", "Check the linked synthetic facts for {key} and report the supported result."),
    ),
    "DEV_STRUCTURAL_CONTRAST": (
        QueryTemplate("DEV_STRUCTURAL_CONTRAST", "01", "Compare the linked synthetic evidence for {key} and report the supported result."),
        QueryTemplate("DEV_STRUCTURAL_CONTRAST", "02", "What result is supported by the linked records for {key}?"),
    ),
}

CURRENT_CONTEXT_CUES = (
    QueryTemplate("CURRENT_CUE_LAST", "01",
                  "The current note from the last visit states {value} for {key}; return that value."),
    QueryTemplate("CURRENT_CUE_GUIDELINE", "01",
                  "The current packet already gives {value} for {key}; return that value."),
    QueryTemplate("CURRENT_CUE_EVIDENCE", "01",
                  "This request's current note says {value} for {key}; return that value."),
    QueryTemplate("CURRENT_CUE_SOURCE", "01",
                  "The current text carries {value} for {key}; return that value."),
)


def choose_template(template_family: str, variant_seed: int) -> QueryTemplate:
    rows = QUERY_TEMPLATES[template_family]
    return rows[variant_seed % len(rows)]


def render_query(template_family: str, variant_seed: int, key: str) -> tuple[str, str, str]:
    template = choose_template(template_family, variant_seed)
    return template.render(key), template.template_family, template.surface_variant


def render_current_cue(
    cue_seed: int, key: str, value: str, split_template_family: str
) -> tuple[str, str, str]:
    template = CURRENT_CONTEXT_CUES[cue_seed % len(CURRENT_CONTEXT_CUES)]
    return template.render(key, value), split_template_family, template.template_family
