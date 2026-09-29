"""Finite, project-authored query grammar with capability-blind surface forms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryTemplate:
    template_family: str
    surface_variant: str
    text: str

    def render(self, key: str, answer_value: str | None = None) -> str:
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
}

CURRENT_CONTEXT_CUES = (
    QueryTemplate("CURRENT_CUE_LAST", "01",
                  "The current note from the last visit states {value}; report this value."),
    QueryTemplate("CURRENT_CUE_GUIDELINE", "01",
                  "The current packet already gives {value}. What is the guideline-related value?"),
    QueryTemplate("CURRENT_CUE_EVIDENCE", "01",
                  "This request's current note says {value}; what is the evidence-associated field?"),
    QueryTemplate("CURRENT_CUE_SOURCE", "01",
                  "The current text carries {value}; return the source-labeled field."),
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
