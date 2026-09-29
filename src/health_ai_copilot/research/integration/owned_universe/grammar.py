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
    "TRAIN_LOOKUP": (
        QueryTemplate("TRAIN_LOOKUP", "01", "Return the field indexed by {key}."),
        QueryTemplate("TRAIN_LOOKUP", "02", "What value is associated with {key}?"),
    ),
    "TRAIN_TEMPORAL": (
        QueryTemplate("TRAIN_TEMPORAL", "01", "Compare the earlier and later entries for {key}."),
        QueryTemplate("TRAIN_TEMPORAL", "02", "At the requested time, which value for {key} applies?"),
    ),
    "TRAIN_UPDATE": (
        QueryTemplate("TRAIN_UPDATE", "01", "After the latest synthetic update, what is stored for {key}?"),
        QueryTemplate("TRAIN_UPDATE", "02", "Report the current state associated with {key}."),
    ),
    "TRAIN_AGGREGATE": (
        QueryTemplate("TRAIN_AGGREGATE", "01", "Aggregate the synthetic entries indexed by {key}."),
        QueryTemplate("TRAIN_AGGREGATE", "02", "Combine the matching records for {key} and return the result."),
    ),
    "TRAIN_SOURCE": (
        QueryTemplate("TRAIN_SOURCE", "01", "Within the named synthetic source family, resolve {key}."),
        QueryTemplate("TRAIN_SOURCE", "02", "Using the permitted evidence sources, return the entry for {key}."),
    ),
    "TRAIN_BOOLEAN": (
        QueryTemplate("TRAIN_BOOLEAN", "01", "Verify whether the synthetic condition indexed by {key} is present."),
        QueryTemplate("TRAIN_BOOLEAN", "02", "Does the record for {key} satisfy the stated condition?"),
    ),
    "TRAIN_SEQUENCE": (
        QueryTemplate("TRAIN_SEQUENCE", "01", "List the values for {key} in chronological order."),
        QueryTemplate("TRAIN_SEQUENCE", "02", "Return the ordered sequence associated with {key}."),
    ),
    "TRAIN_NUMERIC": (
        QueryTemplate("TRAIN_NUMERIC", "01", "Compute the requested arithmetic result for synthetic values at {key}."),
        QueryTemplate("TRAIN_NUMERIC", "02", "Calculate the numeric aggregate from entries indexed by {key}."),
    ),
    "TRAIN_ABSTAIN": (
        QueryTemplate("TRAIN_ABSTAIN", "01", "Using records valid at the requested time, determine the value for {key}."),
        QueryTemplate("TRAIN_ABSTAIN", "02", "Can the available synthetic sources support a value for {key}?"),
    ),
    "TRAIN_COMPOSE": (
        QueryTemplate("TRAIN_COMPOSE", "01", "Compose the related personal and external entries for {key}."),
        QueryTemplate("TRAIN_COMPOSE", "02", "Join the requested synthetic records associated with {key}."),
    ),
    "TRAIN_CONVERSATION": (
        QueryTemplate("TRAIN_CONVERSATION", "01", "Apply the user-confirmed preference linked to {key}."),
        QueryTemplate("TRAIN_CONVERSATION", "02", "Which prior user-confirmed instruction is indexed by {key}?"),
    ),
    "TRAIN_TREND": (
        QueryTemplate("TRAIN_TREND", "01", "Determine the direction of change in the values for {key}."),
        QueryTemplate("TRAIN_TREND", "02", "Summarize the ordered measurements associated with {key}."),
    ),
    "TRAIN_CURRENT": (
        QueryTemplate("TRAIN_CURRENT", "01", "The current synthetic note states {value} for {key}; return that value."),
        QueryTemplate("TRAIN_CURRENT", "02", "Given the current context value {value} for {key}, report it."),
    ),
    "DEV_STRUCTURAL_ORDER": (
        QueryTemplate("DEV_STRUCTURAL_ORDER", "01", "Starting with {key}, arrange the related synthetic entries by their dependency order."),
        QueryTemplate("DEV_STRUCTURAL_ORDER", "02", "Resolve the linked records for {key}, preserving the requested sequence."),
    ),
    "DEV_STRUCTURAL_COMPOSE": (
        QueryTemplate("DEV_STRUCTURAL_COMPOSE", "01", "Combine the independently described records that refer to {key}."),
        QueryTemplate("DEV_STRUCTURAL_COMPOSE", "02", "Across the available synthetic sources, assemble the required result for {key}."),
    ),
    "DEV_STRUCTURAL_CONDITION": (
        QueryTemplate("DEV_STRUCTURAL_CONDITION", "01", "Given the stated condition, verify the record selected by {key}."),
        QueryTemplate("DEV_STRUCTURAL_CONDITION", "02", "Check the linked synthetic facts for {key} and report the supported result."),
    ),
    "DEV_STRUCTURAL_CONTRAST": (
        QueryTemplate("DEV_STRUCTURAL_CONTRAST", "01", "Contrast the time-qualified entries associated with {key}."),
        QueryTemplate("DEV_STRUCTURAL_CONTRAST", "02", "Which of the linked records for {key} remains valid at the query time?"),
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
