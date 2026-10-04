"""Small internal contracts for adaptive MDAgents reasoning and its skills."""

from enum import StrEnum


class RouteMode(StrEnum):
    SINGLE = "SINGLE"
    TEAM = "TEAM"


class WorkerRole(StrEnum):
    PATIENT_CONTEXT = "patient_context"
    EVIDENCE = "evidence"
    CARE = "care"
