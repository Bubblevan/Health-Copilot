"""Evidence-aware generation interfaces and providers."""

from .base import GenerationError, Generator
from .openai_compatible import OpenAICompatibleGenerator

__all__ = ["GenerationError", "Generator", "OpenAICompatibleGenerator"]
