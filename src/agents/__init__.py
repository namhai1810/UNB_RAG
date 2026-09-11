"""Agent layer.

Every agent is a plain function over typed inputs and outputs - no framework
coupling - so it can be unit-tested and swapped without touching the graph.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

DOMAIN = (
    "an English-language corpus of authoritative cyber-security publications "
    "(NIST Cybersecurity Framework and Special Publications, CISA/FBI joint "
    "advisories and guides). It covers incident response, ransomware defence, "
    "risk management, governance, detection and recovery."
)


class AgentOutput(BaseModel):
    """Base for agent response schemas.

    `extra="forbid"` makes Pydantic emit `additionalProperties: false`, which is
    required by strict structured outputs on both backends.
    """

    model_config = ConfigDict(extra="forbid")
