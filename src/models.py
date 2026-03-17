from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class AgentResult:
    agent_name:       str
    answer:           str
    tools_called:     List[str] = field(default_factory=list)
    raw_data:         Dict      = field(default_factory=dict)
    confidence:       float     = 0.0
    issues_found:     List[str] = field(default_factory=list)
    prompt_tokens:    int       = 0
    completion_tokens: int      = 0
    cost_usd:         float     = 0.0
