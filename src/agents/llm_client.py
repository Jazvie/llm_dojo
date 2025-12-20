from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

from .base import get_difficulty_guidance


@dataclass
class TacticSuggestion:
    tactic: str
    confidence: float = 0.0
    explanation: str = ""


@dataclass
class ConjectureProposal:
    name: str
    statement: str
    tactics: list[str]
    difficulty: Optional[float] = None


def _extract_json_object(content: str) -> dict[str, Any] | None:
    if not content:
        return None
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None


class LLMClient:
    TACTIC_PROMPT = """You are a Lean 4 theorem proving assistant. Given the current proof state, suggest tactics to apply.

Current goal: {goal}

Hypotheses:
{hypotheses}

Tactics applied so far:
{tactics_so_far}

Suggest {num_suggestions} Lean 4 tactics that could make progress. For each tactic, give:
1. The exact tactic syntax
2. A brief explanation

Format each suggestion as:
TACTIC: <tactic>
EXPLANATION: <why this might work>

Be precise with Lean 4 syntax. Prioritize tactics likely to make progress."""

    SKETCH_PROMPT = """You are a Lean 4 theorem proving expert. Generate a proof for the following theorem.

Theorem: {theorem}

Provide a sequence of Lean 4 tactics that proves this theorem. Output ONLY the tactics, one per line.
Do not include comments or explanations. Use standard Lean 4 Mathlib tactics."""

    CONJECTURE_PROMPT = """You are participating in a Lean 4 H.O.R.S.E. match.
Your goal is to propose a single theorem that is provable in Mathlib along with a short tactic script.

Target difficulty: {target_difficulty:.2f}/1.0

{difficulty_guidance}

IMPORTANT FORMAT RULES:
1. "statement" field should contain ONLY the type signature (no "theorem" keyword or name)
2. Use ASCII type names: Nat (not ℕ), Int (not ℤ), Real (not ℝ)
3. Use "forall" keyword (not ∀ symbol)
4. Ensure the statement is mathematically TRUE and provable
5. Provide complete, working tactics

Respond with valid JSON ONLY in the following format:
{{
  "name": "short_identifier",
  "statement": "forall n m : Nat, n + m = m + n",
  "tactics": ["intro n m", "ring"],
  "difficulty": {target_difficulty:.2f}
}}
"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
    ):
        # Fall back to env vars so CLI/config can supply endpoints without parameters
        self.api_key = (
            api_key or os.environ.get("ANSPG_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        )
        self.base_url = (
            base_url or os.environ.get("ANSPG_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        )
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client

    async def suggest_tactics(
        self,
        goal: str,
        hypotheses: list[str],
        tactics_so_far: list[str],
        num_suggestions: int = 5,
    ) -> list[TacticSuggestion]:
        client = self._get_client()

        hyp_str = "\n".join(f"  {h}" for h in hypotheses) if hypotheses else "  (none)"
        tactics_str = "\n".join(f"  {t}" for t in tactics_so_far) if tactics_so_far else "  (none)"

        prompt = self.TACTIC_PROMPT.format(
            goal=goal,
            hypotheses=hyp_str,
            tactics_so_far=tactics_str,
            num_suggestions=num_suggestions,
        )

        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )

        content = response.choices[0].message.content or ""
        return self._parse_tactic_suggestions(content)

    async def generate_proof_sketch(
        self,
        theorem_statement: str,
    ) -> list[str]:
        client = self._get_client()

        prompt = self.SKETCH_PROMPT.format(
            theorem=theorem_statement,
        )

        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )

        content = response.choices[0].message.content or ""
        tactics = [
            line.strip()
            for line in content.strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        return tactics

    async def propose_conjecture(
        self,
        difficulty: float | None = None,
    ) -> ConjectureProposal | None:
        client = self._get_client()
        target = difficulty if difficulty is not None else 0.4
        difficulty_guidance = get_difficulty_guidance(target)
        prompt = self.CONJECTURE_PROMPT.format(
            target_difficulty=target,
            difficulty_guidance=difficulty_guidance,
        )

        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=600,
        )

        content = response.choices[0].message.content or ""
        data = _extract_json_object(content)
        if not data:
            return None

        tactics = data.get("tactics") or []
        if isinstance(tactics, str):
            tactics = [tactics]

        difficulty_value = data.get("difficulty")
        try:
            parsed_difficulty = float(difficulty_value) if difficulty_value is not None else None
        except (ValueError, TypeError):
            parsed_difficulty = None

        return ConjectureProposal(
            name=str(data.get("name", "")),
            statement=str(data.get("statement", "")),
            tactics=[str(t).strip() for t in tactics if str(t).strip()],
            difficulty=parsed_difficulty,
        )

    def _parse_tactic_suggestions(self, content: str) -> list[TacticSuggestion]:
        suggestions = []
        current_tactic = None
        current_explanation = ""

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("TACTIC:"):
                if current_tactic:
                    suggestions.append(
                        TacticSuggestion(
                            tactic=current_tactic,
                            explanation=current_explanation.strip(),
                        )
                    )
                current_tactic = line[7:].strip()
                current_explanation = ""
            elif line.startswith("EXPLANATION:"):
                current_explanation = line[12:].strip()
            elif current_tactic and line:
                current_explanation += " " + line

        if current_tactic:
            suggestions.append(
                TacticSuggestion(
                    tactic=current_tactic,
                    explanation=current_explanation.strip(),
                )
            )

        return suggestions


def create_llm_client(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> LLMClient:
    return LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model or os.environ.get("ANSPG_LLM_MODEL") or "gpt-4o-mini",
    )
