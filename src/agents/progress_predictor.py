"""
Progress Prediction for Dense Reward Signals.

Implements the LeanProgress approach for guiding proof search:
- Predicts estimated steps remaining to complete proof
- Provides dense reward signal (vs binary pass/fail)
- Enables A* search with informed heuristics

From the architecture doc:
"This module introduces a dense reward signal by predicting the
number of remaining steps to the proof. This estimate serves as
the heuristic function h(n) in search algorithms like A*."

The predictor uses the current ProofState (goals, hypotheses) to
estimate proof complexity. Can be backed by:
- Rule-based heuristics (fast, interpretable)
- Neural network (more accurate, requires training)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol


class PredictionConfidence(Enum):
    """Confidence level of a progress prediction."""

    HIGH = auto()  # Strong signal (e.g., obvious tactics)
    MEDIUM = auto()  # Moderate confidence
    LOW = auto()  # Weak signal
    UNCERTAIN = auto()  # No clear prediction


@dataclass
class ProgressPrediction:
    """
    A prediction of remaining proof steps.

    Attributes:
        estimated_steps: Predicted number of remaining tactics
        confidence: How confident we are in this estimate
        reasoning: Human-readable explanation
        heuristic_value: Value for A* search (lower = closer to goal)
    """

    estimated_steps: float
    confidence: PredictionConfidence = PredictionConfidence.MEDIUM
    reasoning: str = ""

    # Additional signals for search
    is_trivial: bool = False  # Likely solvable in 1-2 steps
    is_complex: bool = False  # Likely needs 10+ steps
    suggested_tactics: list[str] = field(default_factory=list)

    @property
    def heuristic_value(self) -> float:
        """
        Heuristic value for A* search.

        Lower values are better (closer to goal).
        Adjusted by confidence.
        """
        confidence_factor = {
            PredictionConfidence.HIGH: 0.8,
            PredictionConfidence.MEDIUM: 1.0,
            PredictionConfidence.LOW: 1.5,
            PredictionConfidence.UNCERTAIN: 2.0,
        }[self.confidence]

        return self.estimated_steps * confidence_factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_steps": self.estimated_steps,
            "confidence": self.confidence.name,
            "reasoning": self.reasoning,
            "heuristic_value": self.heuristic_value,
            "is_trivial": self.is_trivial,
            "is_complex": self.is_complex,
            "suggested_tactics": self.suggested_tactics,
        }


class ProgressPredictor(Protocol):
    """Protocol for progress predictors."""

    async def predict(
        self,
        goals: list[str],
        hypotheses: list[dict[str, str]],
        tactics_so_far: list[str],
    ) -> ProgressPrediction:
        """Predict remaining steps to complete the proof."""
        ...


class RuleBasedPredictor:
    """
    Rule-based progress predictor using heuristics.

    This is a fast, interpretable baseline that analyzes:
    - Goal structure (quantifiers, connectives)
    - Hypothesis availability
    - Patterns that suggest specific tactics

    While less accurate than neural approaches, it's:
    - Instant (no inference latency)
    - Interpretable (explains predictions)
    - Doesn't require training data
    """

    # Patterns that suggest trivial goals
    _TRIVIAL_PATTERNS = [
        r"^True$",  # Trivial truth
        r"^\d+\s*=\s*\d+$",  # Numeric equality
        r"^(\w+)\s*=\s*\1$",  # Reflexive equality
        r"^∀.*,\s*(\w+)\s*=\s*\1$",  # Universal reflexivity
    ]

    # Patterns that suggest specific tactics
    _TACTIC_HINTS = {
        r"∀": ("intro", 1),  # Universal quantifier
        r"→": ("intro", 1),  # Implication
        r"∃": ("use", 2),  # Existential
        r"∧": ("constructor", 2),  # Conjunction
        r"∨": ("left|right", 2),  # Disjunction
        r"¬": ("intro", 1),  # Negation (via contradiction)
        r"=": ("rfl|simp|ring", 1),  # Equality
        r"≤|<|≥|>": ("linarith|omega", 1),  # Inequalities
        r"↔": ("constructor", 3),  # Iff
        r"ℕ|ℤ|Nat|Int": ("omega|simp|ring", 2),  # Numeric types
    }

    # Complexity multipliers for goal features
    _COMPLEXITY_FACTORS = {
        "forall_depth": 0.5,  # Each ∀ adds ~0.5 steps (intro)
        "exists_depth": 1.5,  # Each ∃ adds ~1.5 steps (construct witness)
        "conjunction_depth": 1.0,  # Each ∧ adds ~1 step (constructor)
        "disjunction_depth": 2.0,  # Each ∨ adds ~2 steps (case analysis)
        "negation_depth": 2.0,  # Negation often needs contradiction
        "function_apps": 0.3,  # Each application adds complexity
    }

    def __init__(self):
        # Compile patterns
        self._trivial_compiled = [re.compile(p, re.IGNORECASE) for p in self._TRIVIAL_PATTERNS]

    async def predict(
        self,
        goals: list[str],
        hypotheses: list[dict[str, str]],
        tactics_so_far: list[str],
    ) -> ProgressPrediction:
        """
        Predict remaining steps using rule-based heuristics.
        """
        if not goals:
            # No goals = proof complete
            return ProgressPrediction(
                estimated_steps=0,
                confidence=PredictionConfidence.HIGH,
                reasoning="No goals remaining",
                is_trivial=True,
            )

        # Analyze each goal and aggregate
        total_estimate = 0.0
        all_suggestions: list[str] = []
        is_trivial = True
        is_complex = False

        for goal in goals:
            estimate, suggestions = self._analyze_goal(goal, hypotheses)
            total_estimate += estimate
            all_suggestions.extend(suggestions)

            if estimate > 2:
                is_trivial = False
            if estimate > 10:
                is_complex = True

        # Adjust for number of goals
        if len(goals) > 1:
            total_estimate *= 1.2  # Multiple goals add overhead

        # Adjust for tactics already used
        # Longer proofs tend to get longer (but with diminishing returns)
        depth_factor = 1.0 + 0.1 * math.log1p(len(tactics_so_far))
        total_estimate *= depth_factor

        # Determine confidence
        if is_trivial:
            confidence = PredictionConfidence.HIGH
        elif is_complex:
            confidence = PredictionConfidence.LOW
        else:
            confidence = PredictionConfidence.MEDIUM

        return ProgressPrediction(
            estimated_steps=total_estimate,
            confidence=confidence,
            reasoning=self._generate_reasoning(goals, total_estimate),
            is_trivial=is_trivial and len(goals) == 1,
            is_complex=is_complex,
            suggested_tactics=list(set(all_suggestions))[:5],
        )

    def _analyze_goal(
        self,
        goal: str,
        hypotheses: list[dict[str, str]],
    ) -> tuple[float, list[str]]:
        """
        Analyze a single goal for complexity estimate.

        Returns (estimated_steps, suggested_tactics).
        """
        # Check trivial patterns
        for pattern in self._trivial_compiled:
            if pattern.match(goal.strip()):
                return (1.0, ["rfl", "trivial"])

        estimate = 1.0  # Base: at least one tactic needed
        suggestions: list[str] = []

        # Count structural features
        forall_count = goal.count("∀") + goal.count("forall")
        exists_count = goal.count("∃") + goal.count("exists")
        conj_count = goal.count("∧") + goal.count("/\\")
        disj_count = goal.count("∨") + goal.count("\\/")
        neg_count = goal.count("¬") + goal.count("not ")
        arrow_count = goal.count("→") + goal.count("->")

        # Add complexity for each feature
        estimate += forall_count * self._COMPLEXITY_FACTORS["forall_depth"]
        estimate += exists_count * self._COMPLEXITY_FACTORS["exists_depth"]
        estimate += conj_count * self._COMPLEXITY_FACTORS["conjunction_depth"]
        estimate += disj_count * self._COMPLEXITY_FACTORS["disjunction_depth"]
        estimate += neg_count * self._COMPLEXITY_FACTORS["negation_depth"]

        # Suggest tactics based on goal structure
        for pattern, (tactic, _) in self._TACTIC_HINTS.items():
            if re.search(pattern, goal):
                suggestions.extend(tactic.split("|"))

        # Check if hypotheses can help
        hyp_names = set()
        for h in hypotheses:
            hyp_names.update(h.keys())

        # If we have hypotheses that match parts of the goal, reduce estimate
        for name in hyp_names:
            if name in goal:
                estimate *= 0.8  # Hypothesis likely useful
                suggestions.append(f"exact {name}")
                suggestions.append(f"apply {name}")

        # Normalize: caps and floors
        estimate = max(1.0, min(estimate, 50.0))

        return (estimate, suggestions)

    def _generate_reasoning(self, goals: list[str], estimate: float) -> str:
        """Generate human-readable reasoning for the prediction."""
        parts = []

        if len(goals) == 0:
            return "Proof complete"

        parts.append(f"{len(goals)} goal(s) remaining")

        if estimate <= 2:
            parts.append("likely solvable with basic tactics")
        elif estimate <= 5:
            parts.append("moderate complexity")
        elif estimate <= 10:
            parts.append("requires multi-step proof")
        else:
            parts.append("complex proof expected")

        return "; ".join(parts)


class NeuralPredictor:
    """
    Neural network-based progress predictor.

    Uses a language model fine-tuned on proof trajectories
    to predict remaining steps. More accurate than rule-based
    but requires inference.

    From architecture doc:
    "It utilizes a fine-tuned language model (e.g., DeepSeek Coder 1.3B)
    that takes the current ProofState and outputs a scalar estimate."
    """

    def __init__(
        self,
        model_name: str = "deepseek-coder-1.3b-instruct",
        api_endpoint: str | None = None,
    ):
        """
        Initialize the neural predictor.

        Args:
            model_name: Model to use for prediction
            api_endpoint: Optional API endpoint for hosted model
        """
        self._model_name = model_name
        self._api_endpoint = api_endpoint
        self._llm_client = None

    def set_llm_client(self, client) -> None:
        """Set the LLM client for inference."""
        self._llm_client = client

    async def predict(
        self,
        goals: list[str],
        hypotheses: list[dict[str, str]],
        tactics_so_far: list[str],
    ) -> ProgressPrediction:
        """
        Predict remaining steps using neural model.
        """
        if not self._llm_client:
            # Fallback to rule-based
            fallback = RuleBasedPredictor()
            return await fallback.predict(goals, hypotheses, tactics_so_far)

        # Format prompt
        prompt = self._format_prompt(goals, hypotheses, tactics_so_far)

        try:
            # Get prediction from LLM
            response = await self._llm_client.complete(
                prompt,
                max_tokens=50,
                temperature=0.0,  # Deterministic
            )

            return self._parse_response(response, goals)

        except Exception:
            # Fallback to rule-based
            fallback = RuleBasedPredictor()
            return await fallback.predict(goals, hypotheses, tactics_so_far)

    def _format_prompt(
        self,
        goals: list[str],
        hypotheses: list[dict[str, str]],
        tactics_so_far: list[str],
    ) -> str:
        """Format the prediction prompt."""
        parts = ["Predict the number of remaining proof steps."]
        parts.append("")
        parts.append("Current goals:")
        for i, goal in enumerate(goals):
            parts.append(f"  {i + 1}. {goal}")

        if hypotheses:
            parts.append("")
            parts.append("Available hypotheses:")
            for h in hypotheses[:10]:  # Limit
                for name, typ in h.items():
                    parts.append(f"  {name} : {typ}")

        if tactics_so_far:
            parts.append("")
            parts.append(f"Tactics used so far ({len(tactics_so_far)}):")
            for tac in tactics_so_far[-5:]:  # Last 5
                parts.append(f"  {tac}")

        parts.append("")
        parts.append("Respond with only a number (estimated remaining steps):")

        return "\n".join(parts)

    def _parse_response(
        self,
        response: str,
        goals: list[str],
    ) -> ProgressPrediction:
        """Parse the model's response."""
        # Extract number from response
        numbers = re.findall(r"\d+(?:\.\d+)?", response)

        if numbers:
            estimate = float(numbers[0])
        else:
            estimate = len(goals) * 3  # Default guess

        # Determine confidence based on response quality
        if estimate <= 0:
            estimate = 1.0
            confidence = PredictionConfidence.UNCERTAIN
        elif estimate > 50:
            estimate = 50.0
            confidence = PredictionConfidence.LOW
        else:
            confidence = PredictionConfidence.MEDIUM

        return ProgressPrediction(
            estimated_steps=estimate,
            confidence=confidence,
            reasoning=f"Neural estimate: {estimate:.1f} steps",
            is_trivial=estimate <= 2,
            is_complex=estimate > 10,
        )


class EnsemblePredictor:
    """
    Ensemble of multiple predictors for robust estimates.

    Combines rule-based and neural predictions.
    """

    def __init__(
        self,
        rule_weight: float = 0.3,
        neural_weight: float = 0.7,
    ):
        self._rule_predictor = RuleBasedPredictor()
        self._neural_predictor = NeuralPredictor()
        self._rule_weight = rule_weight
        self._neural_weight = neural_weight

    def set_llm_client(self, client) -> None:
        """Set LLM client for neural predictor."""
        self._neural_predictor.set_llm_client(client)

    async def predict(
        self,
        goals: list[str],
        hypotheses: list[dict[str, str]],
        tactics_so_far: list[str],
    ) -> ProgressPrediction:
        """Combine predictions from multiple sources."""
        # Get both predictions
        rule_pred = await self._rule_predictor.predict(goals, hypotheses, tactics_so_far)
        neural_pred = await self._neural_predictor.predict(goals, hypotheses, tactics_so_far)

        # Weighted average
        combined_estimate = (
            self._rule_weight * rule_pred.estimated_steps
            + self._neural_weight * neural_pred.estimated_steps
        )

        # Use higher confidence
        if rule_pred.confidence.value < neural_pred.confidence.value:
            confidence = rule_pred.confidence
        else:
            confidence = neural_pred.confidence

        # Combine suggestions
        suggestions = list(set(rule_pred.suggested_tactics + neural_pred.suggested_tactics))

        return ProgressPrediction(
            estimated_steps=combined_estimate,
            confidence=confidence,
            reasoning=f"Rule: {rule_pred.estimated_steps:.1f}, Neural: {neural_pred.estimated_steps:.1f}",
            is_trivial=rule_pred.is_trivial and neural_pred.is_trivial,
            is_complex=rule_pred.is_complex or neural_pred.is_complex,
            suggested_tactics=suggestions[:5],
        )


def create_progress_predictor(
    use_neural: bool = False,
    llm_client=None,
) -> ProgressPredictor:
    """
    Factory to create a progress predictor.

    Args:
        use_neural: Whether to use neural prediction
        llm_client: Optional LLM client for neural prediction

    Returns:
        Configured predictor
    """
    if use_neural and llm_client:
        predictor = EnsemblePredictor()
        predictor.set_llm_client(llm_client)
        return predictor
    else:
        return RuleBasedPredictor()
