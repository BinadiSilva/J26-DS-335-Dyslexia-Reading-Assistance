# COMP02 Output Contract (Components 3 and 4)

This document defines the JSON returned by the Component 2 inference pipeline via `predict_session(...)` and the downstream intervention result stored in the session log.

The project test suite asserts the exact top-level key set, so this contract is the canonical output schema.

## Top-level output schema

The result object has exactly these keys:

```python
{
    "child_id": str,
    "session_id": str,
    "timestamp": str,
    "difficulty_score": float,
    "difficulty_level": str,
    "weak_skill": str,
    "weak_skill_confidence": float,
    "intervention_intensity": str,
    "intervention_confidence": float,
    "level_consistency": dict,
    "low_confidence_flags": dict,
    "shap_top_features": dict,
    "cognitive_profile": dict,
    "prosody_engagement_index": dict,
    "intervention": dict,
    "model_versions": dict,
}
```

## Field-by-field contract

### 1) Identity and timing

- `child_id: str`
  - The learner identifier, e.g. `demo_child_01` or a raw child id from the dataset.

- `session_id: str`
  - Unique session identifier for the learner.

- `timestamp: str`
  - Session date or timestamp string. It is written as the session date value in the data pipeline.

### 2) Difficulty prediction

- `difficulty_score: float`
  - Continuous difficulty estimate in the 0–1 range used by the synthetic data contract.
  - Example: `0.41`

- `difficulty_level: str`
  - Derived label from the score using the project thresholds:
    - `Low` if score < `0.26`
    - `Moderate` if score < `0.50`
    - `High` otherwise

### 3) Weak-skill prediction

- `weak_skill: str`
  - One of:
    - `phonological`
    - `decoding`
    - `fluency`
    - `pronunciation`

- `weak_skill_confidence: float`
  - Confidence of the model output for the weak-skill prediction, in the range `0.0` to `1.0`.

### 4) Intervention recommendation

- `intervention_intensity: str`
  - One of:
    - `Low`
    - `Moderate`
    - `High`

- `intervention_confidence: float`
  - Confidence of the intervention-intensity prediction, in the range `0.0` to `1.0`.

### 5) Consistency and review gating

- `level_consistency: dict`
  - Schema:

    ```python
    {
        "difficulty_level": str,
        "support_level": str,
        "consistent": bool,
    }
    ```

  - `difficulty_level` is the predicted reading difficulty label.
  - `support_level` is the predicted intervention intensity.
  - `consistent` is `True` when the support level matches the difficulty level.
  - If they disagree, the pipeline appends a review reason and sets the professional-review flag to `True`.

- `low_confidence_flags: dict`
  - Schema:

    ```python
    {
        "weak_skill": bool,
        "intervention_intensity": bool,
    }
    ```

  - These are boolean flags for whether the model confidence is below the configured low-confidence threshold.

### 6) Explainability output

- `shap_top_features: dict`
  - A dictionary keyed by the three model targets:
    - `difficulty_score`
    - `weak_skill`
    - `intervention_intensity`
  - Each value is a list of the top feature contributions, shaped as:

    ```python
    [
        ["word_error_rate", 0.68],
        ["fluency_score", 0.23],
        ...
    ]
    ```

  - Each entry is a 2-item list: `[feature_name: str, contribution: float]`.

### 7) Cognitive profile

- `cognitive_profile: dict`
  - Rule-based summary of the learner profile, built from subskill scores.
  - Schema:

    ```python
    {
        "subskill_scores": {
            "phonological": float,
            "decoding": float,
            "fluency": float,
            "pronunciation": float,
        },
        "strengths": list[str],
        "weaknesses": list[str],
        "difficulty_level": str,
        "primary_focus_area": str,
        "rule_based_weakest_area": str,
        "model_rule_agreement": bool,
    }
    ```

  - `primary_focus_area` is the weak-skill prediction; `rule_based_weakest_area` is the weakest child-skill from the rule engine.

### 8) Prosody

- `prosody_engagement_index: dict`
  - A rule-based prosody summary, not a clinical diagnosis.
  - Schema:

    ```python
    {
        "prosody_engagement_index": float | None,
        "components": dict,
        "note": str,
    }
    ```

  - `components` is a dictionary of per-feature scores.
  - `note` is the disclaimer text: `Rule-based indicator; not clinically validated.`

### 9) Intervention object

- `intervention: dict`
  - Final post-model recommendation bundle used by the UI and downstream review logic.
  - Schema:

    ```python
    {
        "intervention_intensity": str,
        "focus_area": str,
        "requires_professional_review": bool,
        "review_reasons": list[str],
        "disclaimer": str,
    }
    ```

  - The key review flag is:

    - `requires_professional_review: bool`
      - `True` when the system needs a human teacher or specialist to review the result.
      - It is triggered when any of these conditions apply:
        - weak-skill confidence is below the low-confidence threshold
        - intervention confidence is below the low-confidence threshold
        - intensity is `High`
        - multiple weak areas are present
        - difficulty level and support intensity disagree
      - The actual reasons are listed in `review_reasons`.

  - The output does not include `recommended_activities` in the backend payload. The activity library remains internal and the frontend does not rely on it for the contract.

### 10) Model versions

- `model_versions: dict`
  - Maps each target to the winning model name selected during training.
  - Schema:

    ```python
    {
        "difficulty_score": str,
        "weak_skill": str,
        "intervention_intensity": str,
    }
    ```

## Exact key set

This is the exact top-level key set asserted by the pipeline tests:

```python
{
    "child_id",
    "session_id",
    "timestamp",
    "difficulty_score",
    "difficulty_level",
    "weak_skill",
    "weak_skill_confidence",
    "intervention_intensity",
    "intervention_confidence",
    "level_consistency",
    "low_confidence_flags",
    "shap_top_features",
    "cognitive_profile",
    "prosody_engagement_index",
    "intervention",
    "model_versions",
}
```

## Review semantics

`requires_professional_review` means the result is a decision-support output that should not be treated as a definitive clinical judgment. It is a human-review flag, not a diagnosis. If `True`, the system is saying: “this prediction needs teacher or specialist review because confidence, intensity, or profile complexity suggests a higher-risk / less reliable decision.”

The exact review reasons are the values in `intervention["review_reasons"]`.

## Compatibility note

- JSON-serialisable output is required.
- All values in the returned result must be plain Python values (`str`, `float`, `bool`, `dict`, `list`) so they can be stored in `outputs/session_log.jsonl`.
- This contract is stable for the Component 2 output and is used directly by the session logging and frontend consumers.
