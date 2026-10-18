from __future__ import annotations

from typing import Any

from backend.app.services.reading_difficulty.explain import LOW_CONFIDENCE_THRESHOLD

ACTIVITY_LIBRARY: dict[str, list[str]] = {
    "phonological": [
        "Tap and copy rhythmic syllables.",
        "Listen for and repeat target sounds.",
        "Play sound-matching games with minimal pairs.",
    ],
    "decoding": [
        "Blend letter-sound chunks into words.",
        "Read short decodable passages aloud.",
        "Use flashcards for common word patterns.",
    ],
    "fluency": [
        "Read the same passage twice for smooth pacing.",
        "Do timed repeated reading with phrase cues.",
        "Track reading speed and expression goals.",
    ],
    "pronunciation": [
        "Repeat words with clear mouth movements.",
        "Practice sentence stress and intonation.",
        "Use mirror work for sound formation cues.",
    ],
}


def build_intervention(
    intensity: str,
    confidence: float,
    difficulty_level: str,
    profile: dict[str, Any],
    prosody: dict[str, Any],
    weak_skill_confidence: float | None = None,
) -> dict[str, Any]:
    """Build the post-model intervention recommendation with review rules."""
    focus_area = profile.get("primary_focus_area") if isinstance(profile, dict) else None
    valid_focus_areas = tuple(ACTIVITY_LIBRARY.keys())
    if focus_area not in valid_focus_areas:
        raise ValueError(
            f"Invalid focus_area '{focus_area}'. Expected one of {valid_focus_areas}. "
            "This must come from the model profile and should never be silently defaulted."
        )
    weaknesses = list((profile or {}).get("weaknesses", [])) if isinstance(profile, dict) else []

    review_reasons: list[str] = []
    requires_professional_review = False

    if weak_skill_confidence is not None and weak_skill_confidence < LOW_CONFIDENCE_THRESHOLD:
        review_reasons.append("low confidence in weak-skill prediction")
        requires_professional_review = True

    if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
        review_reasons.append("low confidence")
        requires_professional_review = True

    if str(intensity).lower() == "high" or str(intensity) == "High":
        review_reasons.append("high intensity")
        requires_professional_review = True

    if len(weaknesses) >= 3:
        review_reasons.append("multiple weak areas")
        requires_professional_review = True

    return {
        "intervention_intensity": intensity,
        "focus_area": focus_area,
        "requires_professional_review": requires_professional_review,
        "review_reasons": review_reasons,
        "disclaimer": "Decision support only; not a diagnosis.",
    }
