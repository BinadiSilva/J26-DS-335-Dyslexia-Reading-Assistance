"""All parent/child-facing wording lives here so it can be reviewed with a specialist.
Rule: never show raw feature names, model names or the word 'confidence' to parents/children."""
from __future__ import annotations

SKILLS = {
    "phonological": {"nick": "Sound Detective", "emoji": "🔊",
                     "blurb": "listening for the small sounds inside words"},
    "decoding": {"nick": "Word Builder", "emoji": "🧩",
                 "blurb": "putting letters and sounds together to build words"},
    "fluency": {"nick": "Smooth Reader", "emoji": "🌊",
                "blurb": "reading smoothly, like telling a story"},
    "pronunciation": {"nick": "Clear Speaker", "emoji": "🗣️",
                      "blurb": "saying words clearly"},
}

# level -> (kid headline, marker position on the journey track 0-1, kid sentence, grown-up sentence)
LEVELS = {
    "Low": ("Reading star in the making!", 0.86,
            "Reading is going really smoothly today.",
            "Reading went smoothly in this session."),
    "Moderate": ("Growing stronger every day!", 0.56,
                 "Reading is coming along nicely, with a few things to keep practising.",
                 "Reading is progressing, with a few areas that would benefit from regular practice."),
    "High": ("What a brave reader!", 0.28,
             "Some words are tricky right now, and that is okay. A little extra help makes it easier.",
             "Some parts of reading were difficult in this session. Extra support and regular, short "
             "practice are likely to help."),
}

FEATURE_PHRASES = {
    "words_correct": "how many words were read correctly",
    "words_incorrect": "how many words were misread",
    "words_total": "the length of the passage",
    "word_error_rate": "how many words had mistakes",
    "reading_speed_wpm": "the reading pace",
    "reading_duration_sec": "how long the reading took",
    "fluency_score": "how smoothly the reading flowed",
    "pronunciation_accuracy": "how clearly the sounds were said",
    "hesitation_count": "how often there were pauses or hesitations",
    "repetition_count": "repeated words or sounds",
    "omission_count": "words or sounds that were skipped",
    "substitution_count": "words swapped for other words",
    "insertion_count": "extra words added while reading",
    "pitch_variability": "how the voice rose and fell",
    "energy_variability": "how steady the voice volume was",
    "energy_mean": "the overall energy of the voice",
    "pause_pattern_score": "the pattern of pauses between words",
    "avg_pause_duration": "the length of the pauses",
    "normalized_speed": "reading pace compared with other children in the same grade",
    "normalized_fluency": "smoothness compared with other children in the same grade",
    "normalized_accuracy": "accuracy compared with other children in the same grade",
}
FALLBACK_PHRASE = "other patterns in the way the reading sounded"

# Background details are not reasons a parent should be shown; the teacher view still lists every feature.
_DEMOGRAPHIC_EXACT = {"age", "grade", "gender", "passage_id", "passage_level"}
_DEMOGRAPHIC_PREFIX = ("gender_", "passage_id_", "passage_level_")


def _is_background(name: str) -> bool:
    n = str(name).strip().lower()
    return n in _DEMOGRAPHIC_EXACT or n.startswith(_DEMOGRAPHIC_PREFIX)


def skill_nick(skill: str) -> str:
    s = str(skill or "").strip().lower()
    return SKILLS.get(s, {}).get("nick", "Reading Explorer")


def skill_emoji(skill: str) -> str:
    return SKILLS.get(str(skill or "").strip().lower(), {}).get("emoji", "📖")


def skill_blurb(skill: str) -> str:
    return SKILLS.get(str(skill or "").strip().lower(), {}).get("blurb", "reading skills")


def level_parts(level: str) -> tuple:
    return LEVELS.get(str(level).title(), LEVELS["Moderate"])


def parent_level_message(level: str) -> str:
    return level_parts(level)[3]


def parent_feature_phrase(feature_name: str) -> str:
    """Return a plain-language sentence for a feature. Never returns the raw feature name."""
    key = str(feature_name).strip().lower()
    phrase = FEATURE_PHRASES.get(key, FALLBACK_PHRASE)
    return f"This was influenced by {phrase}."


def explanation_phrases(shap_top_features: dict, target: str, n: int = 3) -> list[str]:
    """Top-n unique plain-language reasons for a target (weak_skill / intervention_intensity)."""
    items = shap_top_features.get(target, []) if isinstance(shap_top_features, dict) else []
    out: list[str] = []
    for feature_name, _v in items:
        if _is_background(feature_name):
            continue
        phrase = FEATURE_PHRASES.get(str(feature_name).strip().lower(), FALLBACK_PHRASE)
        if phrase not in out:
            out.append(phrase)
        if len(out) >= n:
            break
    return out or [FALLBACK_PHRASE]


def build_parent_explanation(shap_top_features: dict, target_name: str | None = None) -> str:
    phrases = explanation_phrases(shap_top_features, target_name or "weak_skill")
    return "This was mostly influenced by " + ", ".join(phrases[:-1]) + (" and " if len(phrases) > 1 else "") + phrases[-1] + "."
