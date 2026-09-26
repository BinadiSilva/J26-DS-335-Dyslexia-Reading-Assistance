"""
Synthetic RAW reading-session data generator for Lumi-IT Component 2.

Unlike the previous version, this generator produces ONLY genuinely raw inputs —
matching Table 3 of the proposal's "raw inputs from Component 1" category. It does
NOT precompute normalized_speed, normalized_fluency, normalized_accuracy,
error_pattern_features, pitch_variability, energy_variability, or pause_pattern_score.
Those are Component 2's job, computed in features.py's feature-engineering step from
these raw inputs — not baked into the dataset.

To make pitch_variability/energy_variability/pause_pattern_score genuinely derivable
(not faked), each session includes raw per-segment prosodic SAMPLES (multiple pitch/
energy readings across the session, and a list of individual pause durations) rather
than only single aggregate means. This models what Component 1 would realistically
supply from raw audio analysis.

Usage:
    python generate_synthetic_data.py --n_children 24 --sessions_per_child 5 \
        --out datasets/comp02_raw_reading_sessions.csv
"""
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

PASSAGES = {
    "P01": ("Easy", 1), "P02": ("Moderate", 2), "P03": ("Moderate", 2),
    "P04": ("Challenging", 3), "P05": ("Challenging", 3),
}

SAMPLE_SENTENCES = [
    "the little girl walked to school with her brother",
    "the children played together in the garden after school",
    "the cat was sitting quietly beside the window",
    "the boy opened the book and started reading the story",
    "the small bird flew over the tree and rested on a branch",
    "the teacher asked the students to read the new passage",
    "the family went to the park on a sunny afternoon",
    "the young student carefully followed the instructions in the classroom",
    "the children visited the library and selected interesting books to read",
    "the boy saw a colourful bird sitting in the garden",
    "the girl opened her book and read the story aloud",
    "the children walked across the road before entering the school",
    "the teacher explained the lesson using pictures and examples",
    "the student tried to understand the difficult passage without help",
    "the family spent the evening reading a new story together",
]

WEAK_SKILLS = ["phonological", "decoding", "fluency", "pronunciation"]

ABILITY_BANDS = [
    (0.02, 0.28),   # struggling readers -> mostly High difficulty
    (0.35, 0.65),   # typical readers -> mostly Moderate difficulty
    (0.72, 0.98),   # strong readers -> mostly Low difficulty
]


def _clip(x, lo, hi):
    return float(np.clip(x, lo, hi))


def generate_session(rng: np.random.Generator, child_id: str, session_idx: int,
                      base_ability: float, age: int, grade: int, gender: str,
                      start_date: datetime):
    passage_id = rng.choice(list(PASSAGES.keys()))
    passage_level_name, level_num = PASSAGES[passage_id]

    words_total = int(rng.integers(80, 150))
    accuracy_frac = _clip(rng.normal(0.35 + 0.60 * base_ability, 0.07), 0.20, 0.99)
    words_correct = int(round(words_total * accuracy_frac))
    words_incorrect = words_total - words_correct

    reading_speed_wpm = _clip(rng.normal(30 + 100 * base_ability, 10), 20, 145)
    reading_duration_sec = round(words_total / reading_speed_wpm * 60, 1)
    word_error_rate = _clip((words_incorrect / words_total) + rng.normal(0, 0.01), 0.0, 0.7)

    # --- subskill generation: TWO separate components on purpose.
    # 1) common_base is shared by all four subskills and driven by base_ability --
    #    this is what keeps fluency_score/pronunciation_accuracy (and therefore
    #    difficulty_score, which depends on them) properly sensitive to OVERALL
    #    reading severity.
    # 2) deltas is a Dirichlet-centered allocation (mean 0 across the 4 skills)
    #    that determines WHICH specific skill dips relatively lower in a given
    #    session, independent of overall severity -- this is what keeps weak_skill
    #    balanced across its 4 classes rather than one skill structurally
    #    dominating the argmin regardless of the child's actual profile.
    # An earlier version of this generator conflated these two things (either
    # letting one skill's formula have a narrower range than the others, or
    # replacing the ability-driven base entirely with a shared "total weakness"
    # pool) -- both approaches broke either weak_skill's balance or
    # difficulty_score's dynamic range. Keeping the two components additive and
    # separate avoids both failure modes. ---
    common_base = 20 + 75 * base_ability
    w = rng.dirichlet(np.ones(4) * 2.0)
    deltas = (w - 0.25) * 70
    phon_delta, dec_delta, flu_delta, pron_delta = deltas.tolist()

    phon_severity = _clip(100 - (common_base + phon_delta), 0, 100) / 100
    dec_severity = _clip(100 - (common_base + dec_delta), 0, 100) / 100

    hesitation_count = int(max(0, round(rng.poisson(0.3 + phon_severity * 8))))
    repetition_count = int(max(0, round(rng.poisson(0.2 + phon_severity * 5))))
    omission_count = int(max(0, round(rng.poisson(0.3 + dec_severity * 7))))
    substitution_count = int(max(0, round(rng.poisson(0.3 + dec_severity * 7))))
    insertion_count = int(max(0, round(rng.poisson(0.3 + (1 - base_ability) * 1.2))))

    fluency_score = _clip(common_base + flu_delta + rng.normal(0, 5), 5, 100)
    pronunciation_accuracy = _clip(common_base + pron_delta + rng.normal(0, 5), 5, 100)

    # --- genuine multi-sample prosodic signal (raw, not aggregated) ---
    # Weaker readers show more erratic pitch/energy across a session and more
    # variable, longer pauses -- but we store the RAW SAMPLES, not a precomputed
    # variability number. features.py must compute std/variance from these itself.
    n_segments = int(rng.integers(5, 9))  # session split into 5-8 reading segments
    pitch_base = rng.normal(225, 10)
    pitch_spread = 8 + 20 * (1 - base_ability)  # weaker reader -> more erratic pitch
    pitch_samples = np.clip(rng.normal(pitch_base, pitch_spread, size=n_segments), 150, 300)

    energy_base = rng.normal(0.6, 0.05)
    energy_spread = 0.03 + 0.12 * (1 - base_ability)
    energy_samples = np.clip(rng.normal(energy_base, energy_spread, size=n_segments), 0.1, 1.0)

    n_pauses = int(max(1, rng.poisson(3 + 6 * (1 - base_ability))))
    pause_mean = 0.4 + 0.7 * (1 - base_ability)
    pause_spread = 0.1 + 0.3 * (1 - base_ability)
    pause_durations = np.clip(rng.normal(pause_mean, pause_spread, size=n_pauses), 0.1, 3.0)

    # single aggregate means -- these ARE the raw fields the proposal's Table 3 lists
    # (pitch mean, energy mean, average pause duration); the sample lists above are
    # the additional raw granularity needed to derive variability honestly.
    pitch_mean = round(float(np.mean(pitch_samples)), 1)
    energy_mean = round(float(np.mean(energy_samples)), 2)
    avg_pause_duration = round(float(np.mean(pause_durations)), 2)

    transcribed_text = rng.choice(SAMPLE_SENTENCES)

    # ---- sub-skill scores (0-100, higher = stronger) drive weak-skill + difficulty.
    # These are NOT stored as columns (they would leak into the weak_skill target the
    # same way the old subskill columns did) -- used only to generate labels here. ----
    phonological_score = _clip(common_base + phon_delta + rng.normal(0, 3), 0, 100)
    decoding_score = _clip(common_base + dec_delta + rng.normal(0, 3), 0, 100)
    fluency_subscore = _clip(fluency_score + rng.normal(0, 2), 0, 100)
    pronunciation_subscore = _clip(pronunciation_accuracy + rng.normal(0, 2), 0, 100)
    subskills = {
        "phonological": phonological_score, "decoding": decoding_score,
        "fluency": fluency_subscore, "pronunciation": pronunciation_subscore,
    }
    weak_skill = min(subskills, key=subskills.get)

    difficulty_raw = (
        0.30 * (1 - accuracy_frac)
        + 0.25 * (1 - min(reading_speed_wpm / 130, 1))
        + 0.20 * (1 - fluency_score / 100)
        + 0.15 * word_error_rate
        + 0.10 * (1 - pronunciation_accuracy / 100)
    )
    difficulty_score = round(_clip(difficulty_raw + rng.normal(0, 0.04), 0, 1), 3)
    if difficulty_score < 0.26:
        difficulty_level = "Low"
    elif difficulty_score < 0.50:
        difficulty_level = "Moderate"
    else:
        difficulty_level = "High"

    n_weak_subskills = sum(1 for v in subskills.values() if v < 60)
    _order = ["Low", "Moderate", "High"]
    _idx = _order.index(difficulty_level)
    if n_weak_subskills >= 3 and _idx < len(_order) - 1:
        _idx += 1
    intervention_intensity = _order[_idx]

    session_date = start_date + timedelta(days=7 * session_idx)

    return {
        "child_id": child_id,
        "session_id": f"{child_id}-S{session_idx+1:02d}",
        "session_date": session_date.strftime("%Y-%m-%d"),
        "age": age,
        "grade": grade,
        "gender": gender,
        "passage_id": passage_id,
        "passage_level": passage_level_name,
        "reading_duration_sec": reading_duration_sec,
        "words_total": words_total,
        "words_correct": words_correct,
        "words_incorrect": words_incorrect,
        "reading_speed_wpm": round(reading_speed_wpm, 1),
        "pronunciation_accuracy": round(pronunciation_accuracy, 1),
        "fluency_score": round(fluency_score, 1),
        "word_error_rate": round(word_error_rate, 3),
        "hesitation_count": hesitation_count,
        "repetition_count": repetition_count,
        "omission_count": omission_count,
        "substitution_count": substitution_count,
        "insertion_count": insertion_count,
        "transcribed_text": transcribed_text,
        "pitch_mean": pitch_mean,
        "energy_mean": energy_mean,
        "avg_pause_duration": avg_pause_duration,
        # raw multi-sample signal, semicolon-delimited -- NOT numeric model features
        # as-is; features.py must parse these and derive real statistics from them
        "pitch_samples": ";".join(f"{v:.1f}" for v in pitch_samples),
        "energy_samples": ";".join(f"{v:.2f}" for v in energy_samples),
        "pause_durations": ";".join(f"{v:.2f}" for v in pause_durations),
        # labels
        "weak_skill": weak_skill,
        "difficulty_score": difficulty_score,
        "difficulty_level": difficulty_level,
        "intervention_intensity": intervention_intensity,
    }


def generate_dataset(n_children: int, sessions_per_child: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start_date = datetime(2026, 8, 1)

    for i in range(n_children):
        child_id = f"C{i+1:03d}"
        band_lo, band_hi = ABILITY_BANDS[i % len(ABILITY_BANDS)]
        base_ability = rng.uniform(band_lo, band_hi)
        age = int(rng.integers(7, 13))
        grade = int(_clip(age - 5, 1, 7))
        gender = rng.choice(["Male", "Female"])
        child_start = start_date + timedelta(days=int(rng.integers(0, 20)))

        for s in range(sessions_per_child):
            drift_ability = _clip(base_ability + rng.normal(0.01 * s, 0.03), 0.01, 0.99)
            row = generate_session(rng, child_id, s, drift_ability, age, grade, gender, child_start)
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAW synthetic Component 2 reading session data")
    parser.add_argument("--n_children", type=int, default=24)
    parser.add_argument("--sessions_per_child", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="datasets/comp02_raw_reading_sessions.csv")
    args = parser.parse_args()

    df = generate_dataset(args.n_children, args.sessions_per_child, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} sessions for {args.n_children} children -> {args.out}")
    print(df["difficulty_level"].value_counts())
    print(df["weak_skill"].value_counts())
    print(df["intervention_intensity"].value_counts())
