import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_experience_matching(
    candidate_exp_years: float, jd_required_years: float
) -> float:
    """Compute normalized delta score for experience metrics."""
    if jd_required_years == 0:
        return 1.0
    ratio = candidate_exp_years / jd_required_years
    return min(ratio, 2.0)  # Caps score at 2.0x match efficiency


def calculate_trajectory_velocity(experience_history: List[Dict[str, Any]]) -> float:
    """Determine speed of candidate career growth (promotions frequency over time).

    Trajectory Velocity is calculated as:
    V = (Count of unique promotion titles) / (Total years of work history)
    """
    if not experience_history:
        return 0.0
    total_months = sum(item.get("duration_months", 0) for item in experience_history)
    unique_roles = len(set(item.get("role", "") for item in experience_history))

    years = max(total_months / 12.0, 0.5)
    return float(unique_roles / years)


def calculate_skill_decay(skills: List[str], history: List[Dict[str, Any]]) -> float:
    """Calculate skill freshness score.

    Simulates skills decay based on how recently they were exercised in work history.
    """
    # Dummy coefficient computation
    return 0.95


def build_ltr_features(
    candidate: Dict[str, Any], job: Dict[str, Any], semantic_score: float
) -> Dict[str, float]:
    """Transform candidate profile and job requirements into raw features for LightGBM."""
    logger.info("Computing features for ranking model pipeline...")

    cand_skills = candidate.get("skills", [])
    jd_skills = job.get("required_skills", [])

    # Calculate simple intersections
    common_skills = set(cand_skills).intersection(set(jd_skills))
    skill_match_score = len(common_skills) / max(len(jd_skills), 1)

    # Exp extraction
    cand_exp = (
        sum(item.get("duration_months", 0) for item in candidate.get("experience", []))
        / 12.0
    )
    jd_exp = float(job.get("min_experience_years", 0))

    velocity = calculate_trajectory_velocity(candidate.get("experience", []))
    exp_ratio = calculate_experience_matching(cand_exp, jd_exp)
    decay = calculate_skill_decay(cand_skills, candidate.get("experience", []))

    return {
        "experience_years": cand_exp,
        "skill_match_score": skill_match_score,
        "trajectory_velocity": velocity,
        "exp_ratio": exp_ratio,
        "skill_decay": decay,
        "semantic_similarity": semantic_score,
    }
