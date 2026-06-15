from src.member3_ranking.feature_engineering import (
    calculate_trajectory_velocity,
    build_ltr_features,
)
from src.member3_ranking.ranker import LightGBMRanker


def test_trajectory_velocity_calculation():
    exp = [
        {"role": "Junior Eng", "duration_months": 12},
        {"role": "Senior Eng", "duration_months": 24},
    ]
    # Total years = 36 / 12 = 3.0
    # Unique titles = 2
    # Velocity = 2 / 3.0 = 0.666...
    velocity = calculate_trajectory_velocity(exp)
    assert round(velocity, 3) == 0.667


def test_lightgbm_ranker_inference():
    ranker = LightGBMRanker()
    candidate_features = [
        {
            "skill_match_score": 0.8,
            "semantic_similarity": 0.7,
            "trajectory_velocity": 0.5,
            "exp_ratio": 1.0,
        },
        {
            "skill_match_score": 0.4,
            "semantic_similarity": 0.5,
            "trajectory_velocity": 0.2,
            "exp_ratio": 0.5,
        },
    ]
    scores = ranker.predict_ranking(candidate_features)
    assert len(scores) == 2
    assert scores[0] > scores[1]  # First candidate features are better
