"""
Standalone grader for Google Brain - Ventilator Pressure Prediction.
Metric: Mean Absolute Error (MAE), excluding rows where u_out == 1.

Adapted from mle-bench without mle-bench dependencies.
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error


class InvalidSubmissionError(Exception):
    pass


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Compute MAE between submission and private answers.

    Args:
        submission: DataFrame with columns [id, pressure]
        answers:    Private test DataFrame with columns [id, pressure, u_out, ...]

    Returns:
        MAE score (lower is better)

    Raises:
        InvalidSubmissionError: if the submission format is invalid
    """
    assert "id" in answers.columns, "Answers must contain the 'id' column"
    assert "pressure" in answers.columns, "Answers must contain the 'pressure' column"

    if "id" not in submission.columns:
        raise InvalidSubmissionError("Submission must contain the 'id' column")
    if "pressure" not in submission.columns:
        raise InvalidSubmissionError("Submission must contain the 'pressure' column")
    if len(submission) != len(answers[answers["u_out"] == 0]) and len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers must have the same length")

    # Align by id
    submission = submission.sort_values(by="id").reset_index(drop=True)
    answers = answers.sort_values(by="id").reset_index(drop=True)

    # Only grade rows where u_out == 0 (expiratory phase is not scored)
    if "u_out" in answers.columns:
        graded_answers = answers[answers["u_out"] == 0].reset_index(drop=True)
        graded_submission = submission[submission["id"].isin(graded_answers["id"])].sort_values("id").reset_index(drop=True)
        graded_answers = graded_answers.sort_values("id").reset_index(drop=True)
    else:
        graded_answers = answers
        graded_submission = submission

    if (graded_submission["id"].values != graded_answers["id"].values).any():
        raise InvalidSubmissionError("Submission IDs do not match answer IDs")

    return mean_absolute_error(graded_answers["pressure"].values, graded_submission["pressure"].values)
