from browns_tracking.constants import (
    YARDS_PER_SECOND_SQ_TO_M_PER_SECOND_SQ,
    YARDS_PER_SECOND_TO_MPH,
)


def test_speed_conversion_constant() -> None:
    assert YARDS_PER_SECOND_TO_MPH == 2.0454545454545454


def test_accel_conversion_constant() -> None:
    assert YARDS_PER_SECOND_SQ_TO_M_PER_SECOND_SQ == 0.9144

