import pandas as pd
from dimensions import compute_d1_approval_rate


def test_d1_approval_rate():
    # Hand-built: 5 claims, 4 approved, 1 not -> expected rate = 0.8
    claims = pd.DataFrame({
        "claim_id": ["C1", "C2", "C3", "C4", "C5"],
        "approved": [1, 1, 1, 1, 0],
    })
    result = compute_d1_approval_rate(claims)
    expected = 0.8
    assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"
    print(f"D1 test passed: {result}")

    # Edge case: empty window
    empty_claims = pd.DataFrame({"claim_id": [], "approved": []})
    result_empty = compute_d1_approval_rate(empty_claims)
    assert result_empty == 0.0, f"Expected 0.0 for empty window, got {result_empty}"
    print(f"D1 empty-window test passed: {result_empty}")





from dimensions import compute_d1_approval_rate, compute_d2_garage_concentration


def test_d2_garage_concentration():
    # Hand-built: 10 claims, 5 to GAR-A, 3 to GAR-B, 2 to GAR-C
    # Shares: 0.5, 0.3, 0.2
    # HHI = 0.5^2 + 0.3^2 + 0.2^2 = 0.25 + 0.09 + 0.04 = 0.38
    claims = pd.DataFrame({
        "garage_id": ["GAR-A"] * 5 + ["GAR-B"] * 3 + ["GAR-C"] * 2,
    })
    result = compute_d2_garage_concentration(claims)
    expected = 0.38
    assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"
    print(f"D2 test passed: {result}")

    # Edge case: all claims to ONE garage -> HHI should be exactly 1.0 (max concentration)
    single_garage_claims = pd.DataFrame({"garage_id": ["GAR-X"] * 5})
    result_max = compute_d2_garage_concentration(single_garage_claims)
    assert abs(result_max - 1.0) < 1e-9, f"Expected 1.0, got {result_max}"
    print(f"D2 max-concentration test passed: {result_max}")

    # Edge case: empty window
    empty_claims = pd.DataFrame({"garage_id": []})
    result_empty = compute_d2_garage_concentration(empty_claims)
    assert result_empty == 0.0, f"Expected 0.0, got {result_empty}"
    print(f"D2 empty-window test passed: {result_empty}")



from dimensions import compute_d1_approval_rate, compute_d2_garage_concentration, compute_d3_threshold_proximity


def test_d3_threshold_proximity():
    threshold = 1000.0  # simple round number for easy hand-calculation
    # 5 claims: 2 are threshold-hugging (950, 980 -- both in [950, 1000)),
    # 3 are not (500, 1000 exactly, 1200)
    claims = pd.DataFrame({
        "claim_amount": [950.0, 980.0, 500.0, 1000.0, 1200.0],
    })
    result = compute_d3_threshold_proximity(claims, threshold)
    expected = 2 / 5  # = 0.4
    assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"
    print(f"D3 test passed: {result}")

    # Edge case: claim exactly AT threshold should NOT count (we said < threshold, not <=)
    at_threshold_claims = pd.DataFrame({"claim_amount": [1000.0]})
    result_at = compute_d3_threshold_proximity(at_threshold_claims, threshold)
    assert result_at == 0.0, f"Expected 0.0 (claim at threshold shouldn't count), got {result_at}"
    print(f"D3 at-threshold edge case passed: {result_at}")

    # Edge case: empty window
    empty_claims = pd.DataFrame({"claim_amount": []})
    result_empty = compute_d3_threshold_proximity(empty_claims, threshold)
    assert result_empty == 0.0, f"Expected 0.0, got {result_empty}"
    print(f"D3 empty-window test passed: {result_empty}")


from dimensions import (
    compute_d1_approval_rate, compute_d2_garage_concentration,
    compute_d3_threshold_proximity, compute_d4_surveyor_diversity
)


def test_d4_surveyor_diversity():
    # Hand-built: 10 claims, 4 unique surveyors used
    # Formula: (4 / 10) * 50 = 20.0
    claims = pd.DataFrame({
        "surveyor_id": ["SUR-A", "SUR-A", "SUR-B", "SUR-B", "SUR-B",
                         "SUR-C", "SUR-C", "SUR-D", "SUR-D", "SUR-D"],
    })
    result = compute_d4_surveyor_diversity(claims)
    expected = 20.0
    assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"
    print(f"D4 test passed: {result}")

    # Edge case: perfect diversity -- every claim uses a different surveyor
    # 5 claims, 5 unique surveyors -> (5/5)*50 = 50.0
    perfect_claims = pd.DataFrame({"surveyor_id": ["SUR-1", "SUR-2", "SUR-3", "SUR-4", "SUR-5"]})
    result_perfect = compute_d4_surveyor_diversity(perfect_claims)
    assert abs(result_perfect - 50.0) < 1e-9, f"Expected 50.0, got {result_perfect}"
    print(f"D4 perfect-diversity test passed: {result_perfect}")

    # Edge case: zero diversity -- every claim uses the SAME surveyor
    # 5 claims, 1 unique surveyor -> (1/5)*50 = 10.0
    collapsed_claims = pd.DataFrame({"surveyor_id": ["SUR-X"] * 5})
    result_collapsed = compute_d4_surveyor_diversity(collapsed_claims)
    assert abs(result_collapsed - 10.0) < 1e-9, f"Expected 10.0, got {result_collapsed}"
    print(f"D4 collapsed-diversity test passed: {result_collapsed}")

    # Edge case: empty window
    empty_claims = pd.DataFrame({"surveyor_id": []})
    result_empty = compute_d4_surveyor_diversity(empty_claims)
    assert result_empty == 0.0, f"Expected 0.0, got {result_empty}"
    print(f"D4 empty-window test passed: {result_empty}")


from dimensions import (
    compute_d1_approval_rate, compute_d2_garage_concentration,
    compute_d3_threshold_proximity, compute_d4_surveyor_diversity,
    compute_d5_turnaround_variance
)


def test_d5_turnaround_variance():
    # Hand-built: processing days = [5, 7, 9]
    # mean = 7, deviations = [-2, 0, 2], squared = [4, 0, 4]
    # sample variance = (4+0+4) / (3-1) = 4.0  [pandas .std() uses n-1 by default]
    # std = sqrt(4.0) = 2.0
    claims = pd.DataFrame({"processing_days": [5.0, 7.0, 9.0]})
    result = compute_d5_turnaround_variance(claims)
    expected = 2.0
    assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"
    print(f"D5 test passed: {result}")

    # Edge case: all claims processed in exactly the same time -> std = 0
    consistent_claims = pd.DataFrame({"processing_days": [7.0, 7.0, 7.0, 7.0]})
    result_consistent = compute_d5_turnaround_variance(consistent_claims)
    assert result_consistent == 0.0, f"Expected 0.0, got {result_consistent}"
    print(f"D5 zero-variance test passed: {result_consistent}")

    # Edge case: fewer than 2 claims -> can't compute variance meaningfully
    single_claim = pd.DataFrame({"processing_days": [7.0]})
    result_single = compute_d5_turnaround_variance(single_claim)
    assert result_single == 0.0, f"Expected 0.0, got {result_single}"
    print(f"D5 single-claim edge case passed: {result_single}")

    # Edge case: empty window
    empty_claims = pd.DataFrame({"processing_days": []})
    result_empty = compute_d5_turnaround_variance(empty_claims)
    assert result_empty == 0.0, f"Expected 0.0, got {result_empty}"
    print(f"D5 empty-window test passed: {result_empty}")


from dimensions import (
    compute_d1_approval_rate, compute_d2_garage_concentration,
    compute_d3_threshold_proximity, compute_d4_surveyor_diversity,
    compute_d5_turnaround_variance, compute_d6_after_hours_pct
)


def test_d6_after_hours_pct():
    # Hand-built: 6 claims at hours [8, 10, 14, 18, 20, 22]
    # After-hours (< 9 or >= 18): 8 (yes), 10 (no), 14 (no), 18 (yes), 20 (yes), 22 (yes)
    # 4 out of 6 are after-hours -> 4/6 = 0.6667
    claims = pd.DataFrame({"approval_hour": [8, 10, 14, 18, 20, 22]})
    result = compute_d6_after_hours_pct(claims)
    expected = 4 / 6
    assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"
    print(f"D6 test passed: {result}")

    # Edge case: boundary hours exactly at 9 and 18
    # hour=9 -> NOT after-hours (9 is within working hours, 9 AM start)
    # hour=18 -> IS after-hours (18 = 6 PM, working hours END at 6PM, so 18 counts as after)
    boundary_claims = pd.DataFrame({"approval_hour": [9, 18]})
    result_boundary = compute_d6_after_hours_pct(boundary_claims)
    expected_boundary = 1 / 2  # only hour=18 counts
    assert abs(result_boundary - expected_boundary) < 1e-9, f"Expected {expected_boundary}, got {result_boundary}"
    print(f"D6 boundary test passed: {result_boundary}")

    # Edge case: empty window
    empty_claims = pd.DataFrame({"approval_hour": []})
    result_empty = compute_d6_after_hours_pct(empty_claims)
    assert result_empty == 0.0, f"Expected 0.0, got {result_empty}"
    print(f"D6 empty-window test passed: {result_empty}")


if __name__ == "__main__":
    test_d1_approval_rate()
    test_d2_garage_concentration()
    test_d3_threshold_proximity()
    test_d4_surveyor_diversity()
    test_d5_turnaround_variance()
    test_d6_after_hours_pct()