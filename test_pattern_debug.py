"""Debug pattern matching issue."""

from false_alarm_patterns import get_all_patterns, match_incident_to_patterns

# Check patterns loaded
patterns = get_all_patterns()
print(f"Loaded {len(patterns)} patterns")
for p in patterns:
    print(f"  Pattern: {p.get('pattern_id')} for site {p.get('site_id')}")

# Create a test incident similar to what processing.py would see
test_inc = {
    "event_id": "Property 11127__2024-12-02__2024-12-05",
    "site_id": "Property 11127",
    "subscores_ui": {
        "MNF": 0.8,
        "RESIDUAL": 1.0,
        "CUSUM": 1.0,
        "AFTERHRS": 1.0,
        "BURSTBF": 1.0,
    },
    "start_day": "2024-12-02",
    "end_day": "2024-12-05",
    "volume_liters": 20.0,
}

# Try to match
print(f"\nTesting match for site: {test_inc['site_id']}")
matches = match_incident_to_patterns(test_inc, test_inc["site_id"])
print(f"Matches returned: {len(matches)}")
for m in matches:
    print(f"  Pattern: {m.get('pattern_id')}")
    print(f"    is_strong_match: {m.get('is_strong_match')}")
    print(f"    final_score: {m.get('final_score')}")
    print(f"    signal_similarity: {m.get('signal_similarity')}")
    print(f"    time_similarity: {m.get('time_similarity')}")
