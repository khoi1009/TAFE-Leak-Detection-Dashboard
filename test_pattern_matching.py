"""Test pattern matching functionality"""

import logging

logging.disable(logging.CRITICAL)

from false_alarm_patterns import (
    get_patterns_df,
    match_incident_to_patterns,
    check_should_suppress,
    create_signal_fingerprint,
    SIGNAL_MATCH_THRESHOLD,
)

print("=" * 60)
print("TEST 1: RECORDED PATTERNS")
print("=" * 60)

df = get_patterns_df()
print(f"Total patterns: {len(df)}")

for _, p in df.iterrows():
    print(f"\nPattern ID: {p['pattern_id']}")
    print(f"Site: {p['site_id']}")
    print(f"Auto-suppress: {p['auto_suppress']}")
    print(f"Confidence: {p['confidence']}")
    fp = p["signal_fingerprint"]
    print(f"Signals: {fp.get('signals_active', [])}")
    print(f"Volume range: {fp.get('volume_range')}")
    print(f"MNF range: {fp.get('mnf_range')}")

print("\n" + "=" * 60)
print("TEST 2: CREATE TEST INCIDENT")
print("=" * 60)

# Simulate an incident similar to the recorded pattern
test_incident = {
    "event_id": "TEST_INCIDENT_001",
    "site_id": "Property 11127",
    "subscores_ui": {
        "MNF": 0.8,
        "RESIDUAL": 1.0,
        "CUSUM": 1.0,
        "AFTERHRS": 1.0,
        "BURSTBF": 1.0,
    },
    "volume_kL": 20.64,
    "duration_hours": 96.0,
    "start_day": "2024-12-02",
}

print(f"Incident signals: {test_incident['subscores_ui']}")
print(f"Incident volume: {test_incident['volume_kL']} kL")

fp = create_signal_fingerprint(test_incident)
print(f"\nGenerated fingerprint:")
print(f"  Signals: {fp.get('signals_active', [])}")
print(f"  Volume range: {fp.get('volume_range')}")

print("\n" + "=" * 60)
print("TEST 3: PATTERN MATCHING")
print(f"(Threshold for strong match: {SIGNAL_MATCH_THRESHOLD:.0%})")
print("=" * 60)

matches = match_incident_to_patterns(test_incident, "Property 11127")
print(f"\nFound {len(matches)} match(es)")

for m in matches:
    print(f"\n  Pattern: {m['pattern_id']}")
    print(f"  Signal similarity: {m['signal_similarity']:.1%}")
    print(f"  Time similarity: {m['time_similarity']:.1%}")
    print(f"  Combined score: {m['combined_score']:.1%}")
    print(f"  Pattern confidence: {m['pattern_confidence']:.1%}")
    print(f"  Final score: {m['final_score']:.1%}")
    print(f"  Is strong match: {m['is_strong_match']}")

print("\n" + "=" * 60)
print("TEST 4: AUTO-SUPPRESS CHECK")
print("=" * 60)

should_suppress, matching = check_should_suppress(test_incident, "Property 11127")
print(f"\nShould auto-suppress: {should_suppress}")
if matching:
    print(f"Matching pattern: {matching['pattern_id']}")
else:
    print(
        "No matching pattern for auto-suppress (either no match or auto_suppress=False)"
    )

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

if matches and matches[0]["is_strong_match"]:
    print("\n✅ Pattern matching IS WORKING!")
    print("   The test incident matches the recorded pattern.")
    if not matches[0].get("auto_suppress"):
        print("\n⚠️  BUT auto_suppress=False, so incident will be FLAGGED")
        print("   but NOT suppressed. You will see '🧠 Possible False Alarm'")
        print("   indicator on the incident card.")
else:
    print("\n❌ Pattern matching is NOT finding a strong match.")
    print("   This could be due to:")
    print("   1. Signal fingerprints don't match closely enough")
    print("   2. Volume/duration ranges don't overlap")
    print("   3. Pattern confidence is too low")
