"""
Create Demo Data for Fast Dashboard Loading

This script creates a small subset of data for demonstration purposes:
- 5 properties (including Property 11127 which has leak patterns)
- 50 schools for GIS map display
- Dramatically reduces load time from ~2 minutes to ~5 seconds
"""

import os
import json
import pandas as pd
import random

# Configuration
DEMO_PROPERTIES = [
    "Property 11127",  # Must include - has leak patterns
    "Property 11053",  # Hunters Hill
    "Property 18978",  # Bondi Beach
    "Property 20998",  # Sydney Secondary
    "Property 52286",  # Kogarah
]

NUM_DEMO_SCHOOLS = 50

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_DATA_PATH = r"D:\End Use Projects\NSW - TAFE Leak detection model\Data\Water Meter Data\April2024-April2025.xlsx"
FULL_DATA_PATH = os.path.join(BASE_DIR, "data_with_schools.xlsx")
DEMO_DATA_PATH = os.path.join(BASE_DIR, "demo_data.xlsx")
FULL_MAPPING_PATH = os.path.join(BASE_DIR, "property_school_mapping.csv")
DEMO_MAPPING_PATH = os.path.join(BASE_DIR, "demo_school_mapping.csv")
GIS_JSON_PATH = r"D:\End Use Projects\NSW - TAFE Leak detection model\Production_Web_App\School GIS info\3e6d5f6a-055c-440d-a690-fc0537c31095.json"
DEMO_GIS_PATH = os.path.join(BASE_DIR, "demo_schools_gis.json")


def create_demo_water_data():
    """Create demo water meter data with only 5 properties"""
    print("\n" + "=" * 60)
    print("Creating Demo Water Meter Data")
    print("=" * 60)

    # Use the enriched data if available, otherwise use original
    source_path = (
        FULL_DATA_PATH if os.path.exists(FULL_DATA_PATH) else ORIGINAL_DATA_PATH
    )
    print(f"Source: {source_path}")

    # Read all sheets
    xlsx = pd.ExcelFile(source_path)
    sheet_names = xlsx.sheet_names
    print(f"Found {len(sheet_names)} sheets: {sheet_names}")

    with pd.ExcelWriter(DEMO_DATA_PATH, engine="openpyxl") as writer:
        total_rows = 0
        for sheet in sheet_names:
            print(f"  Processing sheet: {sheet}")
            df = pd.read_excel(xlsx, sheet_name=sheet)

            # Find the property column
            prop_col = None
            for col in df.columns:
                if "property" in str(col).lower() or "site" in str(col).lower():
                    prop_col = col
                    break

            if prop_col is None:
                # Try first column
                prop_col = df.columns[0]

            # Filter to demo properties only
            original_len = len(df)
            df_demo = df[df[prop_col].isin(DEMO_PROPERTIES)]

            if len(df_demo) == 0:
                # Try string matching
                df_demo = df[df[prop_col].astype(str).isin(DEMO_PROPERTIES)]

            print(
                f"    {original_len:,} → {len(df_demo):,} rows (filtered to {len(DEMO_PROPERTIES)} properties)"
            )

            df_demo.to_excel(writer, sheet_name=sheet, index=False)
            total_rows += len(df_demo)

    # Get file size
    size_mb = os.path.getsize(DEMO_DATA_PATH) / (1024 * 1024)
    print(f"\n✅ Demo data saved: {DEMO_DATA_PATH}")
    print(f"   Total rows: {total_rows:,}")
    print(f"   File size: {size_mb:.2f} MB")

    return DEMO_DATA_PATH


def create_demo_school_mapping():
    """Create demo school mapping with only demo properties"""
    print("\n" + "=" * 60)
    print("Creating Demo School Mapping")
    print("=" * 60)

    if not os.path.exists(FULL_MAPPING_PATH):
        print(f"Warning: Full mapping not found at {FULL_MAPPING_PATH}")
        return None

    df = pd.read_csv(FULL_MAPPING_PATH)
    df_demo = df[df["property_id"].isin(DEMO_PROPERTIES)]

    print(f"Filtered: {len(df)} → {len(df_demo)} property mappings")

    df_demo.to_csv(DEMO_MAPPING_PATH, index=False)
    print(f"✅ Demo mapping saved: {DEMO_MAPPING_PATH}")

    return DEMO_MAPPING_PATH


def create_demo_gis_data():
    """Create demo GIS data with only 50 schools"""
    print("\n" + "=" * 60)
    print("Creating Demo GIS School Data")
    print("=" * 60)

    if not os.path.exists(GIS_JSON_PATH):
        print(f"Warning: GIS data not found at {GIS_JSON_PATH}")
        return None

    with open(GIS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    fields = data.get("fields", [])
    records = data.get("records", [])

    print(f"Original: {len(records)} schools")

    # Get the school codes from demo mapping to ensure they're included
    demo_school_codes = set()
    if os.path.exists(DEMO_MAPPING_PATH):
        mapping_df = pd.read_csv(DEMO_MAPPING_PATH)
        demo_school_codes = set(mapping_df["school_code"].astype(str).tolist())
        print(f"Must include {len(demo_school_codes)} schools from demo mapping")

    # Find school_code field index
    field_map = {field["id"]: idx for idx, field in enumerate(fields)}
    school_code_idx = field_map.get("School_code", 0)
    lat_idx = field_map.get("Latitude", 41)
    lng_idx = field_map.get("Longitude", 42)

    # Filter records - prioritize mapped schools, then add random ones
    priority_records = []
    other_records = []

    for record in records:
        try:
            code = str(record[school_code_idx])
            lat = record[lat_idx]
            lng = record[lng_idx]

            # Skip records without valid coordinates
            if not lat or not lng or lat == "np" or lng == "np":
                continue

            if code in demo_school_codes:
                priority_records.append(record)
            else:
                other_records.append(record)
        except (IndexError, TypeError):
            continue

    # Select schools: all mapped + random others up to NUM_DEMO_SCHOOLS
    random.seed(42)
    remaining_slots = NUM_DEMO_SCHOOLS - len(priority_records)
    if remaining_slots > 0:
        selected_others = random.sample(
            other_records, min(remaining_slots, len(other_records))
        )
    else:
        selected_others = []

    demo_records = priority_records + selected_others

    # Create demo GIS JSON
    demo_data = {"fields": fields, "records": demo_records}

    with open(DEMO_GIS_PATH, "w", encoding="utf-8") as f:
        json.dump(demo_data, f)

    size_kb = os.path.getsize(DEMO_GIS_PATH) / 1024
    print(f"✅ Demo GIS saved: {DEMO_GIS_PATH}")
    print(
        f"   Schools: {len(demo_records)} ({len(priority_records)} mapped + {len(selected_others)} random)"
    )
    print(f"   File size: {size_kb:.1f} KB")

    return DEMO_GIS_PATH


def update_config_for_demo():
    """Update config to use demo data"""
    print("\n" + "=" * 60)
    print("Updating Configuration")
    print("=" * 60)

    config_path = os.path.join(BASE_DIR, "config_leak_detection.yml")

    if not os.path.exists(config_path):
        print(f"Warning: Config not found at {config_path}")
        return

    with open(config_path, "r") as f:
        content = f.read()

    # Check if already using demo data
    if "demo_data.xlsx" in content:
        print("Config already using demo data")
        return

    # Backup original config
    backup_path = config_path + ".backup"
    if not os.path.exists(backup_path):
        with open(backup_path, "w") as f:
            f.write(content)
        print(f"Backed up config to: {backup_path}")

    # Update data_path
    new_content = content.replace("data_with_schools.xlsx", "demo_data.xlsx")

    with open(config_path, "w") as f:
        f.write(new_content)

    print(f"✅ Config updated to use demo_data.xlsx")


def update_components_map_for_demo():
    """Update components_map.py to use demo GIS data"""
    print("\n" + "=" * 60)
    print("Updating GIS Map Component")
    print("=" * 60)

    map_path = os.path.join(BASE_DIR, "components_map.py")

    if not os.path.exists(map_path):
        print(f"Warning: components_map.py not found")
        return

    with open(map_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if already updated
    if "demo_schools_gis.json" in content:
        print("components_map.py already using demo GIS data")
        return

    # Add demo GIS path as first option
    old_paths = """    possible_paths = [
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "Production_Web_App",
            "School GIS info",
            "3e6d5f6a-055c-440d-a690-fc0537c31095.json",
        ),"""

    new_paths = """    possible_paths = [
        # Demo data (50 schools) for fast loading
        os.path.join(os.path.dirname(__file__), "demo_schools_gis.json"),
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "Production_Web_App",
            "School GIS info",
            "3e6d5f6a-055c-440d-a690-fc0537c31095.json",
        ),"""

    if old_paths in content:
        content = content.replace(old_paths, new_paths)

        with open(map_path, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ components_map.py updated to use demo GIS data first")
    else:
        print("Could not find expected path pattern in components_map.py")
        print("You may need to manually update the GIS data path")


def main():
    print("\n" + "=" * 60)
    print("DEMO DATA GENERATOR")
    print("=" * 60)
    print(f"Creating optimized demo data for fast dashboard loading")
    print(f"Properties: {len(DEMO_PROPERTIES)} ({', '.join(DEMO_PROPERTIES[:3])}...)")
    print(f"Schools: {NUM_DEMO_SCHOOLS}")

    # Create demo datasets
    create_demo_water_data()
    create_demo_school_mapping()
    create_demo_gis_data()

    # Update configuration
    update_config_for_demo()
    update_components_map_for_demo()

    print("\n" + "=" * 60)
    print("✅ DEMO DATA CREATION COMPLETE!")
    print("=" * 60)
    print(
        """
Output files:
  1. demo_data.xlsx          - Water meter data (5 properties)
  2. demo_school_mapping.csv - Property-to-school mapping
  3. demo_schools_gis.json   - GIS data (50 schools)

The config has been updated to use demo_data.xlsx.
Restart the dashboard to see the faster load times!

To switch back to full data:
  - Edit config_leak_detection.yml
  - Change data_path back to 'data_with_schools.xlsx'
  - Remove 'demo_schools_gis.json' from components_map.py paths
"""
    )


if __name__ == "__main__":
    main()
