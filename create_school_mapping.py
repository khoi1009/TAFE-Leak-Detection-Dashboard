"""
Create mapping between de-identified properties and real NSW schools
This script:
1. Loads the original water meter data
2. Loads NSW school GIS information
3. Randomly assigns each property to a real school
4. Creates a new Excel file with enriched data including school details
5. Creates a mapping CSV for reference
"""

import pandas as pd
import json
import os
import random
from datetime import datetime

# Paths
WATER_DATA_PATH = r"D:\End Use Projects\NSW - TAFE Leak detection model\Data\Water Meter Data\April2024-April2025.xlsx"
GIS_DATA_PATH = r"D:\End Use Projects\NSW - TAFE Leak detection model\Production_Web_App\School GIS info\3e6d5f6a-055c-440d-a690-fc0537c31095.json"
OUTPUT_DATA_PATH = r"D:\End Use Projects\NSW - TAFE Leak detection model\Model for Delivery - Modular\data_with_schools.xlsx"
MAPPING_CSV_PATH = r"D:\End Use Projects\NSW - TAFE Leak detection model\Model for Delivery - Modular\property_school_mapping.csv"


def load_gis_schools():
    """Load school data from GIS JSON file"""
    print("Loading NSW school GIS data...")

    with open(GIS_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    fields = data.get("fields", [])
    records = data.get("records", [])

    # Create field index mapping
    field_map = {field["id"]: idx for idx, field in enumerate(fields)}

    schools = []
    for record in records:
        try:
            lat_str = record[field_map.get("Latitude", 41)]
            lng_str = record[field_map.get("Longitude", 42)]

            if not lat_str or not lng_str or lat_str == "np" or lng_str == "np":
                continue

            lat = float(lat_str)
            lng = float(lng_str)

            # Validate coordinates are in Australia
            if not (-44 < lat < -10 and 110 < lng < 160):
                continue

            school = {
                "school_code": record[field_map.get("School_code", 0)] or "",
                "school_name": record[field_map.get("School_name", 2)]
                or "Unknown School",
                "street": record[field_map.get("Street", 3)] or "",
                "suburb": record[field_map.get("Town_suburb", 4)] or "",
                "postcode": record[field_map.get("Postcode", 5)] or "",
                "phone": record[field_map.get("Phone", 6)] or "",
                "email": record[field_map.get("School_Email", 7)] or "",
                "enrolment": record[field_map.get("latest_year_enrolment_FTE", 10)]
                or "",
                "school_type": record[field_map.get("Level_of_schooling", 14)] or "",
                "region": record[field_map.get("Operational_directorate", 31)] or "",
                "network": record[field_map.get("Principal_network", 32)] or "",
                "lga": record[field_map.get("LGA", 26)] or "",
                "remoteness": record[field_map.get("ASGS_remoteness", 40)] or "",
                "latitude": lat,
                "longitude": lng,
            }
            schools.append(school)

        except (ValueError, IndexError, TypeError):
            continue

    print(f"  Loaded {len(schools)} schools with valid coordinates")
    return schools


def get_all_properties():
    """Get all unique property names from water meter data"""
    print("Loading water meter data...")

    xl = pd.ExcelFile(WATER_DATA_PATH)
    all_properties = set()

    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet)
        if "De-identified Property Name" in df.columns:
            all_properties.update(df["De-identified Property Name"].unique())

    properties = sorted(list(all_properties))
    print(f"  Found {len(properties)} unique properties")
    return properties


def create_property_school_mapping(properties, schools):
    """
    Randomly assign each property to a real school
    Uses a fixed seed for reproducibility
    """
    print("Creating property-to-school mapping...")

    # Use fixed seed for reproducible mapping
    random.seed(42)

    # Randomly sample schools for each property
    # If more properties than schools, allow repeats
    if len(properties) <= len(schools):
        selected_schools = random.sample(schools, len(properties))
    else:
        # More properties than schools - use all schools and repeat some
        selected_schools = schools.copy()
        while len(selected_schools) < len(properties):
            selected_schools.extend(
                random.sample(
                    schools, min(len(schools), len(properties) - len(selected_schools))
                )
            )
        selected_schools = selected_schools[: len(properties)]
        random.shuffle(selected_schools)

    # Create mapping
    mapping = {}
    for prop, school in zip(properties, selected_schools):
        mapping[prop] = school

    print(f"  Mapped {len(mapping)} properties to schools")
    return mapping


def create_enriched_data(mapping):
    """
    Create a new Excel file with only latitude and longitude added
    Uses fast merge operation - only 2 extra columns!
    """
    print("Creating enriched data file...")

    # Convert mapping to DataFrame with ONLY lat/lng for fast merge
    mapping_df = pd.DataFrame(
        [
            {
                "De-identified Property Name": prop,
                "latitude": school["latitude"],
                "longitude": school["longitude"],
            }
            for prop, school in mapping.items()
        ]
    )

    xl = pd.ExcelFile(WATER_DATA_PATH)

    # Create a writer for the new Excel file
    with pd.ExcelWriter(OUTPUT_DATA_PATH, engine="openpyxl") as writer:
        for sheet_name in xl.sheet_names:
            print(f"  Processing sheet: {sheet_name}")
            df = pd.read_excel(xl, sheet_name=sheet_name)

            # Fast merge - only adds latitude and longitude columns
            if "De-identified Property Name" in df.columns:
                df = df.merge(mapping_df, on="De-identified Property Name", how="left")

            # Write to new Excel file
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"  Saved enriched data to: {OUTPUT_DATA_PATH}")


def save_mapping_csv(mapping):
    """Save the property-to-school mapping as CSV for reference"""
    print("Saving mapping CSV...")

    rows = []
    for prop, school in mapping.items():
        rows.append(
            {
                "property_id": prop,
                "school_code": school["school_code"],
                "school_name": school["school_name"],
                "suburb": school["suburb"],
                "postcode": school["postcode"],
                "region": school["region"],
                "latitude": school["latitude"],
                "longitude": school["longitude"],
                "school_type": school["school_type"],
                "enrolment": school["enrolment"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(MAPPING_CSV_PATH, index=False)
    print(f"  Saved mapping to: {MAPPING_CSV_PATH}")

    # Print sample mapping
    print("\nSample property-to-school mapping:")
    print("-" * 80)
    for i, row in enumerate(rows[:10]):
        print(
            f"  {row['property_id']:20} → {row['school_name'][:40]:40} ({row['suburb']})"
        )
    print("  ...")
    print(f"  Total: {len(rows)} mappings")


def main():
    print("=" * 60)
    print("Property to School Mapping Creator")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load data
    schools = load_gis_schools()
    properties = get_all_properties()

    # Create mapping
    mapping = create_property_school_mapping(properties, schools)

    # Save mapping CSV
    save_mapping_csv(mapping)

    # Create enriched Excel file
    create_enriched_data(mapping)

    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. {OUTPUT_DATA_PATH}")
    print(f"  2. {MAPPING_CSV_PATH}")
    print(f"\nTo use the new data in the dashboard:")
    print(f"  Update config_leak_detection.yml: data_path: data_with_schools.xlsx")


if __name__ == "__main__":
    main()
