import os
import re
from pathlib import Path

import requests

SAVE_DIR = Path("data/street_view_images")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Curated Street View vantage points for parking lots.
# Format: (lat, lng, base_heading_degrees, pitch_degrees)
# Replace these with your own manually curated points from Street View URLs.
CURATED_LOTS = [
    # @42.3727698,-71.0664534,3a,75y,161.51h,75.92t
    (42.3727698, -71.0664534, 161.51, -14.08),
    # @42.375626,-71.0401407,3a,75y,181.56h,70.12t
    (42.3756260, -71.0401407, 181.56, -19.88),
    # @42.3957311,-71.0416858,3a,75y,126.45h,82.64t
    (42.3957311, -71.0416858, 126.45, -7.36),
    # @42.3991237,-71.0716456,3a,75y,225.99h,88.21t
    (42.3991237, -71.0716456, 225.99, -1.79),
    # @42.395695,-71.0821219,3a,75y,226.83h,92.73t
    (42.3956950, -71.0821219, 226.83, 2.73),
    # @42.3909109,-71.0850492,3a,75y,284.95h,60.7t
    (42.3909109, -71.0850492, 284.95, -29.30),
    # @42.3895436,-71.117998,3a,75y,154.37h,100.7t
    (42.3895436, -71.1179980, 154.37, 10.70),
    # @42.389965,-71.1414532,3a,75y,238.43h,79.17t
    (42.3899650, -71.1414532, 238.43, -10.83),
    # @42.3768226,-71.1184719,3a,75y,132.67h,71.95t
    (42.3768226, -71.1184719, 132.67, -18.05),
    # @42.3897758,-71.1431295,3a,75y,83.59h,80.54t
    (42.3897758, -71.1431295, 83.59, -9.46),
    # @42.4095522,-71.0870545,3a,75y,129.18h,97.94t
    (42.4095522, -71.0870545, 129.18, 7.94),
    # @42.4243255,-71.0705061,3a,75y,175.48h,85.29t
    (42.4243255, -71.0705061, 175.48, -4.71),
    # @42.3538769,-71.2003191,3a,75y,244.05h,85.56t
    (42.3538769, -71.2003191, 244.05, -4.44),
    # @42.3667301,-71.2148168,3a,75y,287.12h,95.35t
    (42.3667301, -71.2148168, 287.12, 5.35),
    # @42.3692844,-71.2203899,3a,75y,281.85h,83.04t
    (42.3692844, -71.2203899, 281.85, -6.96),
    # @42.3810717,-71.2618634,3a,75y,90.02h,82.55t
    (42.3810717, -71.2618634, 90.02, -7.45),
    # @42.389477,-71.2584927,3a,75y,111.9h,82.13t
    (42.3894770, -71.2584927, 111.90, -7.87),
    # @42.4526646,-71.2325704,3a,75y,127.47h,72.13t
    (42.4526646, -71.2325704, 127.47, -17.87),
    # @42.4844636,-71.2117711,3a,75y,148.06h,68.34t
    (42.4844636, -71.2117711, 148.06, -21.66),
    # @42.4855571,-71.2184112,3a,75y,188.84h,98.08t
    (42.4855571, -71.2184112, 188.84, 8.08),
    # @42.4857842,-71.1931608,3a,75y,71.39h,87.3t
    (42.4857842, -71.1931608, 71.39, -2.70),
    # @42.4837592,-71.1866356,3a,75y,27.62h,79.72t
    (42.4837592, -71.1866356, 27.62, -10.28),
    # @42.5049793,-71.1315835,3a,75y,10.76h,84.86t
    (42.5049793, -71.1315835, 10.76, -5.14),
]


def load_api_key(env_path=".env"):
    """Loads the Google Maps key from env vars or a local .env file."""
    env_key = os.getenv("MAPS_PLATFORM_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")
    if env_key:
        return env_key.strip()

    path = Path(env_path)
    if not path.exists():
        raise RuntimeError("No API key found. Set MAPS_PLATFORM_API_KEY or create .env.")

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() in {"MAPS_PLATFORM_API_KEY", "GOOGLE_MAPS_API_KEY"}:
            return value.strip().strip('"').strip("'")

    raise RuntimeError("No MAPS_PLATFORM_API_KEY or GOOGLE_MAPS_API_KEY found in .env.")


def street_view_available(api_key, lat, lng):
    """Checks if Street View imagery exists at a location."""
    metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lng}",
        "source": "outdoor",
        "key": api_key,
    }
    response = requests.get(metadata_url, params=params, timeout=20)
    response.raise_for_status()
    return response.json().get("status") == "OK"


def safe_filename(value):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_") or "parking_lot"


def download_custom_view(api_key, lat, lng, heading, pitch, fov, name):
    """Downloads a Street View image for a specific camera view."""
    if not street_view_available(api_key, lat, lng):
        return False

    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "640x640",
        "location": f"{lat},{lng}",
        "heading": f"{heading:.2f}",
        "fov": f"{fov:.2f}",
        "pitch": f"{pitch:.2f}",
        "source": "outdoor",
        "key": api_key,
    }

    img_response = requests.get(base_url, params=params, timeout=30)
    if img_response.status_code != 200:
        return False

    output_path = SAVE_DIR / f"{safe_filename(name)}.jpg"
    output_path.write_bytes(img_response.content)
    return True


def main(target_count=100):
    api_key = load_api_key()
    if not CURATED_LOTS:
        print("No curated lots defined. Add entries to CURATED_LOTS and rerun.")
        return

    lot_count = len(CURATED_LOTS)
    images_per_lot = max(1, target_count // lot_count)
    remainder = target_count % lot_count

    downloaded = 0
    for lot_index, (lat, lng, base_heading, pitch) in enumerate(CURATED_LOTS):
        shots_for_this_lot = images_per_lot + (1 if lot_index < remainder else 0)
        center_offset = (shots_for_this_lot - 1) / 2.0

        for shot_index in range(shots_for_this_lot):
            heading = (base_heading + (shot_index - center_offset) * 5.0) % 360.0
            fov = max(55.0, min(95.0, 92.0 - shot_index * 2.0))

            file_name = (
                f"lot_{lot_index:02d}_view_{shot_index:02d}_"
                f"{lat:.5f}_{lng:.5f}_h{heading:.1f}_p{pitch:.1f}_f{fov:.1f}"
            )

            if download_custom_view(api_key, lat, lng, heading, pitch, fov, file_name):
                downloaded += 1
                print(f"Downloaded {downloaded}/{target_count}: {file_name}")

            if downloaded >= target_count:
                break

        if downloaded >= target_count:
            break

    print(f"Done. Downloaded {downloaded} Street View images to {SAVE_DIR}.")


if __name__ == "__main__":
    main(target_count=100)