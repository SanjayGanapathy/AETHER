# Data Description: AETHER Kestrel Movement Dataset

This document provides metadata for the processed dataset used in the AETHER manuscript, conforming to FAIR (Findable, Accessible, Interoperable, and Reusable) data principles.

## Source
- **Original Data Repository:** Movebank (available for download on Zenodo: https://zenodo.org/records/16990288)
- **Study Name:** Common Kestrel (Falco tinnunculus) breeding and non-breeding, Doñana, Spain
- **Movebank Study ID:** 2970193504
- **Principal Investigator:** Sergio Hernández-León
- **Citation:** Hernández-León S (2024) Data from: Common Kestrel (Falco tinnunculus) breeding and non-breeding, Doñana, Spain. Movebank Data Repository. doi:10.5441/001/1.h125s56c

## Processed Dataset Columns

This describes the features in the final dataset used for analysis after processing and feature engineering.

### Core GPS Features
- **timestamp:**
  - **Description:** The date and time of the GPS fix.
  - **Units:** UTC
  - **Source:** Original Movebank data.
- **individual_id:**
  - **Description:** Unique identifier for each individual bird.
  - **Units:** N/A
  - **Source:** Original Movebank data.
- **latitude:**
  - **Description:** The geographic latitude of the GPS fix.
  - **Units:** Decimal Degrees
  - **Source:** Original Movebank data.
- **longitude:**
  - **Description:** The geographic longitude of the GPS fix.
  - **Units:** Decimal Degrees
  - **Source:** Original Movebank data.
- **altitude_m:**
  - **Description:** The altitude of the bird above sea level.
  - **Units:** Meters (m)
  - **Source:** Original Movebank data.
- **heading:**
  - **Description:** The direction of travel.
  - **Units:** Degrees (0-360)
  - **Source:** Original Movebank data.
- **speed_ms:**
  - **Description:** The ground speed of the bird.
  - **Units:** Meters per second (m/s)
  - **How it was derived:** *(You need to fill this in - e.g., "Calculated by the GPS device" or if you recalculated it, explain how)*.

### GPS Quality Features (for the GPS Expert)
- **satellite_count:**
  - **Description:** Number of satellites used to determine the GPS fix.
  - **Units:** Integer
  - **Source:** Original Movebank data.
- **dop (Dilution of Precision):**
  - **Description:** A measure of the geometric quality of the satellite configuration. Lower values indicate higher precision.
  - **Units:** N/A
  - **Source:** Original Movebank data.
- **time_to_fix_s:**
  - **Description:** The time it took for the GPS device to acquire a satellite fix.
  - **Units:** Seconds (s)
  - **Source:** Original Movebank data.

### Engineered Kinematic Features (for the Kinematic Expert)
- **acceleration_ms2:**
  - **Description:** The change in speed over time.
  - **Units:** Meters per second squared (m/s²)
  - **How it was derived:** *(You need to fill this in - e.g., "Calculated as the difference in 'speed_ms' between consecutive timestamps, divided by the time difference.")*
- **turning_angle:**
  - **Description:** The change in heading between consecutive points.
  - **Units:** Degrees
  - **How it was derived:** *(You need to fill this in - e.g., "Calculated as the difference in 'heading' between consecutive timestamps.")*
