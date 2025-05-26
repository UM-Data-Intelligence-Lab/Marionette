# Istanbul Dataset Description

## Overview
This document describes the structure and key components of the Istanbul dataset, including the data fields and their meanings. The dataset includes check-in sequences and contextual conditions, with specific details about time, location, and event-based conditions.

## Data Structure
The dataset is organized as follows:

- **`sequences`**: Contains check-in sequences.
- **`num_marks`**: The total number of categories for points of interest (POIs).
- **`num_pois`**: The total number of unique POIs.
- **`poi_gps`**: A dictionary mapping each POI to its GPS coordinates.
- **`poi_category`**: A dictionary mapping each POI to its category.
- **`t_max`**: The maximum time observed in a sequence.

## Check-in Sequence Fields
Each check-in sequence contains the following fields:

- **`arrival_times`**: A sequence of timestamps for check-ins. The hour of the day can be directly derived from this field.
- **`marks`**: A sequence of categories corresponding to each check-in.
- **`checkins`**: A sequence of POIs visited during the check-ins.
- **`gps`**: A sequence of GPS coordinates corresponding to the check-in POIs.
- **`condition1`**: Represents the day of the week. The value from `25` to `31` represent from Monday to Sunday.
- **`condition2`**: Represents the Ramadan period. 
  - `32`: Not during Ramadan.
  - `33`: During Ramadan.
- **`condition3`**: Represents public holidays.
  - `34`: Not during a holiday period.
  - `35`: During a holiday period.
- **`condition4`, `condition5`, `condition6`**: Not used in the Istanbul dataset. These fields have identical values for all sequences.
- **`condition_indicator`**: A vector of size equal to the number of segments (24 in this dataset), where each segment corresponds to the finest granularity of partial contexts. This indicator is used for context alignment to specify conditions for each sampled timestamp. For further details, refer to Section 4.1.1.

## Notes
- The fields `condition3`, `condition4`, and `condition5` are not utilized in the Istanbul dataset and are assigned constant values across all sequences.
- The hour of the day is derived directly from `arrival_times` in the code, and encoded from 1~24, 0 is padding value for contexts.