# Topology-Preserving Polygon Simplification using SEA-DP

This project compares the standard independent **Douglas-Peucker polygon simplification** method with a proposed **Shared-Edge-Aware Douglas-Peucker (SEA-DP)** method.

The goal of SEA-DP is to simplify polygon datasets while preserving shared boundaries between adjacent polygons. Instead of simplifying each polygon independently, SEA-DP detects shared boundary chains, simplifies each shared edge only once, and applies the same simplified geometry to neighboring polygons.

This helps reduce topological errors such as gaps, overlaps, and mismatched borders.

---

## Features

* Implements standard Douglas-Peucker simplification
* Implements Shared-Edge-Aware Douglas-Peucker simplification
* Detects and processes shared polygon boundaries
* Preserves consistency between adjacent polygons
* Evaluates simplification quality using geometric and topological metrics
* Supports experiments on multiple shapefile datasets

---

## Main Metrics

The project evaluates simplification results using the following metrics:

| Metric                            | Description                                                                    |
| --------------------------------- | ------------------------------------------------------------------------------ |
| **Topological Error Count (TEC)** | Counts topology-related issues such as gaps, overlaps, and boundary mismatches |
| **Shared-Edge Consistency (SEC)** | Measures whether shared boundaries remain consistent after simplification      |
| **Hausdorff Distance (HD)**       | Measures geometric difference between original and simplified polygons         |
| **Vertex Reduction Ratio (VRR)**  | Measures how many vertices were removed after simplification                   |
| **Execution Time**                | Measures runtime performance of each method                                    |

---

## Project Structure

```text
codes/
  main algo/
    sea_dp.py

  src/
    evaluation_metrics.py
    test-usa.py
    test-armenia-azerbaijan-seadp.py
    test-rizal-laguna.py
    free-test-gui.py

data/
  raw/
    natural_earth/
      USA/
      Armenia-Azerbaijan/
    GADM/
      philippines/
```

---

## Data

Raw shapefiles are **not included** in this repository because of file size and licensing considerations.

Place datasets manually inside:

```text
data/raw/
```

Expected dataset locations:

```text
data/raw/natural_earth/USA/
data/raw/natural_earth/Armenia-Azerbaijan/
data/raw/GADM/philippines/
```

Make sure each dataset folder contains the required shapefile components, such as:

```text
.shp
.shx
.dbf
.prj
```

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd douglas_peucker_thesis
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

Run the USA test:

```bash
python codes/src/test-usa.py
```

Run the Armenia-Azerbaijan shared-boundary test:

```bash
python codes/src/test-armenia-azerbaijan-seadp.py
```

Run the Rizal-Laguna test:

```bash
python codes/src/test-rizal-laguna.py
```

Run the GUI test tool:

```bash
python codes/src/free-test-gui.py
```

---

## Using VS Code

Open the project folder in VS Code.

Then open a new terminal:

```text
Terminal > New Terminal
```

Make sure the terminal is opened in the project root folder. It should look similar to:

```powershell
PS D:\Code\douglas_peucker_thesis>
```

From there, install dependencies and run the test scripts.

---

## Method Overview

### Standard Douglas-Peucker

The standard Douglas-Peucker method simplifies each polygon independently. While this reduces vertex count, it can create inconsistencies between adjacent polygons because shared boundaries may be simplified differently.

### Shared-Edge-Aware Douglas-Peucker

SEA-DP improves this process by:

1. Detecting shared boundaries between adjacent polygons
2. Extracting shared boundary chains
3. Simplifying each shared boundary only once
4. Applying the same simplified boundary to both neighboring polygons
5. Evaluating the output using topology and geometry metrics

This approach helps preserve shared borders and reduces topology-related errors.

---

## Example Workflow

```bash
pip install -r requirements.txt

python codes/src/test-usa.py
python codes/src/test-armenia-azerbaijan-seadp.py
python codes/src/test-rizal-laguna.py
python codes/src/free-test-gui.py
```

---

## Notes

* Use valid polygon shapefiles as input.
* Make sure the coordinate reference system is appropriate for distance-based simplification.
* Raw datasets are excluded from the repository.
* Results may vary depending on tolerance values and dataset complexity.

---

## Project Title

**Topology-Preserving Polygon Simplification using Douglas-Peucker for Shared Boundaries**
