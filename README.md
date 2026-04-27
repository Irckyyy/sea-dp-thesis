# Topology-Preserving Polygon Simplification using SEA-DP

This project compares standard independent Douglas-Peucker polygon simplification with a Shared-Edge-Aware Douglas-Peucker (SEA-DP) method.

SEA-DP detects shared polygon boundaries, simplifies each shared boundary once, and propagates the same simplified geometry to adjacent polygons.

## Main Metrics

- Topological Error Count (TEC)
- Shared-Edge Consistency (SEC)
- Hausdorff Distance (HD)
- Vertex Reduction Ratio (VRR)
- Execution Time

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

## Data

Raw shapefiles are not included in this repository.

Place datasets manually in:

data/raw/

Expected examples:

data/raw/natural_earth/USA/
data/raw/natural_earth/Armenia-Azerbaijan/
data/raw/GADM/philippines/
Run
pip install -r requirements.txt
python codes/src/test-usa.py
python codes/src/test-armenia-azerbaijan-seadp.py
python codes/src/free-test-gui.py

---

## 5. Open VS Code terminal

In VS Code:

```text
Terminal > New Terminal

Make sure it says something like:

PS D:\Code\douglas_peucker_thesis>