# Sensor Data Recovery System

Missing industrial sensor readings can leave gaps in trend analysis and downstream calculations. This Numerical Analysis semester project detects those gaps, estimates readings from nearby observations, and compares two interpolation methods using known measurements. It runs locally in Streamlit without hardware or an external service.

## Setup

Use Python 3.11 or newer with compatible package wheels. The application was verified with Python 3.12 on Windows. Run these commands from the project folder in PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address shown by Streamlit, typically `http://localhost:8501`. To run the tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

On macOS or Linux, use `.venv/bin/python` in the last three commands. With the virtual environment activated, the equivalent commands are `pip install -r requirements.txt`, `streamlit run app.py`, and `python -m pytest`.

## Dataset

The bundled `datasets/iiot_all_sensors_missing_values.csv` contains 1,000 time records and 1,147 missing readings (7.65% of 15,000 sensor cells). The missing readings were artificially introduced for the academic demonstration.

Its 15 sensor parameters are temperature, vibration, pressure, humidity, rotation speed, voltage, current, oil level, load, motor temperature, gearbox temperature, sound level, fan speed, reactive power, and active power. The file also contains the original timestamp and the machine failure classification indicator.

Upload a UTF-8 comma-separated file with a unique header, unique valid timestamps, and at least one numerical sensor column. Select its time column. ISO 8601 timestamps are recommended for uploads. The sample uses month/day/year and a 24-hour clock. Timezone offsets are respected and naive timestamps use their recorded clock as UTC. For custom uploads, use **Non-sensor columns** to exclude numeric IDs and categorical codes with custom names. Blank cells and common missing-value markers such as `NA`, `NaN`, and `null` are treated as gaps. Non-numeric or non-finite cells in an otherwise numerical sensor produce a validation message and are treated as missing. The original CSV is kept intact.

## Using the app

1. Select the sample or upload a compatible CSV. Expand **Dataset preview** to inspect the input.
2. Select a sensor. **Missing values** shows dataset-wide counts, selected-sensor counts, and the original timestamps of missing readings.
3. In **Interpolation & graphs**, choose Linear, Lagrange, or Compare both methods in the sidebar and click **Apply interpolation**. The table shows the estimates and recovery statuses. Comparison mode includes Lagrange minus Linear for readings recovered by both. The charts distinguish recorded points, estimated points, and missing positions.
4. **Accuracy evaluation** hides approximately 10% of eligible known readings for the selected sensor using a fixed seed. The earliest and latest available readings stay visible. Both methods use the same remaining observations and are scored only on shared finite estimates.
5. In **CSV export**, download the selected sensor or comparison results. To fill all sensors independently, choose a single method and click **Recover all sensors**. Downloads retain original row order and timestamp text, keep recorded values and metadata intact, and leave unrecoverable readings blank.

## Live Sensor Reading Estimator

Open **Live Sensor Reading Estimator** to work without a CSV. Choose a sensor, an interpolation method, and a time format. Enter a time and measurement with **Add Reading**, or enter just a time with **Mark Reading as Missing**. You can use decimal times or ISO date-times such as `2026-01-01 14:30`. Naive date-times use UTC. Each sensor keeps its own readings in the current browser session.

For a quick demo, add 25 at time 1 and 27 at time 2, then mark time 3 missing. The gap stays pending until you add 31 at time 4. Linear Interpolation then estimates 29 at time 3. The table and graph recalculate as you add readings or change methods. **Calculate Missing Values** also runs the calculation on demand.

Use **Edit readings** to correct a time or measurement, mark an entry missing, or remove it. **Download live readings CSV** exports the selected sensor's recorded and estimated values. **Reset Readings** clears the manual readings for all sensors in the session.

## Numerical methods

The app sorts data chronologically and converts timestamps to elapsed hours from the earliest time. Each missing reading is estimated using original known readings from the same sensor.

- **Linear interpolation** uses the closest known reading on either side of the gap. The formula is `y(x) = y₀ + ((x − x₀) / (x₁ − x₀)) × (y₁ − y₀)`. The code evaluates its equivalent weighted sum to reduce overflow risk.
- **Local Lagrange interpolation** starts with the bracketing pair and adds the nearest remaining originals, up to four points. For each anchor `i`, the basis `Lᵢ(x) = ∏ⱼ≠ᵢ (x − xⱼ) / (xᵢ − xⱼ)` is multiplied by its reading, giving `P(x) = Σᵢ yᵢ Lᵢ(x)`. Two, three, or four points give a polynomial of degree at most one, two, or three, respectively.
- **Accuracy** uses mean absolute error `MAE = Σ |actual − estimate| / n` and root mean squared error `RMSE = √(Σ (actual − estimate)² / n)`. Lower scores describe performance on this held-out sample.

Both interpolation formulas are implemented directly. Missing values outside the known range remain blank. Estimates depend on nearby observations and can be inaccurate during sudden changes or long gaps. Local polynomial estimates may overshoot. Lagrange can score higher or lower than Linear depending on the sensor and the observations used.

## Project files

- `app.py` runs the Streamlit interface and CSV downloads.
- `data_processing/` loads, validates, summarizes, and reconstructs sensor data.
- `numerical_methods/` implements the two interpolation formulas.
- `evaluation/` computes reproducible held-out metrics.
- `visualization/` draws the Matplotlib charts.
- `datasets/` contains the original IIoT CSV.
- `tests/` verifies numerical cases, validation, evaluation, and CSV integrity.
