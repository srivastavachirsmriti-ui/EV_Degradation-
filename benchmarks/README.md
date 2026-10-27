# Benchmark contract

`baseline.json` freezes the original code/artifact audit and a reconstructed
baseline. It is machine-readable test evidence, not a generated report.

The original classifier's score on all shipped records is **resubstitution**:
its original training membership is unknown. Do not describe this as test
accuracy. For a comparable baseline, clone its configuration and refit on
the recorded training partition, then score the held-out template families.
Vehicle names and numbers collapse into the same family before splitting.
TF-IDF fits on training text only. Selection must never use test labels.

The documents' Random Forest artifact and endpoint do not exist at the
starting commit. The regression baseline is therefore a reconstructed
200-tree Random Forest on the documented five inputs, and a mean predictor,
using an 80/20 split with seed 42. It is not a verified reproduction of the
PDF's historical figures. Battery health, label, cluster, predicted issue,
severity, and complaint text are excluded from regression inputs.

All 2,500 records are synthetic. There are only 25 text template families.
Scores establish behavior on this generator, not performance on real EVs.
Report absolute percentage-point changes for classification; report MAE and
RMSE in degradation percentage points. R² is not classification accuracy.
Latency is warm, sequential, local, single-record timing with one numerical
thread, excluding HTTP and cold-start model loading.

Run baseline once from the starting artifacts: `python -m ml.baseline`.
Use Git's starting commit to recover original artifacts if rerunning later.
