# si-library
One-stop library for panel data.

## Preprocessing

**Input of preprocessing step**

An Excel sheet with

- Sheet 1 named "Data"
  - first row: “unit”, “time”, “intervention”, text descriptor for outcome variable 1, text descriptor for outcome variable 2, text descriptor for outcome variable 3, …
  - starting second row: unit UID, time UID, intervention UID, value of outcome variable 1, value of outcome variable 2, value of outcome variable 3, ....
- Sheet 2 named "Unit covariates"
  - first row: “unit”, text descriptor for unit covariate 1, text descriptor for unit covariate 2
  - second row: unit UID, unit covariate 1, unit covariate 2, ....
- Sheet 3 named "Time covariates"
  - first row: “time”, text descriptor for time covariate 1, text descriptor for time covariate 2
  - second row: time UID, time covariate 1, time covariate 2, ....
- Sheet 4 named "Intervention covariates"
  - first row: “intervention”, text descriptor for intervention covariate 1, text descriptor for intervention covariate 2
  - second row: intervention UID, intervention covariate 1, intervention covariate 2, ....

**Output of preprocessing step**

(N: number of units, T: number of timestamps, M: number of unique outcome variables, D: number of unique identifications of interventions)

- (N × T x D x M) tensor of units x timestamps x outcomes x interventions for each sample
- optional dictionary of unit id : covariate
- optional dictionary of timestamp id : covariate
- optional dictionary of intervention id : covariate

