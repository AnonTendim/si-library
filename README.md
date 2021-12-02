# si-library
*Note: This project is under active development.*

## Description

This library includes the functionality of estimating counterfactual outcomes using synthetic interventions (SI). More information about SI can be found [here](https://arxiv.org/abs/2006.07691).



## Installation

Available via PyPI:

```
pip install si-library
```

Or install the git version:

```
pip install git+https://github.com/AnonTendim/si-library.git
```



## Design

### Pre-processing

**Input of preprocessing step**

An Excel sheet with

- Sheet 1 named "data"
  - first row: text discriptor for axis 1  (default: unit), axis 2 name (default: measurement), axis 3 name (default: intervention), text descriptor for outcome variable 1, text descriptor for outcome variable 2, text descriptor for outcome variable 3, …
  - starting second row: value of axis 1, value of axis 2, value of axis 3, value of outcome variable 1, value of outcome variable 2, value of outcome variable 3, ....
- Sheet 2 named "unit covariates" (optional)
  - first row: text descriptor for axis 1 (default: unit), text descriptor for axis 1 covariate 1, text descriptor for axis 1 covariate 2, ...
  - starting second row: value of axis 1, axis 1 covariate 1, axis 1 covariate 2, ....
- Sheet 3 named "time covariates" (optional)
  - first row: text descriptor for axis 2 (default: measurement), text descriptor for axis 2 covariate 1, text descriptor for axis 2 covariate 2, ...
  - starting second row: value of axis 2, axis 2 covariate 1, axis 2 covariate 2, ....
- Sheet 4 named "intervention covariates" (optional)
  - first row: text descriptor for axis 3 (default: intervention), text descriptor for axis 3 covariate 1, text descriptor for axis 3 covariate 2, ...
  - starting second row: value of axis 3, axis 3 covariate 1, axis 3 covariate 2, ....

**Output of preprocessing step**

(N: number of axis 1 elements, T: number of axis 2 elements, D: number of axis 3 elements, M: number of unique outcome variables)

- A Tensor() object
  - (N × T x D x M) Numpy array of axis 1 elements x axis 2 elements x axis 3 elements x outcomes for each sample

- optional dictionary of axis 1 id : covariate
- optional dictionary of axis 2 id : covariate
- optional dictionary of axis 3 id : covariate

**Usage**

1. (Optional) Call `define_axis_name(name_axis1 = 'new axis 1', name_axis2 = 'new axis 2', name_axis3 = 'new axis 3', name_axis4 = 'new axis 4')` to rename the axes rather than using default names. If hoping to use the default name for some axis, skip refining it.
2. Use `df, id_to_covariates = import_excel("input.xlsx")` to import the data in excel sheet.
3. Call `tensor = convert_to_tensor(df)` to convert data from dataframe to a Tensor() object as discussed above.
   1. May call `tensor.data` to look at the 4-d Numpy array data
   2. May call `tensor.print_info()` to get basic statistics of the data
   3. May call `tensor.print_table(x_axis = ..., y_axis = ..., constant = ..., entry = ..., show_data = True/False)` to get a snapshot of the data

### Estimation

1. Initiate a SyntheticIntervention class and set up the threshold and rho with `model = SyntheticIntervention(threshold, rho)`
2. Fit mode with tensor data from preprocessing step by calling `model.fit(tensor)`
3. Predict targets of interest with `predictions, tests, cis = model.predict([(axis 1 value, axis 2 value, axis 3 value, some outcome), (axis 1 value, axis 2 value, axis 3 value, some outcome), ...])`
4. enter title name and plot spectrum and call `model.plot_spectrum(4-tuple target, title)`



## Contributors

Jessy Xinyi Han

Anish Agarwal

Dennis Shen



## License

MIT license



## Contact

For any questions or bugs related to this library, feel free to contact xyhan [at] mit [dot] edu.

