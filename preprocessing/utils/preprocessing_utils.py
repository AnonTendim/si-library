import pandas as pd
import numpy as np
import sys
from sklearn import preprocessing
from prettytable import PrettyTable

row_length = 78
axis1_name = 'unit'
axis2_name = 'measurement'
axis3_name = 'intervention'
axis4_name = 'outcome'
data_name = 'data'

def define_axis_name(name_axis1 = 'unit', name_axis2 = 'measurement', name_axis3 = 'intervention', name_axis4 = 'outcome', 
                    name_data = 'data'):
    """ User defines the axis names and data worksheet name

    Parameters:
    name_axis1: optional, default: 'unit'
    name_axis2: optional, default: 'measurement'
    name_axis3: optional, default: 'intervention'
    name_axis4: optional, default: 'outcome'
    name_data: optional, default: 'data'
    """
    global axis1_name, axis2_name, axis3_name, axis4_name, data_name
    axis1_name = name_axis1
    axis2_name = name_axis2
    axis3_name = name_axis3
    axis4_name = name_axis4
    data_name = name_data

def import_excel(filename, drop_duplicates = True):
    ''' Reads in an xlsx file

    Parameters:
    filename: name of an xlsx file
    drop_duplicates: optional flag to remove potential duplicates in data

    Returns:
    df: a dataframe that holds the data
    id_to_covariates: a dictionary that contains optional id-to-covaraites dictionaries

    Example of one id_to_covariate dict: {0: {'covariate_1': 100, 'covariate_2': 'red'}, 1: {'covariate_1': 110, 'covariate_2': 'yellow'}, 2: {'covariate_1': 10, 'covariate_2': 'red'}}
    '''
    manual = 'Please refer to https://github.mit.edu/xyhan/si-library/blob/master/README.md#preprocessing.'
    try:
        file = pd.ExcelFile(filename)
    except FileNotFoundError:
        raise ValueError("%s file not found. Please check and try again." % filename)

    id_to_covariates = {'%s covariates' % axis1_name: None, '%s covariates' % axis2_name: None, '%s axis covariates' % axis3_name: None}

    # handle error cases: worksheets
    if data_name not in file.sheet_names:
        raise ValueError('Worksheet %s not found in file. %s' % (data_name, manual))
    if len(set(file.sheet_names) - {data_name, '%s covariates' % axis1_name, '%s covariates' % axis2_name, '%s axis covariates' % axis3_name}) > 1:
        raise ValueError('Irrelevant information found in file or incorrect naming of worksheets. ' + manual)

    # read in data
    df = pd.read_excel(filename, sheet_name = data_name)

    # handle error cases: columns
    if len(df.columns) < 4:
        raise ValueError('Missing columns or incorrect naming of columns in worksheet %s. %s' % (data_name, manual))
    for item in [axis1_name, axis2_name, axis3_name]:
        if item not in df.columns:
            raise ValueError('Missing or incorrect naming of %s column in Data worksheet. %s' % (item, manual))
    for item in df.columns[3:]:
        if not pd.api.types.is_numeric_dtype(df[item]):
            raise ValueError('Non-numeric data found in column %s of worksheet %s. %s' % (item, data_name, manual))

    if drop_duplicates:
            df.drop_duplicates(inplace = True)
    df.sort_values([axis1_name, axis2_name], inplace = True)

    for name in ['%s covariates' % axis1_name, '%s covariates' % axis2_name, '%s axis covariates' % axis3_name]:
        if name in file.sheet_names:
            id_to_covariates[name] = get_covariate_dict(pd.read_excel(filename, sheet_name = name))
    return df, id_to_covariates

def get_stats(df):
    ''' Gets basic statistics of a dataframe where
    unit: list of units
    measurement: list of measurements
    intervention: list of interventions
    outcome: list of outcome variables

    *_num: number
    *_encoder: encoder
    
    Parameters:
    df: dataframe
    
    Returns:
    Stats: includes most basic statistics.
    '''
    stats = {}
    
    for axis in [axis1_name, axis2_name, axis3_name]: # unit, measurement, intervention
        stats[axis] = df[axis].astype("str").unique().tolist() # list
        stats['%s_num' % axis] = df[axis].nunique() # number
        stats['%s_encoder' % axis] = preprocessing.LabelEncoder() # encoder
        stats['%s_encoder' % axis].fit(df[axis].astype("str")) 

    stats[axis4_name] = list(df.columns[3:]) # list of outcome variables
    stats['%s_num' % axis4_name] = len(df.columns) - 3 # number of outcome variables
    stats['%s_encoder' % axis4_name] = preprocessing.LabelEncoder() # encoder of outcome variables
    stats['%s_encoder' % axis4_name].fit([str(v) for v in stats[axis4_name]])
    
    return stats

def get_covariate_dict(df):
    ''' Converts a dataframe to an id-to-covariate dictionary

    Parameters:
    df: dataframe

    Returns:
    covariate_dict: a dictionary that contains optional id-to-covaraites dictionaries
    '''
    # handle errors
    if len(df.columns) < 2:
        raise ValueError('Missing columns in worksheet. ' + manual)
    if not pd.api.types.is_integer_dtype(df.iloc[:, 0]):
        raise ValueError('Non-categorical data found in column. ' + manual)
    
    covariate_dict = {}
    covariates = df.columns[1:]
    df.apply(lambda row: fill_in_dict(row, covariate_dict, covariates), axis = 1)
    return covariate_dict

def fill_in_dict(row, covariate_dict, covariates):
    ''' Fills in a dicitionary in-place

    Parameters:
    row: a row of data from dataframe
    covariate_dict: the dict to be filled in with data
    covariates: text descriptors of covariates
    '''
    temp_dict = {}
    for i, feature in enumerate(covariates):
        temp_dict[feature] = row.iloc[i+1]
    covariate_dict[row.iloc[0]] = temp_dict

def fill_in_tensor(row, tensor, outcomes):
    ''' Fills in a tensor in-place

    Parameters:
    row: a row of data
    tensor: tensor to be filled in with data
    outcomes: text descriptors of oucomes
    '''
    for i in range(tensor.shape[-1]):
        tensor[int(row[axis1_name]), int(row[axis2_name]), int(row[axis3_name]), i] = row[outcomes[i]]

def convert_to_tensor(df):
    ''' Converts dataframe into a standard tensor format

    Parameters:
    df: dataframe

    Returns:
    tensor: a (N × T x D x M) tensor of units x timestamps x interventions x outcomes,
    missing data represented with np.nan
    '''
    stats = get_stats(df)
    df_encoded = df.copy()

    for axis in [axis1_name, axis2_name, axis3_name]: # unit, measurement, intervention
        df_encoded[axis] = stats['%s_encoder' % axis].transform(df[axis].astype("str"))

    tensor = Tensor(np.full((stats["%s_num" % axis1_name], stats["%s_num" % axis2_name], stats["%s_num" % axis3_name], stats["%s_num" % axis4_name]), np.nan), stats)
    df_encoded.apply(lambda row: fill_in_tensor(row, tensor.data, stats[axis4_name]), axis = 1)
    return tensor

def pretty_print(string):
    print(' ' * int((row_length-len(string))/2) + string + ' ' * (row_length-int((row_length-len(string))/2)))

def return_pretty_print_list(a_list):
    return "[" + ", ".join([str(item) for item in a_list]) + "]"

class Tensor:
    def __init__(self, data, stats):
        self.data = data
        self.stats = stats
        self.axis_names = [axis1_name, axis2_name, axis3_name, axis4_name]

    def print_info(self, verbose = False):
        pretty_print("Summary of Data")
        print('=' * row_length)
        for axis in [axis1_name, axis2_name, axis3_name, axis4_name]:
            print('No. %ss: %d    List of %ss: %s' % (axis, self.stats["%s_num" % axis], axis,  return_pretty_print_list(self.stats[axis])))

        for axis_name in [axis2_name, axis3_name]:
            print('=' * row_length)
            pretty_print('%ss under %ss' % (axis1_name.title(), axis_name.title()))
            for i, axis4_item in enumerate(self.stats[axis4_name]):
                print('-' * row_length)
                pretty_print(axis4_item)
                print('-' * row_length)
                axis1_items_list = []
                axis1_items_list_len = []
                for j, axis_item in enumerate(self.stats[axis_name]):
                    if axis_name == axis2_name:
                        axis1_items_list.append(self.stats['%s_encoder' % axis1_name].inverse_transform(np.where((~np.isnan(self.data[:, j, :, i])).any(axis = 1))[0]))
                    elif axis_name == axis3_name:
                        axis1_items_list.append(self.stats['%s_encoder' % axis1_name].inverse_transform(np.where((~np.isnan(self.data[:, :, j, i])).any(axis = 1))[0]))
                    axis1_items_list_len.append(len(axis1_items_list[-1]))
                    if verbose == True:
                        print('%s %s: %d %ss    List of %ss: %s' % (axis_name.title(), str(axis_item), axis1_items_list_len[-1], axis_name, axis_name,  return_pretty_print_list(axis1_items_list[-1])))
                print("Statistics of number of %ss under a %s:" % (axis1_name, axis_name))
                print('Max: %d %ss    Median: %d %ss    Min: %d %ss    Mean: %.2f %ss' % (max(axis1_items_list_len), axis1_name, np.median(axis1_items_list_len), axis1_name, min(axis1_items_list_len), axis1_name, np.mean(axis1_items_list_len), axis1_name))

    def print_table(self, x_axis, y_axis, constant, entry, show_data = False):
        if entry not in self.stats[axis4_name]:
            raise ValueError("%s %s not found in data." % (axis4_name.title(), entry))
        if x_axis not in [axis1_name, axis2_name, axis3_name] or y_axis not in [axis1_name, axis2_name, axis3_name]:
            raise ValueError("Axis must be %s, %s or %s." % (axis1_name, axis2_name, axis3_name))
        if x_axis == y_axis:
            raise ValueError("X axis and Y axis cannot be the same.")
        constant_axis_name = list(set([axis1_name, axis2_name, axis3_name])-set([x_axis, y_axis]))[0]
        if constant not in self.stats[constant_axis_name]:
            raise ValueError("Constant not found in column %s." % constant_axis_name)

        print("Under %s %s and %s %s (X axis: %s, Y axis: %s)" % (axis4_name.title(), entry, constant_axis_name[:-1], constant, x_axis, y_axis))

        if constant_axis_name == axis1_name:
            results = self.data[self.stats['%s_encoder' % constant_axis_name].transform([constant])[0], :, :, self.stats[axis4_name].index(entry)].copy()
        elif constant_axis_name == axis2_name:
            results = self.data[:, self.stats['%s_encoder' % constant_axis_name].transform([constant])[0], :, self.stats[axis4_name].index(entry)].copy()
        elif constant_axis_name == axis3_name:
            results = self.data[:, :, self.stats['%s_encoder' % constant_axis_name].transform([constant])[0], self.stats[axis4_name].index(entry)].copy()
        
        if [axis1_name, axis2_name, axis3_name].index(x_axis) < [axis1_name, axis2_name, axis3_name].index(y_axis):
            results = results.T

        firstrow = self.stats[x_axis]
        firstcol = self.stats[y_axis]

        # reverse order of rows and columns so that sorting based on scores gives ascending order
        col_indices = range(len(firstrow))
        row_indices = range(len(firstcol))
        firstrow, col_indices = zip(*sorted(zip(firstrow, col_indices), reverse = True))
        firstcol, row_indices = zip(*sorted(zip(firstcol, row_indices), reverse = True))
        results = results[:, col_indices]
        results = results[row_indices, :]

        # sort rows and columns based on number of filled entries descendingly
        col_indices = range(len(firstrow))
        row_indices = range(len(firstcol))
        col_scores = np.count_nonzero(~np.isnan(results), axis = 0)
        row_scores = np.count_nonzero(~np.isnan(results), axis = 1)
        col_scores, col_indices, firstrow = zip(*sorted(zip(col_scores, col_indices, firstrow), reverse = True))
        row_scores, row_indices, firstcol = zip(*sorted(zip(row_scores, row_indices, firstcol), reverse = True))
        
        results = results[:, col_indices]
        results = results[row_indices, :]
        firstrow = [' '] + list(firstrow)
        x = PrettyTable(firstrow)
        if show_data:
            for i in range(results.shape[0]):
                x.add_row([firstcol[i]] + [' ' if np.isnan(x) else x for x in results[i, :]]) 
        else:
            for i in range(results.shape[0]):
                x.add_row([firstcol[i]] + [' ' if np.isnan(x) else '*' for x in results[i, :]])
        print(x)

def get_tensor_time_slice(tensor, start_time, end_time, outcome):
    ''' Gets data in tensor within specificed timeframe [start_time, end_time)

    Parameters:
    tensor: data
    start_time: start time_UID of the sliced data
    end_time: end time_UID (not included) of the sliced data
    outcome: index of outcome variable

    Returns:
    a (N × T x D) tensor where
    N: number of unique units, 
    T: number of unique timestamps within [start_time, end_time)
    D: number of unique interventions,

    '''
    return tensor[:, start_time:end_time, :, outcome]


def get_tensor_intervention_slice(tensor, intervention, outcome):
    ''' Gets data in tensor with specified intervention

    Parameters:
    tensor: data
    intervention: intervention UID
    outcome: index of outcome variable

    Returns:
    a (N × T) matrix where
    N: number of unique units, 
    T: number of unique timestamps
    '''
    return tensor[:, :, intervention, outcome]

def get_tensor_intervention_slices(tensor, outcome):
    ''' Gets data in tensor with each possible intervention

    Parameters:
    tensor: data
    intervention: intervention UID
    outcome_name: name specifier of one outcome variable

    Returns:
    a (D x N × T) matrix where
    D: number of unique interventions
    N: number of unique units, 
    T: number of unique timestamps
    '''
    intervention_slices = []
    for intervention in range(tensor.shape[2]):
        intervention_slices.append(get_tensor_intervention_slice(tensor, intervention, outcome))
    return np.array(intervention_slices)


def get_tensor_time_intervention_slice(tensor, start_time, end_time, intervention, outcome):
    ''' Gets data in tensor within specificed timeframe [start_time, end_time)

    Parameters:
    tensor: data
    start_time: start date of the sliced data
    end_time: end date (not included) of the sliced data
    intervention: intervention UID
    outcome: index of the outcome variable

    Returns:
    a (N × T) tensor where
    N: number of unique units, 
    T: number of unique timestamps within [start_time, end_time)
    '''
    return tensor[:, start_time:end_time, intervention, outcome]


def get_tensor_time_intervention_slices(tensor, start_time, end_time, outcome):
    ''' Gets data in tensor within specificed timeframe [start_time, end_time)

    Parameters:
    tensor: data
    start_time: start date of the sliced data
    end_time: end date (not included) of the sliced data
    outcome: name specifier of the outcome variable

    Returns:
    a (D x N × T) tensor where
    D: number of unique interventions
    N: number of unique units, 
    T: number of unique timestamps within [start_time, end_time)
    '''
    time_intervention_slices = []
    for intervention in range(tensor.shape[2]):
        time_intervention_slices.append(get_tensor_time_intervention_slice(tensor, start_time, end_time, intervention, outcome))
    return np.array(time_intervention_slices)

