import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from itertools import combinations
import sys

row_length = 78
axis1_name = 'unit'
axis2_name = 'measurement'
axis3_name = 'intervention'
axis4_name = 'outcome'

'''
Usage
model = SyntheticIntervention(threshold)
model.initiate(tensor)
prediction = model.predict(unit_ID, intervention_ID, timestamp_ID, outcome_ID)
model.plot_spectrum()
print(model.summary(unit_ID, intervention_ID, potential_outcome_metric, testing_error_metric))
'''
class SyntheticIntervention:
    '''
    A class of a model using synthetic intervention.

    Attributes:
        num_units_
        num_interventions_
        num_timestamps_
        num_outcomes_
        target_to_donor


    Methods:
        fit(unit, intervention, timestamp, outcome)
        predict(unit, intervention, timestamp, outcome)
        plot_spectrum(title)
        summary([unit, intervention, potential_outcome_metric, testing_error_metric])

    '''
    def __init__(self, threshold = 0.99, rho = 0.05):
        self.threshold = threshold
        self.rho = rho
        self.target_to_donor = dict()

    def fit(self, tensor):
        '''
        Store data and initiate some parameters.

        Parameter: tensor: N x T x D x M tensor
        '''
        global axis1_name, axis2_name, axis3_name, axis4_name
        axis1_name = tensor.axis_names[0]
        axis2_name = tensor.axis_names[1]
        axis3_name = tensor.axis_names[2]
        axis4_name = tensor.axis_names[3]
        self.tensor_obj = tensor
        self.num_units_ = tensor.data.shape[0]
        self.num_timestamps_ = tensor.data.shape[1]
        self.num_interventions_ = tensor.data.shape[2]
        self.num_outcomes_ = tensor.data.shape[3]

    def predict(self, utio_tuples):
        '''
        Predict the value at (unit, intervention, timestamp, outcome) of the tensor

        Parameters: 
            unit: unit ID
            intervention: intervention ID
            timestamp: timestamp ID
            outcome: outcome ID
            threshold: threshold of spectral energy

        Returns: {(unit, timestamp, intervention, outcome): predicted value}
        '''
        result = dict()
        test = dict()
        ci = dict()
        for (axis1_item, axis2_item, axis3_item, axis4_item) in utio_tuples:
            axis1_index = self.tensor_obj.stats['%s_encoder' % axis1_name].get(axis1_item)
            axis2_index = self.tensor_obj.stats['%s_encoder' % axis2_name].get(axis2_item)
            axis3_index = self.tensor_obj.stats['%s_encoder' % axis3_name].get(axis3_item)
            axis4_index = self.tensor_obj.stats['%s_encoder' % axis4_name].get(axis4_item)
            print("Target (%s = %s, %s = %s, %s = %s, %s = %s):" % (axis1_name, axis1_item, axis2_name, 
                axis2_item, axis3_name, axis3_item, axis4_name, axis4_item))
            if (axis1_item, axis2_item, axis3_item, axis4_item) in self.target_to_donor:
                X_train, y_train, X_test = self.target_to_donor[(axis1_index, axis2_index, axis3_index, axis4_index)]
            else:
                tensor = self.tensor_obj.data[:, :, :, axis4_index].copy()
                tensor[axis1_index, axis2_index, axis3_index] = np.nan

                X_train, y_train, X_test = get_donor_tensor(self.tensor_obj.stats, tensor, axis1_index, axis2_index, axis3_index)
                self.target_to_donor[(axis1_item, axis2_item, axis3_item, axis4_item)] = [X_train, y_train, X_test]
            if X_train is None:
                print("(%s = %s, %s = %s, %s = %s, %s = %s): Data not sufficient for prediction" \
                    % (axis1_name, axis1_item, axis2_name, axis2_item, axis3_name, axis3_item, axis4_name, axis4_item))
                print("-"*78)
                continue
            u_train, s_train, v_train, k_train = choose_rank(X_train, t = self.threshold)
            beta = get_linear_coef(u_train, s_train, v_train, k_train, y_train)
            
            result[(axis1_item, axis2_item, axis3_item, axis4_item)] = beta @ X_test              
            interval = get_ci(beta, y_train, X_train)
            ci[(axis1_item, axis2_item, axis3_item, axis4_item)] = (result[(axis1_item, axis2_item, axis3_item, axis4_item)]-interval, result[(axis1_item, axis2_item, axis3_item, axis4_item)]+interval)
            test[(axis1_item, axis2_item, axis3_item, axis4_item)] = subspace_inclusion_hypothesis_test(X_train, X_test, self.threshold, self.threshold, self.rho)
            if test[(axis1_item, axis2_item, axis3_item, axis4_item)] == 'pass':
                print("Subspace inclusion test passed!")
            else:
                print("Subspace inclusion test failed!")
            if np.isnan(self.tensor_obj.data[axis1_index, axis2_index, axis3_index, axis4_index]):
                print("(%s = %s, %s = %s, %s = %s, %s = %s): %.3f [%.3f, %.3f] (counterfactual prediction)" \
                    % (axis1_name, axis1_item, axis2_name, axis2_item, axis3_name, axis3_item, axis4_name, axis4_item, 
                        result[(axis1_item, axis2_item, axis3_item, axis4_item)], ci[(axis1_item, axis2_item, axis3_item, axis4_item)][0], 
                        ci[(axis1_item, axis2_item, axis3_item, axis4_item)][1]))
                plot(y_train, beta @ X_train, beta @ X_test, True, 
                    y_text = axis4_item, title = "\nTarget %s = %s, %s = %s, %s = %s, %s = %s" % (axis1_name, axis1_item, axis2_name, axis2_item, axis3_name, axis3_item, axis4_name, axis4_item))
            else:
                print("(%s = %s, %s = %s, %s = %s, %s = %s): %.3f [%.3f, %.3f] (factual prediction) (%.3f (reality))" \
                    % (axis1_name, axis1_item, axis2_name, axis2_item, axis3_name, axis3_item, axis4_name, axis4_item, 
                        result[(axis1_item, axis2_item, axis3_item, axis4_item)], ci[(axis1_item, axis2_item, axis3_item, axis4_item)][0], 
                        ci[(axis1_item, axis2_item, axis3_item, axis4_item)][1], 
                        self.tensor_obj.data[axis1_index, axis2_index, axis3_index, axis4_index]))
                plot(y_train, beta @ X_train, beta @ X_test, False, y_test = result[(axis1_item, axis2_item, axis3_item, axis4_item)], 
                    y_text = axis4_item, title = "\nTarget %s = %s, %s = %s, %s = %s, %s = %s" % (axis1_name, axis1_item, axis2_name, axis2_item, axis3_name, axis3_item, axis4_name, axis4_item))
            print("-"*78)

        return result, test, ci

    def plot_spectrum(self, target, title = "Placeholder"):
        ''' Plots singular values and spectral energy
    
        Parameters:
        title: title

        Returns:
        Plots of singular values and spectral energy
        '''
        if target in self.target_to_donor:
            X_train, _, _ = self.target_to_donor[target]
        else:
            _, _, _ = self.predict([target])
            X_train, _, _ = self.target_to_donor[target]
            if X_train is None:
                print("No sufficient data for plotting.")
                return
        _, s_train, _, k_train = choose_rank(X_train, t = self.threshold)
        spectrum = spectral_energy(s_train)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        p = self.threshold * 100
        axes[0].plot(range(1, len(s_train)+1), s_train)
        axes[1].plot(range(1, len(spectrum)+1), spectrum)
        axes[1].axvline(k_train, ls = '--', c = 'orange', label = '%.1f%% of energy: k=%d' % (p, k_train))
        axes[0].set_title("%s: singular value" % title)
        axes[1].set_title("%s: spectral energy" % title)
        axes[1].legend(facecolor = 'white')
        axes[0].set_xlabel('Index of principal component')
        axes[1].set_xlabel('Number of principal components')
        axes[0].set_ylabel('Magnitude of singular value')
        axes[1].set_ylabel('Percent of spectral energy')
        fig.tight_layout()
        plt.show()


    def summary(self, unit=None, intervention=None, potential_outcome_metric=None, testing_error_metric=None): 
        return 
        ''' Summary of results
        Parameters:
            unit: int, default = None. If not specified, the printout aggregates results across all units; if specified, only result of that unit is presented.
            intervention: int, default = None. If not specified, the printout aggregates results across all interventions; if specified, only result of that intervention is presented.
            potential_outcome_metric: str, default = 'mean', can also choose from 'min', 'max' or 'sum'. The metric to calculate potential outcomes.
            testing_error_metric: str, default = 'adjusted_r2', can also choose from 'mse', 'mae', 'mape', 'ame', 'ampe' or 'r2'. The metric to calculate testing errors. If pred_t represents prediction of the synthetic intervention model for timestamp t and true_t represents the ground truth for timestamp t, then 'mse', mean squared error, is calculated as sum(pred_t - true_t)**2/T; 'mae', mean absulte error, is calculate as sum|pred_t - true_t|/T; 'mape', mean absolute percent error, is calculated as sum|pred_t/true_t - 1|/T; 'ame', absolute mean error, is calculated as |sum(pred_t-true_t)/T|; 'ampe', absolute mean percent error, is calculated as |sum(pred_t/true_t - 1)/T|. 

        '''
        #for unit in 
        #model.predict()
        """
                                Synthetic Intervention Results                            
        ==============================================================================
        No. units:                         40   No. interventions:                   3
        ==============================================================================
                                        Estimation #(potential outcome across all units)
                                        (average potential outcome across across units)
        ------------------------------------------------------------------------------                  
        Intervention 0      min    25%    50%    75%    max    mean
        Intervention 1      min    25%    50%    75%    max    mean
            ...                                                 
        ==============================================================================                  
                                      Diagnostic Tests
        ------------------------------------------------------------------------------
                                       Testing Error #(observed units, Metric: adjusted R2 score)
        ------------------------------------------------------------------------------
        Intervention 0      min    25%    50%    75%    max    mean
        Intervention 1      min    25%    50%    75%    max    mean
            ...                                 
        ------------------------------------------------------------------------------
                                      SI-specific Test
        ------------------------------------------------------------------------------                  
                                    Subspace Inclusion Test
        ------------------------------------------------------------------------------
        Intervention 0   % pass rate (only predicted)   % pass rate (all units)
        Intervention 1   % pass rate (only predicted)   % pass rate (all units)
            ...
        ============================================================================== 
        """ 

def plot(y_train, y_train_pred, y_test_pred, counterfactual, y_test = None, y_text = "", title = ""):
    if y_test != None:
        plt.plot(np.concatenate((y_train, y_test)), color='black', lw=2, label = 'ground truth')
    else:
        plt.plot(y_train, color='black', lw=2, label = 'ground truth')
    plt.plot(np.concatenate((y_train_pred, y_test_pred), axis = None), color='lightgray', lw=2, label = 'prediction')
    plt.axvline(x=len(y_train)-1, linestyle=':', color='gray')
    plt.legend()
    plt.xticks([0.5, len(y_train)-0.5], ['Training', 'Testing'])
    plt.tick_params(axis=u'both', which=u'both',length=0)
    plt.ylabel(y_text)
    plt.title("Synthetic intervention estimation" + title)
    plt.show()


def subspace_inclusion_hypothesis_test(X_train, X_test, t1 = 0.99, t2 = 0.99, rho = 0.05):
        k1, k2, tau_hat = compute_tau_hat(X_train, X_test, t1=t1, t2=t2)
        # TODO: verify
        tau_test = 'pass' if (tau_hat < rho * np.linalg.norm(X_test)**2) else 'fail'
        return tau_test

def get_donor_tensor(stats, tensor, unit_ID, timestamp_ID, intervention_ID):
    ''' N x T x D tensor'''
    # criterion 1: under the specified intervention at the specified timestmap
    # criterion 2: under the same intervention at some timestamps
    candidate_units_list = np.where(~np.isnan(tensor[:, timestamp_ID, intervention_ID]))[0].tolist() # unit must be under the specified intervention_ID at timestamp_ID in order to make predictions
    max_score = 0 # find max of min(num_units, num_timestamps)
    train_tensor = None
    test_tensor = None
    final_selected_units = None
    final_selected_timestamps = None
    #print(list(map(stats['%s_inverse_encoder' % axis1_name].get, candidate_units_list)))

    # brute force method
    for num_selected_units in range(1, len(candidate_units_list)+1):
        for selected_units in combinations(candidate_units_list, num_selected_units):
            train_temp = tensor[list(selected_units)+[unit_ID], :, :].reshape(num_selected_units+1, -1)
            #print(train_temp)
            #train_temp = np.vstack((np.reshape((1, -1)), train_temp))
            test_temp = tensor[selected_units, timestamp_ID, intervention_ID]
            selected_timestamps = np.where(~np.isnan(train_temp).any(axis=0))[0]
            if min(len(selected_timestamps), num_selected_units) <= max_score:
                break
            train_tensor = train_temp[:, selected_timestamps].copy()
            test_tensor = test_temp.copy()
            max_score = min(len(selected_timestamps), num_selected_units)
            final_selected_units = selected_units
            final_selected_timestamps = selected_timestamps
            #print(train_tensor)
            #print("----")
    #print(final_selected_units, final_selected_timestamps)
    if final_selected_units == None and final_selected_timestamps == None:
        return None, None, None
    X_train = train_tensor[:-1, :]
    y_train = train_tensor[-1, :]
    X_test = test_tensor.reshape(-1, 1)
    print("X train shape:", X_train.shape)
    print("y train shape:", y_train.shape)
    print("X test shape:", X_test.shape)
    print("Donor %ss:" % (axis1_name), list(map(stats['%s_inverse_encoder' % axis1_name].get, final_selected_units)))
    #print(final_selected_timestamps)
    #print("Overlapping %ss of donor %ss:" % (axis2_name, axis1_name), list(map(stats['%s_inverse_encoder' % axis2_name].get, final_selected_timestamps)))
    return X_train, y_train, X_test

def get_ci(beta, y_pre_n, y_pre_d):
    sigma = np.sqrt(1/y_pre_n.shape[0]*(np.linalg.norm(y_pre_n - y_pre_d.T @ beta))**2)
    return 1.96 * sigma * np.linalg.norm(beta)

def spectral_energy(s):
    '''
    Computes spectral energy (squared singular values)
    '''
    return (100 * (s ** 2).cumsum() / (s ** 2).sum())

# choose rank 
def choose_rank(X, t=0.99): 
    # compute svd 
    u, s, v = np.linalg.svd(X, full_matrices=False)
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    v *= signs[:, np.newaxis]

    # compute rank 
    k = approximate_rank(s, t=t) 
    return u, s, v, k 

# get approximate rank 
def approximate_rank(s, t=0.99): 
    total_energy = (s ** 2).cumsum() / (s ** 2).sum()
    k = list((total_energy > t)).index(True) + 1
    return k 

def get_linear_coef(u, s, v, k, y):
    # (hard) threshold
    u_k = u[:, :k]
    v_k = v[:k, :]
    s_k = s[:k]

    # regression estimate y_train = x @ beta + (intercept)
    beta = ((u_k / s_k) @ v_k ) @ y
    return beta

def mse(X, Y): 
    return np.mean((X - Y)**2)

def phi(T, N, alpha=0.05): 
    return np.sqrt(T) + np.sqrt(N) + np.sqrt(np.log(2/alpha))

def compute_tau_hat(X1, X2, t1=0.99, t2=0.99): 
    # choose rank 
    u1, s1, v1, k1 = choose_rank(X1, t=t1)
    u2, s2, v2, k2 = choose_rank(X2, t=t2)
    u1 = u1[:, :k1]
    u2 = u2[:, :k2]


    # compute statistic 
    P = u1 @ u1.T
    delta = u2 - (P @ u2)

    tau_hat = np.linalg.norm(delta, 'fro')**2
    return k1, k2, tau_hat

def compute_tau(X1, X2, t1=0.99, t2=0.99, alpha=0.05):
    T0, N = X1.shape 
    T1, _ = X2.shape 

    # choose rank 
    u1, s1, v1, k1 = choose_rank(X1, t=t1)
    u2, s2, v2, k2 = choose_rank(X2, t=t2)

    # compute phi 
    phi_pre = phi(T0, N, alpha=alpha)
    phi_post = phi(T1, N, alpha=alpha)

    # compute variance 
    s1_ = s1.copy()
    s2_ = s2.copy()
    s1_[k1:].fill(0)
    s2_[k2:].fill(0)
    X1_hsvt = (u1 * s1_) @ v1
    X2_hsvt = (u2 * s2_) @ v2
    var1 = mse(X1, X1_hsvt)
    var2 = mse(X2, X2_hsvt)
    var = (var1 + var2) / 2
    #print(s1_, s2_)
    #print(var1, var2, var)

    # compute tau 
    tau1 = (4*var*k2*(phi_pre**2) / (s1[k1-1]**2))
    tau2 = (4*var*k2*(phi_post**2) / (s2[k2-1]**2))
    tau3 = (4*np.sqrt(var)*phi_pre) / s1[k1-1]
    tau = tau1 + tau2 + tau3 
    return tau 
