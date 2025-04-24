import numpy as np
from scipy import stats
from scipy.stats import norm

def compare_transfer(df, metric_col='error_diff'):
    """
    Run statistical comparisons between conditions.
    
    Args:
        df (pd.DataFrame): DataFrame with columns:
            - condition
            - {metric_col} (the metric to compare)
        metric_col (str): Name of column containing the metric to compare
            
    Returns:
        dict: Statistical test results including means, standard errors,
        t-tests with test statistics, effect sizes, degrees of freedom and p-values
    """
    # Split by condition
    same = df[df['condition'] == 'same'][metric_col]
    near = df[df['condition'] == 'near'][metric_col]
    far = df[df['condition'] == 'far'][metric_col]
    
    # Calculate means and standard errors
    same_mean = round(np.mean(same), 3)
    same_sem = round(stats.sem(same), 3)
    near_mean = round(np.mean(near), 3)
    near_sem = round(stats.sem(near), 3)
    far_mean = round(np.mean(far), 3)
    far_sem = round(stats.sem(far), 3)
    
    # ANOVA
    f_stat, p_value = stats.f_oneway(same, near, far)
    f_stat = round(f_stat, 3)
    p_value = round(p_value, 3)
    df_between = 2  # number of groups - 1
    df_within = len(same) + len(near) + len(far) - 3
    eta_squared = round((f_stat * df_between) / (f_stat * df_between + df_within), 3)
    
    # Post-hoc t-tests with effect sizes
    def calc_cohens_d(x1, x2):
        n1, n2 = len(x1), len(x2)
        pooled_sd = np.sqrt(((n1-1)*np.var(x1) + (n2-1)*np.var(x2)) / (n1+n2-2))
        return round((np.mean(x1) - np.mean(x2)) / pooled_sd, 3)
        
    # Same vs Near
    t_same_near, p_same_near = stats.ttest_ind(same, near, alternative='greater')
    t_same_near = round(t_same_near, 3)
    p_same_near = round(p_same_near, 3)
    d_same_near = calc_cohens_d(same, near)
    df_same_near = len(same) + len(near) - 2
    
    # Same vs Far  
    t_same_far, p_same_far = stats.ttest_ind(same, far, alternative='greater')
    t_same_far = round(t_same_far, 3)
    p_same_far = round(p_same_far, 3)
    d_same_far = calc_cohens_d(same, far)
    df_same_far = len(same) + len(far) - 2
    
    # Near vs Far
    t_near_far, p_near_far = stats.ttest_ind(near, far, alternative='greater')
    t_near_far = round(t_near_far, 3)
    p_near_far = round(p_near_far, 3)
    d_near_far = calc_cohens_d(near, far)
    df_near_far = len(near) + len(far) - 2
    
    return {
        'descriptives': {
            'same': {
                'mean': same_mean,
                'sem': same_sem
            },
            'near': {
                'mean': near_mean,
                'sem': near_sem
            },
            'far': {
                'mean': far_mean,
                'sem': far_sem
            }
        },
        'anova': {
            'f_stat': f_stat,
            'df_between': df_between,
            'df_within': df_within,
            'eta_squared': eta_squared,
            'p_value': p_value
        },
        'posthoc': {
            'same_vs_near': {
                't_stat': t_same_near,
                'df': df_same_near,
                'cohens_d': d_same_near,
                'p_value': p_same_near
            },
            'same_vs_far': {
                't_stat': t_same_far,
                'df': df_same_far,
                'cohens_d': d_same_far,
                'p_value': p_same_far
            },
            'near_vs_far': {
                't_stat': t_near_far,
                'df': df_near_far,
                'cohens_d': d_near_far,
                'p_value': p_near_far
            }
        }
    }

def compare_interference(study_params):
    """
    Compare interference effects between conditions.
    
    Parameters
    ----------
    study_params : pd.DataFrame
        DataFrame containing columns:
        - condition
        - A_weight_A1
        - A_weight_A2
        
    Returns
    -------
    dict
        Dictionary containing test results for:
        - A1 weights comparison
        - A2 weights comparison
        - A1-A2 difference comparison
        Including test statistics, effect sizes,
        degrees of freedom and p-values
    """
    # Extract condition data
    near_A1 = study_params.loc[study_params['condition']=='near', 'A_weight_A1'].values
    far_A1 = study_params.loc[study_params['condition']=='far', 'A_weight_A1'].values
    near_A2 = study_params.loc[study_params['condition']=='near', 'A_weight_A2'].values
    far_A2 = study_params.loc[study_params['condition']=='far', 'A_weight_A2'].values
    
    # Calculate differences
    near_diff = near_A1 - near_A2
    far_diff = far_A1 - far_A2
        
    # A1 weights comparison
    t_A1, p_A1 = stats.ttest_ind(near_A1, far_A1, alternative='less', nan_policy='omit')
    df_A1 = len(near_A1) + len(far_A1) - 2
    pooled_std_A1 = np.sqrt(((len(near_A1)-1)*np.var(near_A1) + (len(far_A1)-1)*np.var(far_A1)) / df_A1)
    d_A1 = round((np.mean(near_A1) - np.mean(far_A1)) / pooled_std_A1, 3)
    
    # A2 weights comparison  
    t_A2, p_A2 = stats.ttest_ind(near_A2, far_A2, alternative='less', nan_policy='omit')
    df_A2 = len(near_A2) + len(far_A2) - 2
    pooled_std_A2 = np.sqrt(((len(near_A2)-1)*np.var(near_A2) + (len(far_A2)-1)*np.var(far_A2)) / df_A2)
    d_A2 = round((np.mean(near_A2) - np.mean(far_A2)) / pooled_std_A2, 3)
    
    # A1-A2 difference comparison
    t_diff, p_diff = stats.ttest_ind(near_diff, far_diff, alternative='greater', nan_policy='omit')
    df_diff = len(near_diff) + len(far_diff) - 2
    pooled_std_diff = np.sqrt(((len(near_diff)-1)*np.var(near_diff) + (len(far_diff)-1)*np.var(far_diff)) / df_diff)
    d_diff = round((np.mean(near_diff) - np.mean(far_diff)) / pooled_std_diff, 3)
    
    return {
        'A1_weights': {
            't_stat': round(t_A1, 3),
            'df': df_A1,
            'cohens_d': d_A1,
            'p_value': round(p_A1, 3),
            'means': {
                'near': round(np.mean(near_A1), 3),
                'far': round(np.mean(far_A1), 3)
            },
            'standard_errors': {
                'near': round(stats.sem(near_A1), 3),
                'far': round(stats.sem(far_A1), 3)
            }
        },
        'A2_weights': {
            't_stat': round(t_A2, 3),
            'df': df_A2,
            'cohens_d': d_A2,
            'p_value': round(p_A2, 3),
            'means': {
                'near': round(np.mean(near_A2), 3),
                'far': round(np.mean(far_A2), 3)
            },
            'standard_errors': {
                'near': round(stats.sem(near_A2), 3),
                'far': round(stats.sem(far_A2), 3)
            }
        },
        'A1_A2_difference': {
            't_stat': round(t_diff, 3),
            'df': df_diff,
            'cohens_d': d_diff,
            'p_value': round(p_diff, 3),
            'means': {
                'near': round(np.mean(near_diff), 3),
                'far': round(np.mean(far_diff), 3)
            },
            'standard_errors': {
                'near': round(stats.sem(near_diff), 3),
                'far': round(stats.sem(far_diff), 3)
            }
        }
    }


def behav_group_comparisons(grouped_df_all, var):
    # Get values for humans
    splitters = grouped_df_all.loc[grouped_df_all['group']=='splitters',var].values
    lumpers = grouped_df_all.loc[grouped_df_all['group']=='lumpers',var].values
    # Calculate t-test and Cohen's d for humans
    t_stat_human, p_val_human = stats.ttest_ind(splitters, lumpers)
    df_human = len(splitters) + len(lumpers) - 2
    d_human = (np.mean(splitters) - np.mean(lumpers)) / np.sqrt((np.var(splitters) + np.var(lumpers)) / 2)
    print(f't-test: {t_stat_human}, p={p_val_human}, df={df_human}')
    print(f"Cohen's d: {d_human}")
    print("\nHuman data:")
    print(f"Splitters: M={np.mean(splitters):.3f}, SE={stats.sem(splitters):.3f}")
    print(f"Lumpers: M={np.mean(lumpers):.3f}, SE={stats.sem(lumpers):.3f}")

    print(f'\nANNs')
    # Get values for ANNs  
    lazy = grouped_df_all.loc[grouped_df_all['group']=='lazy',var].values
    rich = grouped_df_all.loc[grouped_df_all['group']=='rich',var].values
    # Calculate t-test and Cohen's d for ANNs
    t_stat_ann, p_val_ann = stats.ttest_ind(lazy, rich)
    df_ann = len(lazy) + len(rich) - 2
    d_ann = (np.mean(lazy) - np.mean(rich)) / np.sqrt((np.var(lazy) + np.var(rich)) / 2)
    print(f't-test: {t_stat_ann}, p={p_val_ann}, df={df_ann}')
    print(f"Cohen's d: {d_ann}")
    print("\nANN data:")
    print(f"Lazy: M={np.mean(lazy):.3f}, SE={stats.sem(lazy):.3f}")
    print(f"Rich: M={np.mean(rich):.3f}, SE={stats.sem(rich):.3f}")