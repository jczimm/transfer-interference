# src/utils/figure_utils.py

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def _style_axes(ax):
    """Helper function to style plot axes consistently."""
    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Style visible spines
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    
    # Style labels and ticks
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(colors='black', width=.5)


def significance_stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''
    
def plot_transfer(data, var, condition_order, ylabel, xlim, ylim, yticks, schedule_colours, p_values,
                  addtests=1, markersize=1.5, scatter=True, figsize=[3 / 2.54, 4.5 / 2.54]):
    
    """Plots the main data split by condition."""
    fig, ax = plt.subplots(figsize=figsize)

    sns.stripplot(data=data, x='condition', y=var, ax=ax, 
                order=condition_order, hue_order=condition_order,
                palette=schedule_colours, hue='condition', legend=False,
                alpha=0.3, size=2.5, linewidth=0, jitter=True, zorder=1)
     
    # Error bars
    sns.pointplot(data=data, x='condition', y=var, ax=ax, 
                order=condition_order, 
                palette='dark:k',
                hue='condition',
                markers='',
                errorbar=('se'),
                linewidth=1, zorder=2)
    
    # Adjust x-coordinates of error bars
    for line in ax.lines:
        xdata = line.get_xdata()
        if len(xdata) == 2: 
            line.set_xdata([xdata[0] - 0.1, xdata[1] - 0.1])
    
    # Marker on top
    sns.pointplot(data=data, x='condition', y=var, ax=ax, 
                order=condition_order, 
                color='k',
                errorbar=None, markers='o', markersize=3.5, 
                linewidth=0.75, **{'markerfacecolor': 'white'}, zorder=3)
    
    if addtests:
        # Adds significance bars and stars to the plot.
        y_max = data[var].max() + 0.05
        for i, (pair, p_value) in enumerate(zip([(0, 1), (0, 2), (1, 2)], p_values)):
            if p_value < 0.05:
                x_coords = pair
                y_coord = y_max + (i * 0.1)
                ax.plot(x_coords, [y_coord + 0.05] * 2, color='black', linewidth=0.5)
                ax.text(np.mean(x_coords), y_coord, significance_stars(p_value), 
                        ha='center', va='bottom', fontsize=6)
            
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(condition_order, rotation=25)
    
    # Style the plot
    _style_axes(ax)
    
    return fig, ax



def plot_interference(param_df, A2_param, schedule_colours, figsize=[3 / 2.54, 4.5 / 2.54], ylabel='interference weight'):
    """
    Plot interference analysis comparing A2 parameters.
    
    Parameters
    ----------
    param_df : pd.DataFrame
        DataFrame containing von Mises fit parameters
    A2_param : str
        Column name for A2 parameter
    schedule_colours : list
        List of colors for different schedules
    figsize : list
        Figure dimensions [width, height] in cm
    ylabel : str
        Label for y-axis
        
    Returns
    -------
    fig, ax : tuple
        Figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    param_df_tmp = param_df.loc[param_df['condition']!='same'].copy()
    param_df_tmp['A2_interference'] = 1 - param_df[A2_param]
    
    # Cloud plot (stripplot)
    sns.stripplot(data=param_df_tmp, 
                 x='condition', 
                 y='A2_interference', 
                 ax=ax,
                 order=['near','far'], 
                 palette=schedule_colours[1:], 
                 hue='condition',
                 hue_order=['near','far'],
                 legend=False,
                 alpha=0.3, 
                 size=2.5, 
                 linewidth=0, 
                 jitter=True,
                 zorder=1)
    
        # Error bars
    sns.pointplot(data=param_df_tmp, 
                x='condition', 
                y='A2_interference', 
                ax=ax,
                order=['near','far'], 
                palette='dark:k',  # Changed from color='k' to palette='dark:k'
                linestyle='-',
                hue='condition',
                legend=False,
                markers='',
                errorbar=('se'),
                linewidth=1, 
                zorder=2)
    
    # Adjust error bar positions
    for line in ax.lines:
        xdata = line.get_xdata()
        if len(xdata) == 2:
            line.set_xdata([xdata[0] - 0.05, xdata[1] - 0.05])
            
    # Add markers
    sns.pointplot(data=param_df_tmp, 
                 x='condition', 
                 y='A2_interference', 
                 ax=ax,
                 order=['near','far'], 
                 color='k',
                 errorbar=None, 
                 markers='o', 
                 markersize=3.5,
                 linewidth=0.75, 
                 **{'markerfacecolor': 'white'},
                 zorder=3)
    
    # Set axis properties
    ax.set_ylim([-.1, 1.1])
    ax.set_xlim([-0.5, 1.5])
    ax.set_xticks([0, 1], ['near', 'far'])
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.set_yticks([0, 0.5, 1], [0, 0.5, 1], va='center')

    # Style the plot
    _style_axes(ax)
    
    return fig, ax


def plot_accuracy_timecourse(trial_df, feature_idx, schedule_colours, condition_order, figsize=[7 / 2.54, 4 / 2.54]):
    """
    Plot accuracy over time for a specific feature.
    
    Parameters
    ----------
    trial_df : pd.DataFrame
        DataFrame containing columns: feature_idx, task_section, block, accuracy, condition
    feature_idx : int
        0 for summer, 1 for winter
    schedule_colours : list
        List of colors for different conditions
    condition_order : list
        Order of conditions for plotting
    cm_conv : float
        Conversion factor for figure size
        
    Returns
    -------
    fig, ax : tuple
        Figure and axis objects
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot accuracy timecourse
    sns.lineplot(
        ax=ax,
        data=trial_df.loc[(trial_df['feature_idx']==feature_idx)&
                         (trial_df['task_section']!='A2'), :],
        x='block',
        y='accuracy',
        hue='condition',
        errorbar='se',
        palette=schedule_colours,
        hue_order=condition_order,
        linewidth=1
    )
    
    # Add reference lines
    ax.axvline(10, linestyle='--', color='k', linewidth=0.5)
    ax.axhline(0.5, linestyle='-', color='grey', linewidth=5, alpha=0.2)
    
    # Set labels and ticks
    ax.set_ylabel('winter accuracy' if feature_idx == 1 else 'summer accuracy')
    ax.set_xlabel('block')
    ax.set_yticks(np.arange(0.5, 1.1, 0.25), np.arange(0.5, 1.1, 0.25))
    ax.set_ylim([0.4, 1])
    ax.set_xticks(range(0, 21, 10))
    
    # Adjust legend position
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center', bbox_to_anchor=(1.3, 0.5))
    
    # Style the plot
    _style_axes(ax)
    
    fig.tight_layout()
    
    return fig, ax

## Individual differences plotting functions

def plot_near_hist(data, schedule_colours, figsize=[4.5 / 2.54, 3 / 2.54]):

    fig,ax=plt.subplots(figsize=figsize)

    ax.hist(1-data['A_weight_A2'], color=schedule_colours[1], linewidth=0.5, edgecolor='k', alpha=0.5)
    ax.set_yticks(range(0,41,20))
    ax.set_ylim([0,44])
    ax.set_xlabel('retest interference\n$\it{p}$(Rule B)')

    # Style the plot
    _style_axes(ax)

## ANN specific plotting functions

def plot_loss_curves(ann_data, schedule_name, schedule_colours, n_epochs=100, figsize=[3 / 2.54, 2 / 2.54]):
    """
    Plot loss curves for a specific schedule.
    
    Parameters
    ----------
    ann_data : dict
        Dictionary containing ANN data for different schedules
    schedule_name : str
        Name of schedule to plot ('same', 'near', or 'far')
    n_epochs : int, optional
        Number of epochs per phase
    figsize : list, optional
        Figure dimensions [width, height]
        
    Returns
    -------
    fig, ax : tuple
        Figure and axis objects
    """
    schedule_data = ann_data[schedule_name]
    s_idx = ['same', 'near', 'far'].index(schedule_name)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot phase transition lines
    ax.axvline(n_epochs * 6 * 10, color='k', linestyle='--', alpha=0.3)
    ax.axvline(2 * n_epochs * 6 * 10, color='k', linestyle='--', alpha=0.3)
    
    # Prepare loss data
    flattened_length = schedule_data[0]['losses'].shape[0] * schedule_data[0]['losses'].shape[1]
    sched_losses = np.zeros((len(schedule_data), flattened_length))
    
    for subj in range(len(schedule_data)):
        flat_loss = np.concatenate(schedule_data[subj]['losses'], axis=0)
        sched_losses[subj, :] = flat_loss
    
    # Calculate statistics
    mean_losses = np.nanmean(sched_losses, axis=0)[1::2]  # take only summer
    std_losses = np.nanstd(sched_losses, axis=0)[1::2]
    x_values = np.arange(len(mean_losses))
    
    # Plot
    ax.plot(x_values, mean_losses, 
            color=schedule_colours[s_idx], 
            label=schedule_name, 
            alpha=0.8, 
            linewidth=1.5)
    
    # Style
    ax.set_xticks(range(0, len(x_values), n_epochs * 6 * 10), ['A','B','A'])
    ax.set_yticks(np.arange(0, 0.51, 0.5))
    ax.set_yticklabels(np.arange(0, 0.51, 0.5))
    ax.set_xlabel("task")
    ax.set_ylabel("loss (MSE)")
    
    # Style the plot
    _style_axes(ax)
    
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    return fig, ax

def plot_pca_components(agg_df_long, task_colours, cm_conv=1/2.54):
    """
    Plot the number of PCA components needed to explain variance across conditions.
    
    Parameters:
    -----------
    agg_df_long : pandas.DataFrame
        Long-format DataFrame containing PCA components data
    task_colours : dict
        Dictionary mapping tasks to colors
    cm_conv : float, optional
        Conversion factor from cm to inches (default: 1/2.54)
    
    Returns:
    --------
    tuple
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=[3*cm_conv, 3*cm_conv])
    
    # Style the plot
    _style_axes(ax)
    
    # Create bar plot
    sns.barplot(ax=ax, data=agg_df_long, x='condition', y='n_pca', 
                hue='task', palette=task_colours)
    
    # Customize plot
    ax.legend(title=None, fontsize=7)
    ax.set_ylabel('# PCA')
    ax.set_xlabel('')
    
    return fig, ax

def get_axis_limits(data):
    """
    Determine axis limits for consistent plotting.
    """
    lims = {
        'xmin': data[:, 0].min(),
        'xmax': data[:, 0].max(),
        'ymin': data[:, 1].min(),
        'ymax': data[:, 1].max()
    }
    return lims

def plot_2d_pca(ax, data, color, label):
    
    ax.plot(data[:, 0], data[:, 1], c=color, label=label, linestyle='-', marker='o',linewidth=1,markersize=6,markeredgewidth=0)
    ax.plot([data[-1, 0], data[0, 0]], [data[-1, 1], data[0, 1]], c=color,linewidth=1)
    

def plot_split_stim(ax, hiddens_pca, task_colours,lims):
    
    # Split activity for A and B inputs for visualization
    A_stim_hiddens = hiddens_pca[:6]
    B_stim_hiddens = hiddens_pca[6:]
    
    plot_2d_pca(ax, A_stim_hiddens, task_colours[0], 'Task A Stimuli')
    plot_2d_pca(ax, B_stim_hiddens, task_colours[1], 'Task B Stimuli')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')

    # Set matching axis limits for A_hids
    ax.set_xlim(lims['xmin']-.5, lims['xmax']+.5)
    ax.set_ylim(lims['ymin']-.5, lims['ymax']+.5)
    ax.set_xticks([-1,0,1],[-1,0,1])
    ax.set_yticks([-1,0,1],[-1,0,1])



## Individual differences plotting functions

def plot_near_hist(near_participants_group, schedule_colours, figsize=[4.5 / 2.54, 3 / 2.54]):

    fig,ax=plt.subplots(figsize=figsize)

    ax.hist(1-near_participants_group['A_weight_A2'], color=schedule_colours[1], linewidth=0.5, edgecolor='k', alpha=0.5)
    ax.set_yticks(range(0,41,20))
    ax.set_ylim([0,44])
    ax.set_xlabel('retest interference\n$\it{p}$(Rule B)')
    
    # Style the plot
    _style_axes(ax)

    return fig,ax



def plot_id_1group(data, grouping, group_order, group_names, var, yticks, ytick_labs, ylim, ylab, colors, add_tests=0, p_value=np.nan, y_coord=np.nan,figsize=[3 / 2.54, 4.5 / 2.54]):
    """Plot individual differences for one group with error bars and significance testing."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create base plot layers
    sns.stripplot(data=data, x=grouping, y=var, ax=ax,
                 color=colors, order=group_order, legend=False,
                 alpha=0.3, size=3.5, linewidth=0, jitter=True, zorder=1)

    sns.pointplot(data=data, x=grouping, y=var, ax=ax,
                 order=group_order, color='k', linestyle='-',
                 markers='', errorbar='se', linewidth=1, zorder=2)
    
    # Shift error bars slightly left
    for line in ax.lines:
        if len(line.get_xdata()) == 2:  
            line.set_xdata(line.get_xdata() - 0.05)
            
    # Add markers on top
    sns.pointplot(data=data, x=grouping, y=var, ax=ax,
                 order=group_order, color='k', errorbar=None,
                 markers='o', markersize=3.5, linewidth=0.75,
                 linestyles="", markerfacecolor='white', zorder=3)
    
    # Style the plot
    _style_axes(ax)
    
    # Set labels and limits
    ax.set_xticks([0,1], group_names)
    ax.set_xlabel('')
    ax.set_xlim([-0.5, 1.5])
    ax.set_yticks(yticks, ytick_labs)
    ax.set_ylabel(ylab)
    ax.set_ylim(ylim)
    
    # Add significance testing if requested
    if add_tests and p_value < 0.05:
        ax.plot([0.2, 0.8], [y_coord] * 2, color='black', linewidth=0.5)
        ax.text(0.5, y_coord, significance_stars(p_value),
               ha='center', va='bottom', fontsize=6)
    
    return fig, ax

def plot_id_groups(data, grouping, group_order, group_names, var, yticks, ytick_labs, ylim, ylab, colors='grey', add_tests=0, p_value=np.nan, y_coord=np.nan,figsize=[6 / 2.54, 4.5 / 2.54]):
    """Plot individual differences comparing multiple groups with error bars and connecting lines."""
    fig, ax = plt.subplots(figsize=figsize)

    # Create base plot layers
    sns.stripplot(data=data, x=grouping, y=var, ax=ax, hue='ann',
                 palette=colors, order=group_order, legend=False,
                 alpha=0.3, size=3.5, linewidth=0, jitter=True, zorder=1)

    sns.pointplot(data=data, x=grouping, y=var, ax=ax,
                 order=group_order, color='k', linestyle='',
                 markers='', errorbar='se', linewidth=1, zorder=2)
    
    # Add connecting lines between means
    means = [np.mean(data.loc[data[grouping]==group, var]) for group in group_order]
    ax.plot([0,1], means[:2], color='k')
    ax.plot([2,3], means[2:], color='k')
    
    # Shift error bars slightly left
    for line in ax.lines:
        if len(line.get_xdata()) == 2:
            line.set_xdata(line.get_xdata() - 0.05)
            
    # Add markers on top
    sns.pointplot(data=data, x=grouping, y=var, ax=ax,
                 order=group_order, color='k', errorbar=None,
                 markers='o', markersize=3.5, linewidth=0.75,
                 linestyles="", markerfacecolor='white', zorder=3)
    
    # Style the plot
    _style_axes(ax)
    
    # Set labels and limits
    ax.set_xlabel('')
    ax.text(0.25, -0.25, 'participants', ha='center', va='center', transform=ax.transAxes)
    ax.text(0.75, -0.25, 'ANNs', ha='center', va='center', transform=ax.transAxes)
    ax.set_yticks(yticks, ytick_labs)
    ax.set_ylabel(ylab)
    ax.set_ylim(ylim)
    
    # Add significance testing if requested
    if add_tests and p_value < 0.05:
        ax.plot([0.2, 0.8], [y_coord] * 2, color='black', linewidth=0.5)
        ax.text(0.5, y_coord, significance_stars(p_value),
               ha='center', va='bottom', fontsize=6)
    
    return fig, ax

