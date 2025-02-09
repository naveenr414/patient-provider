import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
from collections import Counter
from copy import deepcopy
import matplotlib.patches as patches
from matplotlib.lines import Line2D

color_schemes = {
    'two_color_blue_green': ["#38bae2","#4eb156"], 
    'two_color_blue_red': ["#7aadd1","#df5e5f"], 
    'two_color_blue_red_light': ["#7aadd130","#df5e5f30"], 
    'three_color_america': ["#f7f7f7","#6daedb","#ffb6c2"], 
    'three_color_primary': ["#ff7f0f","#2ba02b","#9467bd"], 
    'six_color': [(0.216, 0.494, 0.722, 0.7),
                (1.0, 0.498, 0.0, 0.7),
                (0.302, 0.686, 0.29, 0.7),
                (0.969, 0.506, 0.749, 0.7),
                (0.596, 0.306, 0.639, 0.7),
                (0.894, 0.102, 0.11, 0.7)], 
        }
markers = [".","v","^","s","*","x"]

def get_or_none(d,key):
    """Helper function to either retunr None or the value of a key
    
    Arguments:
        d: Dictionary
        key: String, key potentially in dictionary
        
    Returns: None or d[key]"""

    if key in d:
        return d[key]
    return None 

def plot_bar(ax,x_groups,y_values,y_errors,labels,formatting):
    """Create a bar plot, based on the following:
    
    Arguments:
        x_groups: Integer list, with each indicating which group
            the particular bar is in
            For example, [1,2,3,1,2,3,1,2,3...]
        y_values: Corresponding y_values
        y_errors: Corresponding errors/standard deviation bars
        labels: Dictionary mapping groups to their labels
        formatting: Dictionary, with different keys for different settings
            keys:
                style_size: 'paper' or 'presentation' depending on size
                color_palette: either a single color, or a selection
                    from the color palette dictionary
                bar_width: float, by default 0.25
                horizontal: Boolean, whether to 
                edgecolor: either a color or None; whether to color the bars
                extra_labels: Dictionary, saying which groups to have extra labels for
                    extra_y_shift: How much to shift the extra labels by on the y
                    extra_x_shift: How much to shift the extra labels by on the x
                    label_rotation: Degree to rotate the labels
                    format_string: Function mapping strings to their display
                        for the extra labels
                    per_group_labels: List of strings to show on top              

    Returns: Nothing
    
    Side Effects: Plots a bar plot"""

    if formatting['style_size'] == 'paper':
        label_size = 10
    if formatting['style_size'] == 'presentation':
        label_size = 14
    

    num_groups = len(set(x_groups))
    max_bars_per_group = max(Counter(x_groups).values())

    values_by_group = {}
    errors_by_group = {}
    ordered_groups = sorted(list(set(x_groups)))

    for i in range(len(x_groups)):
        if x_groups[i] not in values_by_group:
            values_by_group[x_groups[i]] = []
            errors_by_group[x_groups[i]] = []

        values_by_group[x_groups[i]].append(y_values[i])
        errors_by_group[x_groups[i]].append(y_errors[i])

    if 'color_palette' not in formatting:
        formatting['color_palette'] = 'six_color'
    
    if formatting['color_palette'][0] == '#':
        colors = [formatting['color_palette']]
        assert num_groups == 1
    else:
        assert formatting['color_palette'] in color_schemes
        assert len(color_schemes[formatting['color_palette']]) >= num_groups

        colors = color_schemes[formatting['color_palette']]

    if 'bar_width' not in formatting:
        formatting['bar_width'] = 0.25
    bar_width = formatting['bar_width']

    x = np.arange(max_bars_per_group)

    for i, group_num in enumerate(ordered_groups):
        if 'horizontal' in formatting and formatting['horizontal']:
            bars = ax.barh(x + i * bar_width, values_by_group[group_num],
                        height=bar_width, label=labels[group_num],
                        edgecolor=get_or_none(formatting,'edgecolor'),color=colors[i],yerr=errors_by_group[group_num])
        else:
            bars = ax.bar(x + i * bar_width, values_by_group[group_num],
                        width=bar_width, label=labels[group_num],
                        edgecolor=get_or_none(formatting,'edgecolor'),color=colors[i],yerr=errors_by_group[group_num])

        if 'extra_labels' in formatting and group_num in formatting['extra_labels']:  # Check if it's the 3rd model
            if 'extra_x_shift' not in formatting: 
                formatting['extra_x_shift'] = 0
            if 'extra_y_shift' not in formatting: 
                formatting['extra_y_shift'] = 0

            for j,bar in enumerate(bars):
                width = bar.get_width()
                height = bar.get_height()

                if 'format_string' not in formatting:
                    formatting['format_string'] = lambda s: str(s)
                
                if get_or_none(formatting,'horizontal'):
                    ax.text(formatting['extra_x_shift'], bar.get_y() + bar.get_height() / 2+formatting['extra_y_shift'], formatting['format_string'](formatting['extra_labels'][group_num][j]),
                        ha='center', va='bottom', fontsize=label_size, color='black')

                else:
                    ax.text(bar.get_x() + width / 2 + formatting['extra_x_shift'], height +formatting['extra_y_shift'], formatting['format_string'](formatting['extra_labels'][group_num][j]),
                            ha='center', va='bottom', fontsize=label_size, color='black')

    if 'label_rotation' not in formatting:
        formatting['label_rotation'] = 20

    if 'per_group_labels' in formatting:
        if 'horizontal' in formatting and formatting['horizontal']:
            ax.set_yticks(x + bar_width * (num_groups - 1) / 2)
            ax.set_yticklabels(formatting['per_group_labels'],fontsize=14,rotation=formatting['label_rotation'])
        else:
            ax.set_xticks(x + bar_width * (num_groups - 1) / 2)
            ax.set_xticklabels(formatting['per_group_labels'],fontsize=14,rotation=formatting['label_rotation'])

def plot_zero_one_matrix(ax,matrix,row_labels,formatting):
    """Plot a zero-one matrix as a set of dots
    
    Arguments:
        ax: Matplotlib axes object
        matrix: 0-1 numpy matrix
        row_labels: List of labels for each row
        formatting: Dictionary with details on formatting
            style_size: 'paper' or 'presentation' depending on size
            color_palette: either a single color, or a selection
                from the color palette dictionary
            label_x: float, location of the first label
            label_y: float, location of the first label
            x_start: float, location to start circles
            y_start: float, location to start circles
            x_width: float, width between circles
            y_width: float, width between circles
            circle_width: float, how wide to show ellipse
            circle_height: float, how tall to show ellipse"""

    if formatting['style_size'] == 'paper':
        font_size = 14
    elif formatting['style_size'] == 'presentation':
        font_size = 18

    for i in range(len(row_labels)):
        ax.text(formatting['label_x'], formatting['y_start']+i*formatting['y_width'], row_labels[i], color='black', ha='right', va='center', fontsize=font_size)

    # Add patches based on the matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            facecolor = '#222222' if matrix[i][j] == 0 else '#EEEEEE'
            rect = patches.Ellipse((formatting['x_start']+j*formatting['x_width'], formatting['y_start'] + i*formatting['y_width']), formatting['circle_width'], formatting['circle_height'],
                                linewidth=2, facecolor=facecolor, clip_on=False)
            ax.add_patch(rect)

def plot_line(ax,x_values,y_values,y_confidence,labels,formatting):
    """Create a line plot, based on the following:
    
    Arguments:
        ax: The Matplotlib axis to use
        x_values: X values for the x axis; multiple lists for different 
            Plots
        y_values: Y values for the x axis; multiple lists for different 
            Plots
        y_confidence: Plots for the confidence intervals
        formatting: Dictionary, with different keys for different settings
            style_size: 'paper' or 'presentation' depending on size
            color_palette: either a single color, or a selection
                from the color palette dictionary

    Returns: Nothing
    
    Side Effects: Plots a bar plot"""

    if formatting['color_palette'][0] == '#':
        colors = [formatting['color_palette'] for i in range(len(x_values))]
    else:
        assert formatting['color_palette'] in color_schemes
        assert len(color_schemes[formatting['color_palette']]) >= len(x_values)

        colors = color_schemes[formatting['color_palette']]
    
    linewidth = 0.6
    if 'linewidth' in formatting:
        linewidth = formatting['linewidth']
    for i in range(len(x_values)):
        marker = markers[i]
        linestyle='-'
        if 'no_marker' in formatting:
            marker = None
        if 'linestyle' in formatting:
            linestyle = formatting['linestyle']
        ax.plot(x_values[i],y_values[i],label=labels[i],linewidth=linewidth,color=colors[i],marker=marker,linestyle=linestyle)
        ax.fill_between(x_values[i],np.array(y_values[i])-np.array(y_confidence[i]),np.array(y_values[i])+np.array(y_confidence[i]), alpha=0.2,color=colors[i])

def plot_scatter(ax,x_values,y_values,labels,formatting):
    """Create a line plot, based on the following:
    
    Arguments:
        ax: The Matplotlib axis to use
        x_values: X values for the x axis; multiple lists for different 
            Plots
        y_values: Y values for the x axis; multiple lists for different 
            Plots
        formatting: Dictionary, with different keys for different settings
            style_size: 'paper' or 'presentation' depending on size
            color_palette: either a single color, or a selection
                from the color palette dictionary
            size: float, size of the scattered points

    Returns: Nothing
    
    Side Effects: Plots a bar plot"""

    if len(labels) == 0:
        labels = [None for i in range(len(x_values))]

    if formatting['color_palette'][0] == '#':
        colors = [formatting['color_palette']]
        assert len(x_values) == 1
    else:
        assert formatting['color_palette'] in color_schemes
        assert len(color_schemes[formatting['color_palette']]) >= len(x_values)

        colors = color_schemes[formatting['color_palette']]

    if 'size' not in formatting:
        formatting['size'] = 5

    for i in range(len(x_values)):
        ax.scatter(x_values[i],y_values[i],color=colors[i],label=labels[i],s=formatting['size'],marker=markers[i])

def plot_box_whisker(ax,data,labels,formatting):
    """Create a line plot, based on the following:
    
    Arguments:
        ax: The Matplotlib axis to use
        data: The list of values for each box and whisker plot
        labels: List of strings to label each data distro
        formatting: Dictionary, with different keys for different settings
            color_palette: either a single color, or a selection
                from the color palette dictionary

    Returns: Nothing
    
    Side Effects: Plots a bar plot"""

    if formatting['color_palette'][0] == '#':
        colors = [formatting['color_palette']]
    else:
        assert formatting['color_palette'] in color_schemes
        assert len(color_schemes[formatting['color_palette']]) >= len(data)

        colors = color_schemes[formatting['color_palette']]
    
    ax.boxplot(data, labels=labels, patch_artist=True, notch=True, vert=False, boxprops=dict(facecolor=colors[0]),showfliers=False)

def plot_kde(ax,data,labels,formatting):
    """Create a line plot, based on the following:
    
    Arguments:
        ax: The Matplotlib axis to use
        data: The list of values for each box and whisker plot
        labels: List of strings to label each data distro
        formatting: Dictionary, with different keys for different settings
            color_palette: either a single color, or a selection
                from the color palette dictionary
    Returns: Nothing
    
    Side Effects: Plots a bar plot"""

    if formatting['color_palette'][0] == '#':
        colors = [formatting['color_palette']]
    else:
        assert formatting['color_palette'] in color_schemes
        assert len(color_schemes[formatting['color_palette']]) >= len(data)

        colors = color_schemes[formatting['color_palette']]
    
    for i in range(len(data)):
        sns.kdeplot(
            data[i],
            label=labels[i],
            fill=True,
            color=colors[i],
            ax=ax
        )



def plot_text(ax,text,x,y,formatting):
    """Create a line plot, based on the following:
    
    Arguments:
        ax: The Matplotlib axis to use
        text: Text to show
        x: X value
        y: Y value
        formatting: Dictionary, with different keys for different settings
            color_palette: either a single color, or a selection
                from the color palette dictionary
            fontsize: how large the text should be
    Returns: Nothing
    
    Side Effects: Plots a bar plot"""


    ax.text(x,y, text, color=formatting['color_palette'], fontsize=formatting['fontsize'], ha='center')

def create_axes(plot_dimensions,formatting,x_labels=None,y_labels=None,titles=None,sup_x_label="",sup_y_label="",sup_title="",sup_x_label_loc=0.01,sup_y_label_loc=None):
    """Create the figure and axes elements with certain labels, 
        number of subplots, and a title
        
    Arguments:
        plot_dimensions
        formatting: Dictionary with formatting details
            style_size: 'paper' or 'presentation' depending on size
            x_lim: List of List of List of float, x, y limits
            y_lim: float, x, y limits
            x_ticks: List of List of List of pair lists with ticks + labels
            y_ticks: List of List of List of pair lists with ticks + labels
            hide_spines: whether to hide the spines on the top and right
            separate_spines: wheter to separate the bottom and left spines
            has_grid: whether to show the grids
            has_x_grid: show only the x grid
            has_y_grid: show only the y grid"""

    # Plot each model's bars with an offset
    if 'constrained_layout' in formatting and formatting['constrained_layout']:
        constrained_layout = True 
    else:
        constrained_layout = False 
    fig, ax = plt.subplots(plot_dimensions[0],plot_dimensions[1],figsize=formatting['figsize'],constrained_layout=constrained_layout)

    default = [["" for i in range(plot_dimensions[1])] for j in range(plot_dimensions[0])]
    if x_labels == None:
        x_labels = deepcopy(default)
    if y_labels == None:
        y_labels = deepcopy(default)
    if titles == None:
        titles = deepcopy(default)

    if plot_dimensions[0] == plot_dimensions[1] == 1:
        ax = [[ax]]
    elif plot_dimensions[0] == 1:
        ax = [ax]
    elif plot_dimensions[1] == 1:
        ax = [[i] for i in ax]

    if formatting['style_size'] == 'paper':
        label_size = 14
        title_size = 14
        tick_size = 10
    elif formatting['style_size'] == 'presentation':
        label_size = 18
        title_size = 18
        tick_size = 14
    
    if 'label_size' in formatting:
        label_size = formatting['label_size']
        

    for i in range(plot_dimensions[0]):
        for j in range(plot_dimensions[1]):
            ax[i][j].set_xlabel(x_labels[i][j],fontsize=label_size)
            ax[i][j].set_ylabel(y_labels[i][j],fontsize=label_size)
            ax[i][j].set_title(titles[i][j],fontsize=title_size)

    fig.supxlabel(sup_x_label,y=sup_x_label_loc,fontsize=label_size)
    fig.supylabel(sup_y_label,x=sup_y_label_loc,fontsize=label_size)
    fig.suptitle(sup_title,fontsize=title_size)


    for i in range(plot_dimensions[0]):
        for j in range(plot_dimensions[1]):
            ax[i][j].tick_params(axis='x', labelsize=tick_size)
            ax[i][j].tick_params(axis='y', labelsize=tick_size)

            if 'x_lim' in formatting:
                ax[i][j].set_xlim(formatting['x_lim'][i][j])
            if 'y_lim' in formatting:
                ax[i][j].set_ylim(formatting['y_lim'][i][j])
            if 'x_ticks' in formatting:
                if formatting['x_ticks'][i][j] is not None:
                    ax[i][j].set_xticks(formatting['x_ticks'][i][j][0])
                    ax[i][j].set_xticklabels(formatting['x_ticks'][i][j][1],fontsize=tick_size)
            if 'y_ticks' in formatting:
                ax[i][j].set_yticks(formatting['y_ticks'][i][j][0])
                ax[i][j].set_yticklabels(formatting['y_ticks'][i][j][1],fontsize=tick_size)
            if 'remove_ticks' in formatting:
                ax[i][j].tick_params(axis='both',which='both',length=0)

    if 'hide_spines' in formatting and formatting['hide_spines']:
        for i in range(plot_dimensions[0]):
            for j in range(plot_dimensions[1]):
                ax[i][j].spines['top'].set_visible(False)
                ax[i][j].spines['right'].set_visible(False)

    if 'separate_spines' in formatting and formatting['separate_spines']:
        for i in range(plot_dimensions[0]):
            for j in range(plot_dimensions[1]):
                ax[i][j].spines['left'].set_position(('outward', 5))   # Move y-axis spine outward
                ax[i][j].spines['bottom'].set_position(('outward', 5)) # Move x-axis spine outward

    if get_or_none(formatting,'has_grid'):
        for i in range(plot_dimensions[0]):
            for j in range(plot_dimensions[1]):
                ax[i][j].grid() 

    if get_or_none(formatting,'has_x_grid'):
        for i in range(plot_dimensions[0]):
            for j in range(plot_dimensions[1]):
                ax[i][j].grid(axis='x', linestyle='--',alpha=0.7) 

    if get_or_none(formatting,'has_y_grid'):
        for i in range(plot_dimensions[0]):
            for j in range(plot_dimensions[1]):
                ax[i][j].grid(axis='y', linestyle='--',alpha=0.7) 


    return fig, ax

def create_legend(fig,ax,plot_dimensions,formatting):
    """Add the legend to a figure/plot
    
    Arguments:  
        fig: Matplotlib figure object
        ax: List of axes
        plot_dimensions: Tuple with rows, columns for axes
        formatting: Dictionary with legend-specific options
            style_size: 'paper' or 'presentation' depending on size
            type: is_global or is_local
            show_point: In the legend, whether to show the dot
            loc: Legend parameter
            ncol: Legend parameter
            bbox_to_anchor: Legend parameter"""

    if formatting['style_size'] == 'paper':
        legend_size = 12
    elif formatting['style_size'] == 'presentation':
        legend_size = 16
    
    if 'fontsize' in formatting:
        legend_size = formatting['fontsize']

    if formatting['type'] == 'is_global':
        handles, labels = ax[0][0].get_legend_handles_labels()

        if get_or_none(formatting,'show_point'):
            handles, labels = ax[0][0].get_legend_handles_labels()
            custom_lines = [Line2D([0], [0], color=handles[i].get_color(), linestyle=handles[i].get_linestyle(), marker=markers[i]) for i in range(len(handles))]
            fig.legend(custom_lines, labels,loc=formatting['loc'],
                ncol=formatting['ncol'],
                bbox_to_anchor=formatting['bbox_to_anchor'],
                fontsize=legend_size)
        else:
            fig.legend(handles, labels, loc=formatting['loc'],
                    ncol=formatting['ncol'],
                    bbox_to_anchor=formatting['bbox_to_anchor'],
                    fontsize=legend_size)
    elif formatting['type'] == 'is_local':
        for i in range(plot_dimensions[0]):
            for j in range(plot_dimensions[1]):
                if get_or_none(formatting,'show_point'):
                    handles, labels = ax[i][j].get_legend_handles_labels()
                    custom_lines = [Line2D([0], [0], color=handles[i].get_color(), linestyle=handles[i].get_linestyle(), marker=markers[i]) for i in range(len(handles))]
                    ax[i][j].legend(custom_lines, labels,loc=formatting['loc'],
                        ncol=formatting['ncol'],
                        bbox_to_anchor=formatting['bbox_to_anchor'],
                        fontsize=legend_size)

                else:
                    if len(ax[i][j].get_legend_handles_labels()[0]) > 0:
                        ax[i][j].legend(loc=formatting['loc'],
                            ncol=formatting['ncol'],
                            bbox_to_anchor=formatting['bbox_to_anchor'][i][j],
                            fontsize=legend_size)
    return fig, ax