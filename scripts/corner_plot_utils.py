"""
Corner Plot Utilities for Gravitational Lensing Parameter Estimation

This module provides utilities for creating corner plots with:
- Multiple datasets (e.g., HMC vs Fisher)
- Custom legends
- Parameter ranges
- Grouped parameter plots
- Truth value overlays
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import corner
from typing import Dict, List, Tuple, Optional, Union


def add_corner_legend(
    fig, 
    labels: List[str], 
    colors: List[str], 
    loc: str = 'upper right', 
    bbox: Tuple[float, float] = (0.995, 0.995), 
    fontsize: int = 10
):
    """Add a legend to a corner plot using colored patches.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure returned by corner.corner.
    labels : list[str]
        Labels to show in the legend.
    colors : list[str]
        Colors for each label (same order as labels).
    loc : str
        Legend location keyword passed to matplotlib.
    bbox : tuple[float, float]
        (x, y) coordinates for bbox_to_anchor in figure coords.
    fontsize : int
        Legend font size.

    Returns
    -------
    leg : matplotlib.legend.Legend
        The legend object.
    """
    handles = [Patch(facecolor=c, edgecolor=c, label=l) for l, c in zip(labels, colors)]
    axes = fig.get_axes()
    if not axes:
        return None
    leg = axes[0].legend(
        handles=handles,
        loc=loc,
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=fontsize,
        bbox_to_anchor=bbox,
        bbox_transform=fig.transFigure,
    )
    for text, color in zip(leg.get_texts(), colors):
        text.set_color(color)
    return leg


def set_corner_axis_ranges(
    fig, 
    labels: List[str], 
    param_ranges: Dict[str, Tuple[float, float]], 
    verbose: bool = False
):
    """Set x and y axis ranges for specific parameters in a corner plot.
    
    This function automatically matches parameters from param_ranges that are present
    in the plot's labels. You can pass a full param_ranges dict with all parameters,
    and only the ones present in the current plot will be applied.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure returned by corner.corner.
    labels : list[str]
        List of parameter labels in the same order as they appear in the corner plot.
        This should be the labels used when creating the corner plot.
    param_ranges : dict[str, tuple[float, float]]
        Dictionary mapping parameter names to (xmin, xmax) tuples.
        Can contain more parameters than are in the plot - only matching ones will be applied.
        For example: {'noise_sigma_bkg': (0.01, 0.05), 'lens_theta_E': (1.0, 3.0)}
    verbose : bool
        If True, print debug information.
    """
    axes = fig.get_axes()
    if not axes:
        if verbose:
            print("Warning: No axes found in figure.")
        return
    
    # Get the number of parameters (corner plots are square grids)
    n_params = len(labels)
    
    # Find the index of each parameter in the labels list
    param_indices = {label: i for i, label in enumerate(labels)}
    
    # Filter param_ranges to only include parameters that are in the current plot
    applicable_ranges = {k: v for k, v in param_ranges.items() if k in param_indices}
    
    if verbose:
        print(f"Found {len(axes)} axes for {n_params} parameters")
        print(f"Expected {n_params * n_params} axes for full grid")
        print(f"Plot labels: {labels}")
        print(f"Requested ranges for {len(param_ranges)} parameters")
        print(f"Applying ranges for {len(applicable_ranges)} parameters present in plot: {list(applicable_ranges.keys())}")
        if len(applicable_ranges) < len(param_ranges):
            skipped = set(param_ranges.keys()) - set(applicable_ranges.keys())
            print(f"Skipped {len(skipped)} parameters not in current plot: {list(skipped)}")
    
    if not applicable_ranges:
        if verbose:
            print("No matching parameters found. No ranges will be set.")
        return
    
    # Set ranges for each applicable parameter
    for param_name, (xmin, xmax) in applicable_ranges.items():
        
        param_idx = param_indices[param_name]
        if verbose:
            print(f"Setting range for '{param_name}' (index {param_idx}): ({xmin}, {xmax})")
        
        # In corner plots, axes are arranged in a grid where:
        # - Row i corresponds to parameter i (y-axis)
        # - Column j corresponds to parameter j (x-axis)  
        # - Axis at grid position (i, j) has index: i * n_params + j
        # - Only lower triangle is shown (i >= j), but all axes exist in the list
        
        # Set range for diagonal plot (histogram) - parameter vs itself
        diag_idx = param_idx * n_params + param_idx
        if diag_idx < len(axes):
            axes[diag_idx].set_xlim(xmin, xmax)
            if verbose:
                print(f"  Set diagonal axis {diag_idx} xlim to ({xmin}, {xmax})")
        
        # Set x-axis range for all plots in the column (for this parameter as x-axis)
        # These are plots where param_idx is the column (x-axis)
        for row in range(param_idx + 1, n_params):
            col_idx = row * n_params + param_idx
            if col_idx < len(axes):
                axes[col_idx].set_xlim(xmin, xmax)
                if verbose:
                    print(f"  Set column axis {col_idx} (row {row}, col {param_idx}) xlim to ({xmin}, {xmax})")
        
        # Set y-axis range for all plots in the row (for this parameter as y-axis)
        # These are plots where param_idx is the row (y-axis)
        for col in range(param_idx):
            row_idx = param_idx * n_params + col
            if row_idx < len(axes):
                axes[row_idx].set_ylim(xmin, xmax)
                if verbose:
                    print(f"  Set row axis {row_idx} (row {param_idx}, col {col}) ylim to ({xmin}, {xmax})")
    
    # Force figure update
    try:
        fig.canvas.draw()
    except:
        plt.draw()


def create_corner_ranges(
    labels: List[str], 
    param_ranges: Dict[str, Tuple[float, float]], 
    default_range: Optional[Tuple[float, float]] = None
) -> List[Optional[Tuple[float, float]]]:
    """Create a ranges list for corner.corner() from a dictionary of parameter ranges.
    
    This function automatically matches parameters from param_ranges that are present
    in labels. You can pass a full param_ranges dict with all parameters, and only
    the ones present in labels will be used.
    
    Parameters
    ----------
    labels : list[str]
        List of parameter labels in the same order as they appear in the corner plot.
    param_ranges : dict[str, tuple[float, float]]
        Dictionary mapping parameter names to (min, max) tuples.
        Can contain more parameters than are in labels - only matching ones will be used.
        Parameters in labels but not in this dict will use default_range if provided, or None for auto-range.
    default_range : tuple[float, float] or None
        Default range to use for parameters in labels but not specified in param_ranges.
        If None, those parameters will use automatic range (None in the list).
    
    Returns
    -------
    ranges : list[tuple[float, float] | None]
        List of (min, max) tuples or None in the same order as labels.
        None means use automatic range for that parameter.
    """
    if not param_ranges:
        return None
    
    ranges = []
    for label in labels:
        if label in param_ranges:
            ranges.append(param_ranges[label])
        elif default_range is not None:
            ranges.append(default_range)
        else:
            ranges.append(None)  # Use automatic range for unspecified parameters
    
    return ranges


def add_truth_lines(
    fig, 
    labels: List[str], 
    truths: List[Optional[float]], 
    color: str = 'red', 
    linestyle: str = '--',
    alpha: float = 0.5
):
    """Manually add truth value lines to a corner plot.
    
    This function adds vertical and horizontal lines to indicate true parameter values.
    Useful for avoiding arviz backend bugs when passing truths directly to corner.corner().
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure returned by corner.corner.
    labels : list[str]
        List of parameter labels in the same order as they appear in the corner plot.
    truths : list[float | None]
        List of truth values for each parameter. Use None for parameters without truth values.
    color : str
        Color for the truth lines.
    linestyle : str
        Line style for the truth lines.
    alpha : float
        Transparency for off-diagonal truth lines.
    """
    if not truths or all(t is None for t in truths):
        return
    
    n_params = len(labels)
    axes = np.array(fig.get_axes()).reshape(n_params, n_params)
    
    for k1 in range(n_params):
        if truths[k1] is not None:
            # Add vertical line to diagonal (histogram)
            axes[k1, k1].axvline(truths[k1], color=color, linestyle=linestyle, linewidth=2)
            # Add vertical lines to all plots in the column
            for k2 in range(k1 + 1, n_params):
                if truths[k1] is not None:
                    axes[k2, k1].axvline(truths[k1], color=color, linestyle=linestyle, alpha=alpha)
                if truths[k2] is not None:
                    axes[k2, k1].axhline(truths[k2], color=color, linestyle=linestyle, alpha=alpha)


def plot_grouped_corner(
    samples_dict: Dict[str, np.ndarray],
    param_groups: Dict[str, List[str]],
    truths_dict: Optional[Dict[str, Dict[str, float]]] = None,
    color: str = '#2c3e50',
    title: Optional[str] = None,
    show_titles: bool = True,
    title_kwargs: Optional[Dict] = None,
    title_fmt: str = '.3f',
    quantiles: List[float] = [0.05, 0.5, 0.975],
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    truth_color: str = 'red',
    save_path: Optional[str] = None,
    **corner_kwargs
) -> List[plt.Figure]:
    """Create grouped corner plots for different parameter categories.
    
    Parameters
    ----------
    samples_dict : dict[str, np.ndarray]
        Dictionary mapping parameter names to sample arrays.
    param_groups : dict[str, list[str]]
        Dictionary mapping group names to lists of parameter names.
        Example: {'lens_mass': ['lens_theta_E', 'lens_e1', ...], ...}
    truths_dict : dict[str, dict[str, float]] or None
        Dictionary mapping group names to dictionaries of truth values.
        Example: {'lens_mass': {'lens_theta_E': 2.0, ...}, ...}
    color : str
        Color for the corner plot.
    title : str or None
        Optional title prefix for each group plot.
    show_titles : bool
        Whether to show parameter titles on plots.
    title_kwargs : dict or None
        Keyword arguments for title formatting.
    title_fmt : str
        Format string for titles.
    quantiles : list[float]
        Quantiles to show in titles.
    param_ranges : dict[str, tuple[float, float]] or None
        Dictionary mapping parameter names to (min, max) ranges.
    truth_color : str
        Color for truth value lines.
    save_path : str or None
        If provided, save each plot with this path pattern (use {group_name} placeholder).
    **corner_kwargs
        Additional keyword arguments passed to corner.corner().
    
    Returns
    -------
    figures : list[matplotlib.figure.Figure]
        List of figure objects for each group.
    """
    if title_kwargs is None:
        title_kwargs = {'fontsize': 10}
    
    figures = []
    
    for group_name, params in param_groups.items():
        if len(params) < 1:
            continue
        
        # Filter params to only those present in samples
        params = [p for p in params if p in samples_dict]
        if len(params) < 1:
            continue
        
        # Get samples for this group
        samples_grouped = {p: samples_dict[p] for p in params}
        samples_array = np.column_stack([np.asarray(samples_grouped[p]) for p in params])
        
        # Get truths for this group
        truths_grouped = truths_dict.get(group_name) if truths_dict else None
        truths_list = [truths_grouped.get(p) if truths_grouped and p in truths_grouped else None 
                       for p in params] if truths_grouped else None
        
        # Create corner plot
        fig = corner.corner(
            samples_array, 
            labels=params, 
            color=color, 
            truth_color=truth_color,
            show_titles=show_titles,
            title_kwargs=title_kwargs,
            title_fmt=title_fmt,
            quantiles=quantiles,
            **corner_kwargs
        )
        
        # Add truth lines manually (more reliable than passing truths to corner.corner)
        if truths_list is not None:
            add_truth_lines(fig, params, truths_list, color=truth_color)
        
        # Set parameter ranges if provided
        if param_ranges:
            set_corner_axis_ranges(fig, params, param_ranges)
        
        # Add title
        if title:
            plt.suptitle(f'{title} - {group_name.replace("_", " ").title()}', 
                        fontsize=12, y=1.02)
        else:
            plt.suptitle(f'{group_name.replace("_", " ").title()}', 
                        fontsize=12, y=1.02)
        
        # Save if requested
        if save_path:
            save_name = save_path.format(group_name=group_name)
            plt.savefig(save_name, bbox_inches='tight', dpi=300)
            print(f"Saved: {save_name}")
        
        figures.append(fig)
    
    return figures


def plot_comparison_corner(
    samples_dict1: Dict[str, np.ndarray],
    samples_dict2: Dict[str, np.ndarray],
    param_groups: Dict[str, List[str]],
    labels: Tuple[str, str] = ('HMC-EM', 'Fisher-EM'),
    colors: Tuple[str, str] = ('#3B5BA7', '#D2691E'),
    truths_dict: Optional[Dict[str, Dict[str, float]]] = None,
    truth_color: str = 'red',
    show_titles: bool = True,
    title_kwargs: Optional[Dict] = None,
    title_fmt: str = '.3f',
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    save_path: Optional[str] = None,
    **corner_kwargs
) -> List[plt.Figure]:
    """Create grouped corner plots comparing two sets of samples (e.g., HMC vs Fisher).
    
    Parameters
    ----------
    samples_dict1 : dict[str, np.ndarray]
        First set of samples (e.g., HMC samples).
    samples_dict2 : dict[str, np.ndarray]
        Second set of samples (e.g., Fisher samples).
    param_groups : dict[str, list[str]]
        Dictionary mapping group names to lists of parameter names.
    labels : tuple[str, str]
        Labels for the two datasets.
    colors : tuple[str, str]
        Colors for the two datasets.
    truths_dict : dict[str, dict[str, float]] or None
        Dictionary mapping group names to dictionaries of truth values.
    truth_color : str
        Color for truth value lines.
    show_titles : bool
        Whether to show parameter titles on plots.
    title_kwargs : dict or None
        Keyword arguments for title formatting.
    title_fmt : str
        Format string for titles.
    param_ranges : dict[str, tuple[float, float]] or None
        Dictionary mapping parameter names to (min, max) ranges.
    save_path : str or None
        If provided, save each plot with this path pattern (use {group_name} placeholder).
    **corner_kwargs
        Additional keyword arguments passed to corner.corner().
    
    Returns
    -------
    figures : list[matplotlib.figure.Figure]
        List of figure objects for each group.
    """
    if title_kwargs is None:
        title_kwargs = {'fontsize': 10}
    
    figures = []
    
    for group_name, params in param_groups.items():
        # Filter params to only those present in both sample sets
        params = [p for p in params if p in samples_dict1 and p in samples_dict2]
        if len(params) < 1:
            continue
        
        # Get samples for this group
        samples_grouped1 = {p: samples_dict1[p] for p in params}
        samples_grouped2 = {p: samples_dict2[p] for p in params}
        
        # Convert to arrays
        samples_array1 = np.column_stack([np.asarray(samples_grouped1[p]) for p in params])
        samples_array2 = np.column_stack([np.asarray(samples_grouped2[p]) for p in params])
        
        # Get truths for this group
        truths_grouped = truths_dict.get(group_name) if truths_dict else None
        truths_list = [truths_grouped.get(p) if truths_grouped and p in truths_grouped else None 
                       for p in params] if truths_grouped else None
        
        # Create first corner plot
        fig = corner.corner(
            samples_array1, 
            labels=params, 
            color=colors[0], 
            truth_color=truth_color,
            show_titles=show_titles,
            title_kwargs=title_kwargs,
            title_fmt=title_fmt,
            quantiles=[0.05, 0.5, 0.975],
            **corner_kwargs
        )
        
        # Add truth lines manually
        if truths_list is not None:
            add_truth_lines(fig, params, truths_list, color=truth_color)
        
        # Overlay second dataset
        _ = corner.corner(
            samples_array2, 
            labels=params, 
            color=colors[1], 
            fig=fig,
            show_titles=show_titles,
            title_kwargs=title_kwargs,
            title_fmt=title_fmt,
            **corner_kwargs
        )
        
        # Add legend
        add_corner_legend(
            fig=fig,
            labels=list(labels),
            colors=list(colors),
            loc='upper right',
            bbox=(0.995, 0.995),
            fontsize=10,
        )
        
        # Set parameter ranges if provided
        if param_ranges:
            set_corner_axis_ranges(fig, params, param_ranges)
        
        # Add title
        plt.suptitle(f'{group_name.replace("_", " ").title()}', 
                    fontsize=12, y=1.02)
        
        # Save if requested
        if save_path:
            save_name = save_path.format(group_name=group_name)
            plt.savefig(save_name, bbox_inches='tight', dpi=300)
            print(f"Saved: {save_name}")
        
        figures.append(fig)
    
    return figures


def create_default_param_groups(samples_dict: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
    """Create default parameter groups from a samples dictionary.
    
    Parameters
    ----------
    samples_dict : dict[str, np.ndarray]
        Dictionary mapping parameter names to sample arrays.
    
    Returns
    -------
    param_groups : dict[str, list[str]]
        Dictionary mapping group names to parameter lists.
    """
    param_groups = {
        'lens_light': [k for k in samples_dict.keys() if k.startswith('light_')],
        'source_light': [k for k in samples_dict.keys() if k.startswith('source_')],
        'lens_mass': [k for k in samples_dict.keys() if k.startswith('lens_')],
        'cosmology_params': [k for k in samples_dict.keys() if k in ['T_star', 'dL']],
        'GW_image_positions': [k for k in samples_dict.keys() 
                              if k in ['image_x1', 'image_y1', 'image_x2', 'image_y2', 
                                      'image_x3', 'image_y3', 'image_x4', 'image_y4']],
        'GW_source_position': [k for k in samples_dict.keys() if k in ['y0gw', 'y1gw']],
        'Noise_parameters': [k for k in samples_dict.keys() if k in ['noise_sigma_bkg']],
    }
    
    # Filter out empty groups
    param_groups = {k: v for k, v in param_groups.items() if len(v) > 0}
    
    return param_groups


def plot_custom_params(
    samples: Dict[str, np.ndarray],
    params_to_plot: List[str],
    truths: Optional[Dict[str, float]] = None,
    color: str = '#2c3e50',
    truth_color: str = 'red',
    show_titles: bool = True,
    title_kwargs: Optional[Dict] = None,
    title_fmt: str = '.3f',
    quantiles: List[float] = [0.05, 0.5, 0.975],
    save_path: Optional[str] = None,
    **corner_kwargs
) -> plt.Figure:
    """Plot a corner plot for a custom subset of parameters.
    
    This is a simple, direct function for plotting specific parameters without
    needing to create parameter groups.
    
    Parameters
    ----------
    samples : dict[str, np.ndarray]
        Dictionary mapping parameter names to sample arrays.
    params_to_plot : list[str]
        List of parameter names to plot, e.g., ['lens_theta_E', 'lens_e1', 'lens_e2']
    truths : dict[str, float] or None
        Dictionary of truth values {param_name: value}. Optional.
    color : str
        Color for the corner plot.
    truth_color : str
        Color for truth value lines.
    show_titles : bool
        Whether to show parameter titles on plots.
    title_kwargs : dict or None
        Keyword arguments for title formatting.
    title_fmt : str
        Format string for titles.
    quantiles : list[float]
        Quantiles to show in titles.
    save_path : str or None
        Path to save the plot. If None, plot is not saved.
    **corner_kwargs
        Additional keyword arguments passed to corner.corner().
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    
    Example
    -------
    >>> params_to_plot = ['lens_theta_E', 'lens_e1', 'lens_e2']
    >>> fig = plot_custom_params(
    ...     samples=samples,
    ...     params_to_plot=params_to_plot,
    ...     truths=input_params,
    ...     save_path='../plots/corner_custom_params.pdf'
    ... )
    >>> plt.show()
    """
    if title_kwargs is None:
        title_kwargs = {'fontsize': 10}
    
    # Filter params to only those present in samples
    params_to_plot = [p for p in params_to_plot if p in samples]
    if len(params_to_plot) < 1:
        raise ValueError("No parameters from params_to_plot found in samples dictionary")
    
    # Extract samples for these parameters
    samples_array = np.column_stack([np.asarray(samples[p]) for p in params_to_plot])
    
    # Extract truth values (optional)
    truths_list = [truths.get(p) if truths and p in truths else None 
                   for p in params_to_plot]
    
    # Create corner plot
    fig = corner.corner(
        samples_array,
        labels=params_to_plot,
        color=color,
        truth_color=truth_color,
        show_titles=show_titles,
        title_kwargs=title_kwargs,
        title_fmt=title_fmt,
        quantiles=quantiles,
        **corner_kwargs
    )
    
    # Add truth lines (more reliable than passing truths to corner.corner)
    if any(t is not None for t in truths_list):
        add_truth_lines(fig, params_to_plot, truths_list, color=truth_color)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
    
    return fig

