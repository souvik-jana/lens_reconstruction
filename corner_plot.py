import matplotlib
import matplotlib.pyplot as plt
import corner
import numpy as np
import os

class CornerPlot:
    def __init__(self, list_of_sample_dicts=None, sample_labels=None,var_names=None, ordered_dict=None, output_dir=None, output_name=None, ranges=None, Save_plot=False):
        self.list_of_sample_dicts = list_of_sample_dicts
        self.sample_labels = sample_labels
        self.var_names = var_names
        self.ranges = ranges
        self.ordered_dict = ordered_dict
        self.output_dir = output_dir
        self.output_name = output_name
        self.Save_plot = Save_plot
    
    def get_contrasting_colors(self, n=10):
        """
        Returns a list of n visually contrasting colors in hex format.
        """
        cmap = matplotlib.cm.get_cmap('tab10')
        colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(n)]
        return colors

    
    def stack_samples(self, sample_dict, keys):
        arr = np.column_stack([np.asarray(sample_dict[k]) for k in keys])
        return arr

    def plot_corner(self):
        colors = self.get_contrasting_colors(len(self.list_of_sample_dicts))
        fig = None
        legend_elements = []

        if len(self.var_names) <=3:
            bbox_to_anchor = (0.99, 0.98)
            markerscale = 1.5
            handlelength = 1.5
            handletextpad = 0.5
            columnspacing = 1.0
            fontsize = 12
        else:
            bbox_to_anchor = (0.97, 0.98)
            markerscale = 3
            handlelength = 2.5
            handletextpad = 1.0
            columnspacing = 2.0
            fontsize = 18   
        
        for i, sample_dict in enumerate(self.list_of_sample_dicts):
            sample_arr = self.stack_samples(sample_dict, self.var_names)

            if self.ordered_dict is not None:
                truths = [self.ordered_dict[k] for k in self.var_names]
            else:
                truths = None
            if self.ranges is not None:
                ranges = self.ranges
            else:
                ranges = None
            
            # Use sample label for histogram label
            hist_label = self.sample_labels[i] if self.sample_labels else f"Sample {i+1}"
            
            fig = corner.corner(sample_arr, color=colors[i], labels=self.var_names, fill_contours=True, plot_datapoints=False, hist_kwargs=dict(label=hist_label, linewidth=2), truths=truths, truth_color='black', ranges=ranges, fig=fig)
            
            # Create legend element with thick small bar instead of line
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor='none', label=hist_label))

        
        # Add legend to the figure with custom styling
        if legend_elements:
            legend = fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=bbox_to_anchor, 
                              frameon=False, fontsize=fontsize, markerscale=markerscale, 
                              handlelength=handlelength, handletextpad=handletextpad, columnspacing=columnspacing)
        
        if self.Save_plot:
            fig.savefig(os.path.join(self.output_dir, self.output_name), dpi=250, bbox_inches='tight', facecolor='white')
            print(f"Saved corner plot: {self.output_name} at {self.output_dir}")
        
        # Don't close the figure so it can be displayed in notebook
        # plt.close(fig)  # Commented out to allow display in notebook
        print('Corner plot created')
        return fig
