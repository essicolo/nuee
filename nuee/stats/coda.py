import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## REMOVE CLASSES
def closure(parts, total=1):
    """
    Closes the simplex to a constant sum.
    """
    # if parts is a pd.Series, convert to a 1 row pd.DataFrame
    if isinstance(parts, pd.Series):
        parts = pd.DataFrame(parts).T
    if isinstance(parts, np.ndarray):
        # if parts is a 1D numpy array, convert to 2D
        if len(parts.shape) == 1:
            parts = np.array([parts])
        parts_column_names = ['P%d' % (i+1) for i in range(parts.shape[1])]
        parts = pd.DataFrame(parts, columns = parts_column_names)
    condition = parts >= 0
    if not condition.all().all():
        print("Warning. There are negative, infitite of missing values.")
    closed = total * parts.apply(lambda x: x / np.sum(x), 1)
    return closed

def tern2cart(parts):
    """
    Internal function to compute ternary plot coordinates from 3 parts.
    """
    if len(parts.shape) == 1:
        parts = np.array([parts])
    if not parts.shape[1] == 3:
        raise ValueError("parts must have exactly 3 parts.")
    comp = closure(parts)
    x = comp.iloc[:, 0].as_matrix()
    y = comp.iloc[:, 1].as_matrix()
    z = comp.iloc[:, 2].as_matrix()
    xcoord = (2 * y + z) / (2 * (x + y + z))
    ycoord = np.sqrt(3) * z / (2 * (x + y + z))
    return np.array([xcoord, ycoord]).T


class PlotTriangle():
    """
    A class makes a more portable triangle
    """
    def __init__(self, grid_by = 0.1, tick_length = 0.01, labels = ['A', 'B', 'C'],
                  show_arrows = True, show_grid = True, show_ticks = True,
                  show_tick_labels = True):
        self.grid_by = grid_by
        self.tick_length = tick_length
        self.labels = labels
        self.show_arrows = show_arrows
        self.show_grid = show_grid
        self.show_ticks = show_ticks
        self.show_tick_labels = show_tick_labels

    def plot_triangle(self):
        """
        Internal function to plot the trangle background as basis for
        tenrary diagrams.
        """
        plt.axes().set_aspect('equal')
        plt.xlim([0, 1])
        plt.ylim([-0.15 * np.sqrt(3)/2, np.sqrt(3)/2])
        plt.axis('off')
        plt.plot([0, 0.5, 1, 0], [0, np.sqrt(3)/2, 0, 0], color = "black")

        if self.grid_by is not None:
            ngrid = int(1 + 1 / self.grid_by)
            xseq = np.linspace(0.0, 1.0, ngrid)
            yseq = xseq * np.sqrt(3)/2
            xlseq = xseq/2
            xrseq = 0.5 + xlseq

            for i in range(0, ngrid):
                if self.show_grid:
                    # grid
                    plt.plot([xlseq[i], xseq[i]], [yseq[i], 0], color = "#E3E3E3")
                    plt.plot([xrseq[i], xseq[i]], [yseq[::-1][i], 0], color = "#E3E3E3")
                    plt.plot([xlseq[i], xrseq[::-1][i]], [yseq[i], yseq[i]], color = "#E3E3E3")

                if self.show_ticks:
                    plt.plot([xlseq[i], xlseq[i] - self.tick_length], [yseq[i], yseq[i]], color = "#505050") # left ticks
                    plt.plot([xrseq[i], xrseq[i] + self.tick_length * np.sqrt(3)/2], [yseq[::-1][i], yseq[::-1][i] + self.tick_length / (np.sqrt(3)/2) ], color = "#505050") # right ticks
                    plt.plot([xseq[i], xseq[i] + self.tick_length * np.sqrt(3)/2], [0, 0 - self.tick_length / (np.sqrt(3)/2) ], color = "#505050")

                if self.show_tick_labels:
                    plt.text(xlseq[i] - 0.035, yseq[i],
                             s = str(xseq[i]),#[(rounding % xseq[i])],
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize = 12, color = '#505050') # left labels
                    plt.text(x = xrseq[::-1][i] + 0.025,
                             y = yseq[i] + 0.035,
                             s = str(xseq[::-1][i]), #[(rounding % xseq[::-1][i])],
                             rotation = 60,
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize = 12, color = '#505050') # right labels
                    plt.text(x = xseq[i] + 0.015,
                             y = 0 - 0.035,
                             s = str(xseq[::-1][i]), # [(rounding % xseq[::-1][i])],
                             rotation = -60,
                             horizontalalignment='center',
                             verticalalignment='center',
                             fontsize = 12, color = '#505050') # bottom labels

        if self.labels is not None:
            xseq = [0.5, 0.87, 0.12]
            yseq = [-0.13, np.sqrt(3)/4, np.sqrt(3)/4]
            rotseq = [0, -60, 60]
            for i in range(0, len(self.labels)):
                plt.text(xseq[i], yseq[i], s = self.labels[i], rotation = rotseq[i],
                         horizontalalignment='center',
                         verticalalignment='center',
                        fontsize = 18)

        if self.show_arrows:
            arrrow_paw_len = 0.03
            arrow_offset = 0.1
            bottom_x_start, bottom_x_end = [0.7, 0.3]
            bottom_y_start, bottom_y_end = [0 - arrow_offset, 0 - arrow_offset]
            left_x_start, left_x_end = [0.15 - arrow_offset, 0.35 - arrow_offset]
            left_y_start, left_y_end = [np.sqrt(3) * (left_x_start + arrow_offset), np.sqrt(3) * (left_x_end + arrow_offset)]
            right_x_start, right_x_end = [0.65 + arrow_offset, 0.85 + arrow_offset]
            right_y_start, right_y_end = [left_y_end, left_y_start]
            plt.plot([bottom_x_start, bottom_x_start - arrrow_paw_len * np.cos(np.pi/3)],
                     [bottom_y_start, bottom_y_start + arrrow_paw_len * np.sin(np.pi/3)], color = "black") # bottom paw
            plt.arrow(x = bottom_x_start, y = bottom_y_start,
                      dx = bottom_x_end - bottom_x_start, dy = bottom_y_end - bottom_y_start,
                     head_width=0.02, color = 'black')
            plt.plot([left_x_start, left_x_start + arrrow_paw_len],
                     [left_y_start, left_y_start], color = "black") # left paw
            plt.arrow(x = left_x_start, y = left_y_start,
                      dx = left_x_end - left_x_start, dy = left_y_end - left_y_start,
                     head_width=0.02, color = 'black')
            plt.plot([right_x_start, right_x_start - arrrow_paw_len * np.cos(np.pi/3)],
                     [right_y_start, right_y_start - arrrow_paw_len * np.sin(np.pi/3)], color = "black")
            plt.arrow(x = right_x_start, y = right_y_start,
                      dx = right_x_end - right_x_start, dy = right_y_end - right_y_start,
                     head_width=0.02, color = 'black')


# plot user-defined functions
# https://stackoverflow.com/questions/17141001/is-it-possible-to-plot-within-user-defined-function-with-python-and-matplotlib

# pairsplot
# https://stackoverflow.com/questions/2682144/matplotlib-analog-of-rs-pairs
def plot_comp(parts):
    composition = closure(parts, total=1)
    number_of_components = composition.shape[1]
    if not number_of_components == 3:
        raise ValueError("parts must have exactly 3 parts.")
    else:
        coordinates = tern2cart(composition)
    plt.scatter(coordinates[:, 0], coordinates[:, 1])
