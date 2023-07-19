from data_preprocessing import Preprocess
import math as m
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import pyvista as pv
from scipy import stats
import seaborn as sns


class PreliminaryPlots:
    """
    Class for plotting meshes and creating GIFs with curvedness as the activate scalar.
    """
    def __init__(self, mesh_lst):
        self.mesh_lst = mesh_lst

        # Activate curvedness as the scalar (heatmap variable)
        for i in self.mesh_lst:
            i.set_active_scalars('curvedness_VV')

        # Define scalar bar arguments
        self.scalar_args = dict(title='Curvedness', color='white', font_family='times', width=0.7, position_x=0.15,
                                n_labels=5, fmt='%.2e')

    def plot_single(self, start=0, stop=-1):
        """
        Plot one mesh per plot with curvedness as the active scalar. When the interactive plot window is closed, plot
        the subsequent mesh.

        Parameters:
            - start: Index of mesh to start plotting from. If None given, start plotting from first mesh.
            - stop: Index of mesh to stop plotting from. If None given, stop plotting from last mesh.

        Examples:
            - plot_single(start=6, stop=10) plots meshes 7-10.
        """
        for i in self.mesh_lst[start:stop]:
            pv.plot(i, text=str(self.mesh_lst.index(i) + 1), window_size=[3024, 1964], background='black', zoom=0.8,
                    cpos=[-1.0, 0.0, 0], scalar_bar_args=self.scalar_args)

    def make_gifs(self):
        """
        Make one GIF animation per mesh with curvedness as the active scalar.
        """
        print('Starting create_gifs')

        # Specify the angle step for animation
        angle_step = 2

        for i in self.mesh_lst:
            print(f'Making GIF {self.mesh_lst.index(i) + 1} out of {int(len(self.mesh_lst))}')
            pl = pv.Plotter(off_screen=True)

            # Add mesh to the plot
            pl.add_mesh(i, scalar_bar_args=self.scalar_args)

            # Open a GIF file to save the frames
            pl.open_gif(f'mesh{str(self.mesh_lst.index(i) + 1)}.gif', fps=12, subrectangles=True)

            # Generate frames by rotating the camera position around the meshes
            for theta in np.linspace(0, 360, int(360 / angle_step)):
                # Convert the angle from degrees to radians
                theta = theta * m.pi / 180

                # Calculate the x and y coordinates on the unit circle
                x = np.cos(theta)
                y = np.sin(theta)

                # Define the new camera position with the x, y, and z values
                cpos_new = (x, y, m.pi / 6)

                # Update camera position and settings for each frame
                pl.camera_position = cpos_new
                pl.camera.zoom(0.6)
                pl.window_size = [3024, 1964]
                pl.set_background('black')
                pl.write_frame()
            pl.close()
        print('Completed create_gifs')

    def plot_dual(self, left_mesh=0):
        """
        Plot two meshes per plot, where left_mesh is on the left and the subsequent mesh is on the right. Curvedness is
        the active scalar.

        Parameters:
            - left_mesh: Index of mesh on the left.
        """
        # Define scalar bar arguments
        scalar_args = dict(title='Curvedness', title_font_size=36, label_font_size=30, color='white',
                           font_family='times', width=0.7, position_x=0.15, n_labels=5, fmt='%.2e')

        # Create an empty 1 row x 2 column plot
        pl = pv.Plotter(shape=(1, 2))

        # Add left_mesh to the left plot
        pl.subplot(0, 0)
        pl.add_mesh(self.mesh_lst[left_mesh], scalar_bar_args=scalar_args)

        # Add the subsequent mesh to the right plot.
        pl.subplot(0, 1)
        pl.add_mesh(self.mesh_lst[left_mesh + 1], scalar_bar_args=scalar_args)
        pl.set_background('black')
        pl.show(window_size=[3024, 1964])


class Figures:
    """
    Class for making figures.
    """

    def __init__(self, all_df):
        self.all_df = all_df

        # Define colour palette, style, and figure size
        self.replicate_palette = sns.color_palette("Set2")
        sns.set_style('white')
        sns.set(rc={'figure.figsize': (5, 6)})

    def make_figure_3a(self):
        fig, ax = plt.subplots()

        sns.boxplot(self.all_df, x='Old or new pole', y='Curvedness', hue='Replicate', palette=self.replicate_palette,
                    ax=ax)
        sns.stripplot(self.all_df, x='Old or new pole', y='Curvedness', hue='Replicate', dodge=True, jitter=False,
                      palette=self.replicate_palette, linewidth=0.8, legend=False, ax=ax)

        ax.set_xlabel('')
        ax.set_ylabel('Mean curvedness')
        plt.subplots_adjust(left=0.16)
        plt.show()

    def make_figure_3b(self):
        fig, ax = plt.subplots()

        old_new_df = self.all_df[(self.all_df['Old or new pole'] == 'Old pole')]

        sns.stripplot(old_new_df, x='Replicate', y='Ratio', palette=self.replicate_palette, edgecolor='gray',
                      linewidth=0.8, ax=ax)
        ax.axhline(1, c='grey', ls='--')

        ax.set_xlabel('Replicate')
        ax.set_ylabel('Mean curvedness ratio (old pole : new pole)')
        ax.set_ylim([0.8, 2])
        plt.subplots_adjust(left=0.16)
        plt.show()

    def make_figure_4(self):
        fig, ax = plt.subplots()

        adam_df = pd.DataFrame({
            'Distance from midpoint': [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'Mean ratio': [1.729062209, 1.74092948, 1.336157617, 1.197290218, 1.227960662, 1.327210259, 1.218841519,
                               1.578729596, 2.311785647, 2.492214359, 2.113425674],
            'CI upper': [2.264502453, 2.280333658, 1.627236697, 1.359030103, 1.370235027, 1.568473785, 1.448765665,
                         2.445897466, 3.932894728, 4.095260704, 3.605171437],
            'CI lower': [1.193621965, 1.201525301, 1.045078537, 1.035550333, 1.085686297, 1.085946733, 0.988917373,
                         0.711561726, 0.690676566, 0.889168013, 0.621679911]
        })

        sns.lineplot(adam_df, x='Distance from midpoint', y='Mean ratio')
        plt.fill_between(adam_df['Distance from midpoint'], adam_df['CI upper'], adam_df['CI lower'], color='blue',
                         alpha=0.2, label='95% confidence interval')
        ax.axhline(1, c='grey', ls='--')

        ax.set_xlabel('Distance from midpoint (μm)')
        ax.set_ylabel('Mean curvature ratio (old pole : new pole)')
        ax.set_xlim([-0.5, 0.5])
        plt.subplots_adjust(left=0.16)
        plt.show()

    def make_figure_5a(self):
        ax = sns.lmplot(self.all_df, x='Length (µm)', y='Curvedness', hue='Replicate', palette=self.replicate_palette, legend=False)

        nonan_df_curv = self.all_df.dropna(subset=['Length (µm)', 'Curvedness'])
        print(f'global r: {np.round(stats.pearsonr(x=nonan_df_curv["Length (µm)"], y=nonan_df_curv["Curvedness"])[0], 2)}')

        for i in range(self.all_df['Replicate'].nunique()):
            print(f'r for replicate {i + 1}: {np.round(stats.pearsonr(x=nonan_df_curv[nonan_df_curv["Replicate"] == i + 1].loc[:, "Length (µm)"], y=nonan_df_curv[nonan_df_curv["Replicate"] == i + 1].loc[:, "Curvedness"])[0], 2)}')

        plt.subplots_adjust(left=0.16)
        plt.legend(title='Replicate', loc='upper right')
        ax.set(ylabel='Mean curvedness')
        plt.show()

    def make_figure_5b(self):
        ax = sns.lmplot(self.all_df, x='Width (µm)', y='Curvedness', hue='Replicate', palette=self.replicate_palette, legend=False)

        nonan_df_width = self.all_df.dropna(subset=['Width (µm)', 'Curvedness'])
        print(f'global r: {np.round(stats.pearsonr(x=nonan_df_width["Width (µm)"], y=nonan_df_width["Curvedness"])[0], 2)}')

        for i in range(self.all_df['Replicate'].nunique()):
            print(f'r for replicate {i + 1}: {np.round(stats.pearsonr(x=nonan_df_width[nonan_df_width["Replicate"] == i + 1].loc[:, "Width (µm)"], y=nonan_df_width[nonan_df_width["Replicate"] == i + 1].loc[:, "Curvedness"])[0], 2)}')

        plt.subplots_adjust(left=0.16)
        plt.legend(title='Replicate', loc='upper left')
        ax.set(ylabel='Mean curvedness')
        plt.show()

    def make_figure_5c(self):
        ax = sns.lmplot(self.all_df, x='Length (µm)', y='Width (µm)', hue='Replicate', palette=self.replicate_palette,
                        legend=False)

        nonan_df = self.all_df.dropna(subset=['Length (µm)', 'Width (µm)'])
        print(
            f'global r: {np.round(stats.pearsonr(x=nonan_df["Length (µm)"], y=nonan_df["Width (µm)"])[0], 2)}')

        for i in range(self.all_df['Replicate'].nunique()):
            print(f'r for replicate {i + 1}: {np.round(stats.pearsonr(x=nonan_df[nonan_df["Replicate"] == i + 1].loc[:, "Length (µm)"], y=nonan_df[nonan_df["Replicate"] == i + 1].loc[:, "Width (µm)"])[0], 2)}')

        lin_reg = pg.linear_regression(self.all_df[['Length (µm)', 'Width (µm)']], self.all_df['Curvedness'], remove_na=True, relimp=True)
        pd.set_option('display.max_columns', None)
        print(lin_reg)
        print(stats.linregress(y=self.all_df['Length (µm)'], x=self.all_df['Width (µm)']))

        plt.subplots_adjust(left=0.16)
        plt.legend(title='Replicate', loc='upper right')
        plt.show()

    def make_figure_5d(self):
        fig, ax = plt.subplots()

        sns.boxplot(self.all_df, x='Replicate', y='Aspect ratio', palette=self.replicate_palette, ax=ax)
        sns.stripplot(self.all_df, x='Replicate', y='Aspect ratio', jitter=False, dodge=False,
                      palette=self.replicate_palette, legend=False, edgecolor='gray', linewidth=0.8, ax=ax)

        ax.set_ylabel('Aspect ratio (length : width)')
        ax.set_ylim(0)
        plt.show()


class Distances:
    """
    Class for aligning and calculating distances between meshes and creating visualisations.
    """

    def __init__(self, mesh_list):
        self.mesh_lst = mesh_list
        self.aligned_globally_mesh_lst = []
        self.dist_lst = []
        self.mean_dist_lst = []
        self.cl_points_lst = []
        self.coords_lst = []
        self.coords_df = None
        self.aligned_meshes_locally = False
        self.aligned_meshes_globally = False

        # Define distance scalar bar arguments
        self.scalar_args = dict(title='Distance', color='w', font_family='times', width=0.7, position_x=0.15,
                                n_labels=5, fmt='%.2e')

    def align_meshes_locally(self):
        """
        Align even index meshes to the subsequent odd index mesh locally.
        """
        print('Starting align_meshes_locally...')
        for i in range(int(len(mesh_lst) / 2)):
            aligned_mesh = self.mesh_lst[i * 2].rotate_x(180).align(self.mesh_lst[i * 2 + 1], max_mean_distance=1e-10,
                                                                    max_landmarks=500, max_iterations=1000)

            # Replace original mesh in mesh_lst with the aligned mesh
            self.mesh_lst[i * 2] = aligned_mesh
        self.aligned_meshes_locally = True
        print('Completed align_meshes_locally!')
        print()

    def align_meshes_globally(self, show_plot=False):
        """
        Align meshes globally by aligning even index meshes to the first mesh and transforming odd index meshes.
        """
        # If meshes are not aligned locally, do that first
        if self.aligned_meshes_locally is False:
            self.align_meshes_locally()
            self.align_meshes_globally(show_plot)
        else:
            print('Starting align_meshes_globally...')
            transform_matrix = 0
            for i in range(len(self.mesh_lst)):
                # Skip mesh3-mesh4 pair and mesh51
                if i == 2 or i == 3 or i == 50:
                    continue
                aligned_mesh = 0

                # Align even index meshes to the first mesh, then replace original mesh in mesh_lst with the aligned
                # mesh
                if i % 2 == 0:
                    aligned_mesh, transform_matrix = self.mesh_lst[i].align(self.mesh_lst[0], max_mean_distance=1e-12,
                                                                            max_landmarks=500, max_iterations=1000,
                                                                            return_matrix=True)
                    self.mesh_lst[i] = aligned_mesh

                # Transform odd index meshes using the transform matrix, then replace original mesh in mesh_lst with
                # the aligned mesh
                elif i % 2 == 1:
                    aligned_mesh = self.mesh_lst[i].transform(transform_matrix, transform_all_input_vectors=False)
                    self.mesh_lst[i] = aligned_mesh

                # Store globally aligned meshes in aligned_globally_mesh_lst
                self.aligned_globally_mesh_lst.append(aligned_mesh)
                self.aligned_meshes_globally = True
            print('Completed align_meshes_globally!')
            print()

            if show_plot is True:
                # Plot aligned meshes to double-check that all meshes are globally aligned
                pl = pv.Plotter()
                for i in range(len(self.mesh_lst)):
                    # Skip mesh3-mesh4 pair and mesh51
                    if i == 2 or i == 3 or i == 50:
                        continue
                    pl.add_mesh(mesh_lst[i], show_edges=True, edge_color='w', opacity=0.15, scalars=None)

                pl.camera.zoom(0.8)
                pl.window_size = [3024, 1964]
                pl.set_background('black')
                pl.show()

    def calculate_distances(self):
        """
        Calculate distances between even index meshes and the subsequent odd index mesh.
        """
        # If meshes are not aligned globally, do that first
        if self.aligned_meshes_globally is False:
            self.align_meshes_globally()
            self.calculate_distances()
        else:
            print('Starting calculate_distances...')
            for i in range(int(len(self.aligned_globally_mesh_lst) / 2)):
                # Find the closest cells and closest points between even index meshes and the subsequent odd index mesh
                cl_cells, cl_points = self.aligned_globally_mesh_lst[i * 2].find_closest_cell(
                    (self.aligned_globally_mesh_lst[i * 2 + 1]).points,
                    return_closest_point=True)

                # Calculate distances between the closest points
                dist = np.linalg.norm((self.aligned_globally_mesh_lst[i * 2 + 1]).points - cl_points, axis=1)

                # Assign distances to odd index mesh and store them in dist_lst
                self.aligned_globally_mesh_lst[i * 2 + 1]['Distances'] = dist
                self.dist_lst.append(dist)

                # Calculate mean distance and store it in mean_dist_lst
                mean_dist = np.mean(dist)
                self.mean_dist_lst.append(mean_dist)

                # Store closest points in cl_points_lst
                self.cl_points_lst.append(cl_points)
            print('Completed calculate_distances!')
            print()

    def make_gifs(self):
        """
        Make GIF animations visualizing the meshes and distances.
        """
        # If mean_dist_lst is empty, calculate distances and call create_gifs again
        if len(self.mean_dist_lst) == 0:
            self.calculate_distances()
            self.make_gifs()
        else:
            print('Starting create_gifs...')

            # Specify the angle step for animation
            angle_step = 2

            for i in range(int(len(self.mesh_lst) / 2)):
                print(f'Making GIF {i + 1} out of {int(len(self.mesh_lst) / 2)}')
                pl = pv.Plotter(off_screen=True)

                # Add even index meshes and the subsequent odd index mesh to the plot
                pl.add_mesh(self.mesh_lst[i * 2], show_edges=True, edge_color='w', opacity=0.15)
                pl.add_mesh(self.mesh_lst[i * 2 + 1], scalars='Distances', scalar_bar_args=self.scalar_args)

                # Open a GIF file to save the frames
                pl.open_gif(f'dist_mesh{str(i * 2 + 1)}_mesh{str(i * 2 + 2)}.gif', fps=12, subrectangles=True)

                # Create frames by rotating the camera position around the meshes
                for theta in np.linspace(0, 360, int(360 / angle_step)):
                    # Convert the angle from degrees to radians
                    theta = theta * m.pi / 180

                    # Calculate the x and y coordinates on the unit circle
                    x = np.cos(theta)
                    y = np.sin(theta)

                    # Define the new camera position with the x, y, and z values
                    cpos_new = (x, y, m.pi / 6)

                    # Update camera position and settings for each frame
                    pl.camera_position = cpos_new
                    pl.camera.zoom(0.8)
                    pl.window_size = [3024, 1964]
                    pl.set_background('black')
                    pl.write_frame()
                print('Completed create_gifs!')
                pl.close()

    def make_3d_heatmap(self):
        """
        Make a 3D heatmap (/bar plot/histogram) of theta vs phi vs Distance, where the magnitude of Distance is
        represented with a colourmap.
        """
        # If dist_lst is empty, calculate distances and call create_3d_heatmap again
        if len(self.dist_lst) == 0:
            self.calculate_distances()
            self.make_3d_heatmap()
        else:
            # Store coordinates of odd index globally aligned meshes in coords_lst
            for i in range(len(self.aligned_globally_mesh_lst)):
                if i % 2 == 1:
                    self.coords_lst.append(self.aligned_globally_mesh_lst[i].points)

            # Convert coords_lst into a dataframe
            self.coords_lst = [[(x, y, z) for x, y, z in coords_sub_lst] for coords_sub_lst in self.coords_lst]
            coords_flat_lst = [i for coords_sub_lst in self.coords_lst for i in coords_sub_lst]
            self.coords_df = pd.DataFrame(coords_flat_lst, columns=['x', 'y', 'z'])

            # Add columns for spherical coordinates r (radial distance), theta (azimuthal angle), and phi
            # (polar angle)
            self.coords_df['r'] = np.sqrt(
                self.coords_df['x'] ** 2 + self.coords_df['y'] ** 2 + self.coords_df['z'] ** 2)
            self.coords_df['theta'] = np.arctan2(self.coords_df['y'], self.coords_df['x'])
            self.coords_df['phi'] = np.arccos(self.coords_df['z'] / self.coords_df['r'])

            # Convert theta and phi from radians to degrees
            self.coords_df['theta'] = np.degrees(self.coords_df['theta'])
            self.coords_df['phi'] = np.degrees(self.coords_df['phi'])

            # Normalise theta and phi
            self.coords_df['theta'] = self.coords_df['theta'] - self.coords_df['theta'].median()
            self.coords_df['phi'] = self.coords_df['phi'] - self.coords_df['phi'].median()

            # Add a column for distance
            dist_col = [item for sublist in self.dist_lst for item in sublist]
            self.coords_df['Distance'] = dist_col

            # Remove rows where theta and phi are less than ___ and greater than ___
            # self.coords_df = self.coords_df.drop(self.coords_df.index[self.coords_df['theta'] < -6])
            # self.coords_df = self.coords_df.drop(self.coords_df.index[self.coords_df['theta'] > -1])
            self.coords_df = self.coords_df.drop(self.coords_df.index[self.coords_df['phi'] < -1.6])
            self.coords_df = self.coords_df.drop(self.coords_df.index[self.coords_df['phi'] > 1.5])

            # From here onwards, theta = x, phi = y, and Distance = z
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            # Set the number of bins along the x and y axes, such that total number of bins is 50^2. The z values for 
            # each bin are stored in binned_z. The bin edges along the x and y axes are stored in x_edges and y_edges
            num_of_bins = 50
            binned_z, x_edges, y_edges = np.histogram2d(self.coords_df['phi'], self.coords_df['theta'],
                                                        bins=(num_of_bins, num_of_bins))
            binned_z = binned_z.flatten()

            # Calculate the position of each bin's centre point
            x_pos, y_pos = np.meshgrid(x_edges[:-1] + x_edges[1:], y_edges[:-1] + y_edges[1:])
            x_pos = x_pos.flatten() / 2.
            y_pos = y_pos.flatten() / 2.
            z_pos = np.zeros_like(x_pos)

            # Calculate the difference between adjacent bin edges
            dx = x_edges[1] - x_edges[0]
            dy = y_edges[1] - y_edges[0]

            # Sort x and y values into discrete bins
            x_bins = pd.cut(self.coords_df['phi'], num_of_bins)
            y_bins = pd.cut(self.coords_df['theta'], num_of_bins)

            # Group coords_df by these bins and calculate mean for every bin
            group = self.coords_df.groupby([x_bins, y_bins]).mean().fillna(0)

            # Replace Distance values with 0 if the number of data points for 'Distance' within a bin is less than 40% 
            # of the mean number of data points per bin
            filter_percent = 0.4
            filter_ref = np.mean(binned_z)
            group.loc[binned_z <= filter_percent * filter_ref, 'Distance'] = 0
            z_dist = group['Distance']

            # Set colourmap to viridis
            cmap = cm.get_cmap('viridis')

            # Normalise Distance to colourmap by scaling their minimum and maximum height to 0 and 1, respectively
            min_height = np.min(z_dist)
            max_height = np.quantile(z_dist, 0.85)
            rgba = [cmap((i - min_height) / max_height) for i in z_dist]

            ax.bar3d(x_pos, y_pos, z_pos, dx, dy, z_dist, color=rgba, zsort='average')
            plt.xlabel('θ')
            plt.ylabel('φ')
            ax.set_zlabel('Distance')

            plt.show()


if __name__ == '__main__':
    # Import mesh_lst and all_df
    prepro = Preprocess()
    prepro.import_meshes()
    mesh_lst = prepro.mesh_lst
    prepro.make_df()
    all_df = prepro.all_df

    prelim_plots = PreliminaryPlots(mesh_lst)
    prelim_plots.plot_single(start=0, stop=1)  # Make figure 2 from the report
    prelim_plots.make_gifs()
    prelim_plots.plot_dual(left_mesh=0)

    figs = Figures(all_df)  # Make figures 3-5 from the report
    figs.make_figure_3a()
    figs.make_figure_3b()
    figs.make_figure_4()
    figs.make_figure_5a()
    figs.make_figure_5b()
    figs.make_figure_5c()
    figs.make_figure_5d()

    gifs = Distances(mesh_lst)
    gifs.make_gifs()  # Make figure 6 from the report
    gifs.make_3d_heatmap()  # Make figure 7 from the report
