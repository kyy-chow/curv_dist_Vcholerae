import numpy as np
import pandas as pd
import pyvista as pv


class Preprocess:
    """
    Class for importing meshes from VTP files and creating a dataframe that contains all information for all poles.
    """

    def __init__(self):
        self.mesh_lst = None
        self.all_df = None

    def import_meshes(self):
        """
        Import meshes from VTP files and store them in a list.
        """
        # VTP file name abbreviations and meanings:
        #   - d: day
        #   - t: tomogram number
        #   - c: cell
        #   - f: flagellated
        #   - nf: non-flagellated
        #   - c: chemoreceptor
        #   - nc: no chemoreceptor
        #   - a and b: arbitrary labels for poles when old and new poles cannot be differentiated
        #   - vesicle_placeholder: placeholder mesh as the visualisation and statistical analysis scripts rely on a
        #   consecutive mesh list
        mesh1 = pv.read('d0_t3_c1_f.vtp')
        mesh2 = pv.read('d0_t3_c1_nf.vtp')
        mesh3 = pv.read('d0_t4_c1_f.vtp')
        mesh4 = pv.read('vesicle_placeholder.vtp')
        mesh5 = pv.read('d0_t5_c1_f.vtp')
        mesh6 = pv.read('d0_t5_c1_nf.vtp')
        mesh7 = pv.read('d0_t6_c1_c.vtp')
        mesh8 = pv.read('d0_t6_c1_nc.vtp')
        mesh9 = pv.read('d0_t6_c2_f.vtp')
        mesh10 = pv.read('d0_t6_c2_nf.vtp')
        mesh11 = pv.read('d0_t7_c1_a.vtp')
        mesh12 = pv.read('d0_t7_c1_b.vtp')
        mesh13 = pv.read('d0_t7_c2_a.vtp')
        mesh14 = pv.read('d0_t7_c2_b.vtp')
        mesh15 = pv.read('d0_t8_c1_f.vtp')
        mesh16 = pv.read('d0_t8_c1_nf.vtp')
        mesh17 = pv.read('d0_t8_c2_f.vtp')
        mesh18 = pv.read('d0_t8_c2_nf.vtp')
        mesh19 = pv.read('d0_t8_c3_f.vtp')
        mesh20 = pv.read('d0_t8_c3_nf.vtp')
        mesh21 = pv.read('d0_t9_c1_f.vtp')
        mesh22 = pv.read('d0_t9_c1_nf.vtp')
        mesh23 = pv.read('d0_t9_c2_f.vtp')
        mesh24 = pv.read('d0_t9_c2_nf.vtp')
        mesh25 = pv.read('d0_t10_c1_f.vtp')
        mesh26 = pv.read('d0_t10_c1_nf.vtp')
        mesh27 = pv.read('d0_t11_c1_a.vtp')
        mesh28 = pv.read('d0_t11_c1_b.vtp')
        mesh29 = pv.read('d2_t18_c1_f.vtp')
        mesh30 = pv.read('d2_t18_c1_nf.vtp')
        mesh31 = pv.read('d2_t21_c1_f.vtp')
        mesh32 = pv.read('d2_t21_c1_nf.vtp')
        mesh33 = pv.read('d2_t27_c1_a.vtp')
        mesh34 = pv.read('d2_t27_c1_b.vtp')
        mesh35 = pv.read('d2_t28_c1_a.vtp')
        mesh36 = pv.read('d2_t28_c1_b.vtp')
        mesh37 = pv.read('d3_t1_c1_f.vtp')
        mesh38 = pv.read('d3_t1_c1_nf.vtp')
        mesh39 = pv.read('d3_t8_c1_a.vtp')
        mesh40 = pv.read('d3_t8_c1_b.vtp')
        mesh41 = pv.read('d3_t21_c1_a.vtp')
        mesh42 = pv.read('d3_t21_c1_b.vtp')
        mesh43 = pv.read('d3_t22_c1_a.vtp')
        mesh44 = pv.read('d3_t22_c1_b.vtp')
        mesh45 = pv.read('d3_t23_c1_f.vtp')
        mesh46 = pv.read('d3_t23_c1_nf.vtp')
        mesh47 = pv.read('d3_tcc1_c1_c.vtp')
        mesh48 = pv.read('d3_tcc1_c1_nc.vtp')
        mesh49 = pv.read('d3_tcc1_c2_f.vtp')
        mesh50 = pv.read('d3_tcc1_c2_nf.vtp')
        mesh51 = pv.read('vesicle_placeholder.vtp')

        # Create a list to store all meshes
        self.mesh_lst = [mesh1, mesh2, mesh3, mesh4, mesh5, mesh6, mesh7, mesh8, mesh9, mesh10, mesh11, mesh12, mesh13,
                         mesh14, mesh15, mesh16, mesh17, mesh18, mesh19, mesh20, mesh21, mesh22, mesh23, mesh24, mesh25,
                         mesh26, mesh27, mesh28, mesh29, mesh30, mesh31, mesh32, mesh33, mesh34, mesh35, mesh36, mesh37,
                         mesh38, mesh39, mesh40, mesh41, mesh42, mesh43, mesh44, mesh45, mesh46, mesh47, mesh48, mesh49,
                         mesh50, mesh51]

    def make_df(self):
        """
        Create a dataframe in the "tidy" data format that contains all information for all poles. Every row is an
        observation (pole) and every column is a variable.

        Returns:
            DataFrame: A dataframe in the "tidy" data format that contains the following columns:
                - Mesh: Index for each pole that starts at 1 and corresponds to mesh_lst.
                - Day: Day of data collection.
                - Replicate:
                - Grid type:
                - Tomogram no.:
                - Cell no.: Cell from which each pole was measured, as some tomograms contain multiple cells.
                - Length (nm): Length values for each cell.
                - Width (nm): Width values for each cell, measured at the midpoint.
                - Pole: Pole "organ" type (flagellated, chemoreceptor), or lack thereof (nonflagellated,
                nonchemoreceptor, A, B).
                - Old or new pole: Flagellated and chemoreceptor poles are assigned to 'Old pole', nonflagellated and
                nonchemoreceptor are assigned to 'New pole', and A and B are np.nan.
                - Curvedness: Mean curvedness values for each pole.
                - Ratio: For even index observations, this is the ratio between the curvedness of the current pole and
                the curvedness of the subsequent pole. For odd index observations, this is the ratio between the
                curvedness of the current pole and the curvedness of the preceding pole.
                - Aspect ratio: Ratio of length to width for each cell.
        """
        # Calculate mean curvedness for all poles and store them in the mean_dict dictionary
        mean_dict = {}
        for i in self.mesh_lst:
            i.set_active_scalars('curvedness_VV')
            mean = np.mean(i.active_scalars)
            mean_dict[str(self.mesh_lst.index(i))] = mean
        mean_df = pd.Series(mean_dict, name='mean')

        # Create a dataframe in the "tidy" data format for each replicate before concatenating into a larger dataframe
        replicate1 = pd.DataFrame({
            'Mesh': [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                     27, 28],
            'Day': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Replicate': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'Grid type': ['Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey',
                          'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey',
                          'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey'],
            'Tomogram no.': [3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11],
            'Cell no.': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1],
            'Length (nm)': [(1797 + 1330 + 804), (1797 + 1330 + 804), 2632, 2632, (1168 + 998), (1168 + 998),
                            (1341 + 1375 + 581 + 839), (1341 + 1375 + 581 + 839), (717 + 1547), (717 + 1547),
                            (1813 + 1860), (1813 + 1860), (1423 + 695 + 3000), (1423 + 695 + 3000), 2141, 2141, 2217,
                            2217, (1421 + 978), (1421 + 978), (908 + 763 + 1143 + 1000), (908 + 763 + 1143 + 1000),
                            2689, 2689, (931 + 1921), (931 + 1921), (895 + 921 + 967), (895 + 921 + 967)],
            'Width (nm)': [719, 719, 873, 873, 927, 927, 709, 709, 856, 856, 753, 753, 695, 695, 958, 958, 836, 836,
                           843, 843, 739, 739, 933, 933, 811, 811, 872, 872],
            'Pole': ['Flagellated', 'Nonflagellated', 'Flagellated', 'Nonflagellated', 'Flagellated', 'Nonflagellated',
                     'Chemoreceptor', 'Nonchemoreceptor', 'Flagellated', 'Nonflagellated', 'A', 'B', 'A', 'B',
                     'Flagellated', 'Nonflagellated', 'Flagellated', 'Nonflagellated', 'Flagellated', 'Nonflagellated',
                     'Flagellated', 'Nonflagellated', 'Flagellated', 'Nonflagellated', 'Flagellated', 'Nonflagellated',
                     'A', 'B'],
            'Old or new pole': ['Old pole', 'New pole', 'Old pole', 'New pole', 'Old pole', 'New pole', 'Old pole',
                                'New pole', 'Old pole', 'New pole', np.nan, np.nan, np.nan, np.nan, 'Old pole',
                                'New pole', 'Old pole', 'New pole', 'Old pole', 'New pole', 'Old pole', 'New pole',
                                'Old pole', 'New pole', 'Old pole', 'New pole', np.nan, np.nan],
            'A or B': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 'A', 'B', 'A',
                       'B', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                       np.nan, 'A', 'B'],
            'Curvedness': [mean_df.iloc[0], mean_df.iloc[1], np.nan, np.nan, mean_df.iloc[4], mean_df.iloc[5],
                           mean_df.iloc[6], mean_df.iloc[7], mean_df.iloc[8], mean_df.iloc[9], mean_df.iloc[11],
                           mean_df.iloc[10], mean_df.iloc[13], mean_df.iloc[12], mean_df.iloc[14], mean_df.iloc[15],
                           mean_df.iloc[16], mean_df.iloc[17], mean_df.iloc[18], mean_df.iloc[19], mean_df.iloc[20],
                           mean_df.iloc[21], mean_df.iloc[22], mean_df.iloc[23], mean_df.iloc[24], mean_df.iloc[25],
                           mean_df.iloc[26], mean_df.iloc[27]],
            'Ratio': [mean_df.iloc[0] / mean_df.iloc[1], mean_df.iloc[1] / mean_df.iloc[0],
                      mean_df.iloc[2] / mean_df.iloc[3], mean_df.iloc[3] / mean_df.iloc[2],
                      mean_df.iloc[4] / mean_df.iloc[5], mean_df.iloc[5] / mean_df.iloc[4],
                      mean_df.iloc[6] / mean_df.iloc[7], mean_df.iloc[7] / mean_df.iloc[6],
                      mean_df.iloc[8] / mean_df.iloc[9], mean_df.iloc[9] / mean_df.iloc[8],
                      mean_df.iloc[10] / mean_df.iloc[11], mean_df.iloc[11] / mean_df.iloc[10],
                      mean_df.iloc[12] / mean_df.iloc[13], mean_df.iloc[13] / mean_df.iloc[12],
                      mean_df.iloc[14] / mean_df.iloc[15], mean_df.iloc[15] / mean_df.iloc[14],
                      mean_df.iloc[16] / mean_df.iloc[17], mean_df.iloc[17] / mean_df.iloc[16],
                      mean_df.iloc[18] / mean_df.iloc[19], mean_df.iloc[19] / mean_df.iloc[18],
                      mean_df.iloc[20] / mean_df.iloc[21], mean_df.iloc[21] / mean_df.iloc[20],
                      mean_df.iloc[22] / mean_df.iloc[23], mean_df.iloc[23] / mean_df.iloc[22],
                      mean_df.iloc[24] / mean_df.iloc[25], mean_df.iloc[25] / mean_df.iloc[24],
                      mean_df.iloc[26] / mean_df.iloc[27], mean_df.iloc[27] / mean_df.iloc[26]]
        })
        replicate2 = pd.DataFrame({
            'Mesh': [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46],
            'Day': [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            'Replicate': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            'Grid type': ['Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey',
                          'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey', 'Holey'],
            'Tomogram no.': [18, 18, 21, 21, 27, 27, 28, 28, 1, 1, 8, 8, 21, 21, 22, 22, 23, 23],
            'Cell no.': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'Length (nm)': [2359, 2359, (1510 + 1519), (1510 + 1519), 2507, 2507, 2501, 2501, 2607, 2607, (1308 + 1362),
                            (1308 + 1362), 2768, 2768, 2230, 2230, (1561 + 1833), (1561 + 1833)],
            'Width (nm)': [1194, 1194, 1124, 1124, 1257, 1257, 1240, 1240, 1142, 1142, 1168, 1168, 1172, 1172, 1204,
                           1204, 1135, 1135],
            'Pole': ['Flagellated', 'Nonflagellated', 'Flagellated', 'Nonflagellated', 'A', 'B', 'A', 'B',
                     'Flagellated', 'Nonflagellated', 'A', 'B', 'A', 'B', 'A', 'B', 'Flagellated', 'Nonflagellated'],
            'Old or new pole': ['Old pole', 'New pole', 'Old pole', 'New pole', np.nan, np.nan, np.nan, np.nan,
                                'Old pole', 'New pole', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 'Old pole',
                                'New pole'],
            'A or B': [np.nan, np.nan, np.nan, np.nan, 'A', 'B', 'A', 'B', np.nan, np.nan, 'A', 'B', 'A', 'B', 'A', 'B',
                       np.nan, np.nan],
            'Curvedness': [mean_df.iloc[28], mean_df.iloc[29], mean_df.iloc[30], mean_df.iloc[31], mean_df.iloc[32],
                           mean_df.iloc[33], mean_df.iloc[34], mean_df.iloc[35], mean_df.iloc[36], mean_df.iloc[37],
                           mean_df.iloc[38], mean_df.iloc[39], mean_df.iloc[40], mean_df.iloc[41], mean_df.iloc[43],
                           mean_df.iloc[42], mean_df.iloc[44], mean_df.iloc[45]],
            'Ratio': [mean_df.iloc[28] / mean_df.iloc[29], mean_df.iloc[29] / mean_df.iloc[28],
                      mean_df.iloc[30] / mean_df.iloc[31], mean_df.iloc[31] / mean_df.iloc[30],
                      mean_df.iloc[32] / mean_df.iloc[33], mean_df.iloc[33] / mean_df.iloc[32],
                      mean_df.iloc[34] / mean_df.iloc[35], mean_df.iloc[35] / mean_df.iloc[34],
                      mean_df.iloc[36] / mean_df.iloc[37], mean_df.iloc[37] / mean_df.iloc[36],
                      mean_df.iloc[38] / mean_df.iloc[39], mean_df.iloc[39] / mean_df.iloc[38],
                      mean_df.iloc[40] / mean_df.iloc[41], mean_df.iloc[41] / mean_df.iloc[40],
                      mean_df.iloc[42] / mean_df.iloc[43], mean_df.iloc[43] / mean_df.iloc[42],
                      mean_df.iloc[44] / mean_df.iloc[45], mean_df.iloc[45] / mean_df.iloc[44]]
        })
        replicate3 = pd.DataFrame({
            'Mesh': [47, 48, 49, 50],
            'Day': [3, 3, 3, 3],
            'Replicate': [3, 3, 3, 3],
            'Grid type': ['Continuous', 'Continuous', 'Continuous', 'Continuous'],
            'Tomogram no.': [1, 1, 1, 1],
            'Cell no.': [1, 1, 2, 2],
            'Length (nm)': [(1809 + 1231), (1809 + 1231), (1717 + 807 + 1076), (1717 + 807 + 1076)],
            'Width (nm)': [812, 812, 810, 810],
            'Pole': ['Chemoreceptor', 'Nonchemoreceptor', 'Flagellated', 'Nonflagellated'],
            'Old or new pole': ['Old pole', 'New pole', 'Old pole', 'New pole'],
            'A or B': [np.nan, np.nan, np.nan, np.nan],
            'Curvedness': [mean_df.iloc[46], mean_df.iloc[47], mean_df.iloc[48], mean_df.iloc[49]],
            'Ratio': [mean_df.iloc[46] / mean_df.iloc[47], mean_df.iloc[47] / mean_df.iloc[46],
                      mean_df.iloc[48] / mean_df.iloc[49], mean_df.iloc[49] / mean_df.iloc[48]]
        })

        # Concatenate replicate 1, 2, and 3 dataframes into one big dataframe
        self.all_df = pd.concat([replicate1, replicate2, replicate3])
        self.all_df = self.all_df.reset_index(drop=True)

        # Add a column for length and width in µm to the big dataframe
        self.all_df['Length (µm)'] = self.all_df['Length (nm)'] / 1000
        self.all_df['Width (µm)'] = self.all_df['Width (nm)'] / 1000

        # Add a column for ratio of length to width to the big dataframe
        self.all_df['Aspect ratio'] = self.all_df['Length (µm)'] / self.all_df['Width (µm)']


if __name__ == '__main__':
    # Import mesh_lst and all_df
    prepro = Preprocess()
    prepro.import_meshes()
    mesh_lst = prepro.mesh_lst
    prepro.make_df()
    all_df = prepro.all_df

    pd.set_option('display.max_columns', None)
    print(all_df)