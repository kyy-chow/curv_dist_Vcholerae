from data_preprocessing import Preprocess
import numpy as np
from scipy import stats


class Stats:
    def __init__(self, all_df):
        self.all_df = all_df
        self.old = self.all_df[self.all_df['Old or new pole'] == 'Old pole']
        self.new = self.all_df[self.all_df['Old or new pole'] == 'New pole']

    def stats_curvedness(self):
        # Independent t-test for curvedness of all old poles vs. all new poles
        stat, p = stats.ttest_ind(self.old.loc[:, 'Curvedness'], self.new.loc[:, 'Curvedness'],
                                  nan_policy='omit', alternative='greater')

        # Print "Curvedness of sample 1 (M = mean, SD = sd) versus sample 2 (M = mean, SD = sd), t('df') = t, p = p
        # value"
        print(f'Curvedness of old poles ' +
              f'(M = {np.format_float_scientific(np.nanmean(self.old.loc[:, "Curvedness"]), precision=2)}, ' +
              f'SD = {np.format_float_scientific(np.std(self.old.loc[:, "Curvedness"]), precision=2)}) versus ' +

              f'new poles ' +
              f'(M = {np.format_float_scientific(np.nanmean(self.new.loc[:, "Curvedness"]), precision=2)}, ' +
              f'SD = {np.format_float_scientific(np.std(self.new.loc[:, "Curvedness"]), precision=2)}), ' +

              f't(30) = {np.round(stat, 1)}, p = {np.round(p, 3)}')

        # Related t-test for curvedness of old poles vs. new poles in replicates 1, 2, and 3
        for i in range(self.all_df['Replicate'].nunique()):
            stat, p = stats.ttest_rel(self.old[self.old['Replicate'] == i + 1].loc[:, 'Curvedness'],
                                      self.new[self.new['Replicate'] == i + 1].loc[:, 'Curvedness'],
                                      nan_policy='omit', alternative='greater')

            # Print "Curvedness of sample 1 (M = mean, SD = sd) versus sample 2 (M = mean, SD = sd) in replicate x t(
            # 'df') = t, p = p value"
            print(f'Curvedness of old poles ' +
                  f'(M = {np.format_float_scientific(np.nanmean(self.old[self.old["Replicate"] == i + 1].loc[:, "Curvedness"]), precision=2)}, ' +
                  f'SD = {np.format_float_scientific(np.std(self.old[self.old["Replicate"] == i + 1].loc[:, "Curvedness"]), precision=2)}) versus ' +

                  f'new poles ' +
                  f'(M = {np.format_float_scientific(np.nanmean(self.new[self.new["Replicate"] == i + 1].loc[:, "Curvedness"]), precision=2)}, ' +
                  f'SD = {np.format_float_scientific(np.std(self.new[self.new["Replicate"] == i + 1].loc[:, "Curvedness"]), precision=2)}) ' +

                  f'in replicate {i + 1}, ' +

                  f't({self.old[self.old["Replicate"] == i + 1].loc[:, "Curvedness"].count() - 1}) = {np.round(stat, 1)}, p = {np.round(p, 3)}')

        print()

    def stats_curvednesss_ratio(self):
        below_10_old_df = self.old[self.old["Ratio"] < 10]

        # One sample t-test for curvedness ratio of all poles vs. 1
        stat, p = stats.ttest_1samp(below_10_old_df.loc[:, 'Ratio'], 1, nan_policy='omit', alternative='greater')

        # Print "Curvedness ratio (M = mean, SD = sd) versus 1, t('df') = t, p = p value"
        print(f'Curvedness ratio ' +
              f'(M = {np.round(np.nanmean(below_10_old_df.loc[:, "Ratio"]), 2)}, ' +
              f'SD = {np.round(np.std(below_10_old_df.loc[:, "Ratio"]), 2)}) versus 1, ' +

              f't(30) = {np.round(stat, 1)}, p = {np.round(p, 3)}')

        # One sample t-test for curvedness ratio of poles in replicate 1, 2, and 3 vs. 1
        for i in range(self.all_df['Replicate'].nunique()):
            stat, p = stats.ttest_1samp(below_10_old_df[below_10_old_df['Replicate'] == i + 1].loc[:, 'Ratio'], 1, nan_policy='omit',
                                        alternative='greater')

            # Print "Curvedness ratio in replicate x (M = mean, SD = sd) versus 1, t('df') = t, p = p value"
            print(f'Curvedness ratio in replicate {i + 1} ' +
                  f'(M = {np.round(np.nanmean(below_10_old_df[below_10_old_df["Replicate"] == i + 1].loc[:, "Ratio"]), 2)}, ' +
                  f'SD = {np.round(np.std(below_10_old_df[below_10_old_df["Replicate"] == i + 1].loc[:, "Ratio"]), 2)}) versus 1, ' +

                  f't({below_10_old_df[below_10_old_df["Replicate"] == i + 1].loc[:, "Curvedness"].count() - 1}) = {np.round(stat, 1)}, p = {np.round(p, 3)}')

        print()

    def stats_aspect_ratio(self):
        # One sample t-test for curvedness ratio of all poles vs. 1
        stat, p = stats.ttest_1samp(all_df.loc[:, 'Aspect ratio'], 2.5, nan_policy='omit', alternative='greater')

        # Print "Curvedness ratio (M = mean, SD = sd) versus 1, t('df') = t, p = p value"
        print(f'Aspect ratio ' +
              f'(M = {np.round(np.nanmean(all_df.loc[:, "Aspect ratio"]), 2)}, ' +
              f'SD = {np.round(np.std(all_df.loc[:, "Aspect ratio"]), 2)}) versus 2.5, ' +

              f't(30) = {np.round(stat, 1)}, p = {np.round(p, 3)}')

        # One sample t-test for curvedness ratio of poles in replicate 1, 2, and 3 vs. 1
        for i in range(self.all_df['Replicate'].nunique()):
            stat, p = stats.ttest_1samp(all_df[all_df['Replicate'] == i + 1].loc[:, 'Aspect ratio'], 2.5,
                                        nan_policy='omit', alternative='greater')

            # Print "Curvedness ratio in replicate x (M = mean, SD = sd) versus 1, t('df') = t, p = p value"
            print(f'Aspect ratio in replicate {i + 1} ' +
                  f'(M = {np.round(np.nanmean(all_df[all_df["Replicate"] == i + 1].loc[:, "Aspect ratio"]), 2)}, ' +
                  f'SD = {np.round(np.std(all_df[all_df["Replicate"] == i + 1].loc[:, "Aspect ratio"]), 2)}) versus 2.5, ' +

                  f't({all_df[all_df["Replicate"] == i + 1].loc[:, "Curvedness"].count() - 1}) = {np.round(stat, 1)}, p = {np.round(p, 3)}')

        print()


if __name__ == '__main__':
    # Import mesh_lst and all_df
    prepro = Preprocess()
    prepro.import_meshes()
    mesh_lst = prepro.mesh_lst
    prepro.make_df()
    all_df = prepro.all_df

    do_ttest = Stats(all_df)
    do_ttest.stats_curvedness()  # Print t-test results for curvedness of old poles versus new poles
    do_ttest.stats_curvednesss_ratio()  # Print t-test results for curvedness ratio versus 1
    do_ttest.stats_aspect_ratio()  # Print t-test results for aspect ratio versus 1
