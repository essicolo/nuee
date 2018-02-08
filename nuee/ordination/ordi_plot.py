def ordiplot(self, axes=[0, 1], axis_labels=None,
         title='', cmap=None, arrow_scale=1):
    # checks
    if not len(axes) == 2:
        raise ValueError("axes must have two integers.")

    import matplotlib.pyplot as plt
    sample_scores = self.sample_scores.iloc[:, axes]
    biplot_scores = self.biplot_scores.iloc[:, axes]

    # plot
    plt.plot(sample_scores.iloc[:,0], sample_scores.iloc[:,1], '.', alpha=0)

    ## draw samples
    for i in range(sample_scores.shape[0]):
        plt.text(x=sample_scores.iloc[i,0],
                 y=sample_scores.iloc[i,1],
                 s=sample_scores.index.values[i],
                  horizontalalignment='center',
                  verticalalignment='center',
                 color='black')

    ## draw variables
    ### plot transparent point to rescale the plot
    plt.plot(biplot_scores.iloc[:,0], biplot_scores.iloc[:,1], '.', alpha=0)
    ### lines (arrows)
    for i in range(biplot_scores.shape[0]):
        plt.arrow(0, 0,
              biplot_scores.iloc[i,0]*arrow_scale,
              biplot_scores.iloc[i,1]*arrow_scale,
              color = 'blue', head_width=0)
    ### labels
    margin_score_labels = 0.1
    for i in range(biplot_scores.shape[0]):
        plt.text(x=biplot_scores.iloc[i,0]*(arrow_scale+margin_score_labels),
             y=biplot_scores.iloc[i,1]*(arrow_scale+margin_score_labels),
             s = biplot_scores.index.values[i], color='blue',
             horizontalalignment='center',
             verticalalignment='center')

    ## draw species if rda or cca
    if self.short_method_name == 'RDA':
        response_scores = self.response_scores.iloc[:, axes]
        ### plot transparent point to rescale the plot
        plt.plot(response_scores.iloc[:,0], response_scores.iloc[:,1], '.', alpha=0)
        ### labels
        for i in range(response_scores.shape[0]):
            plt.text(x=response_scores.iloc[i,0],
                     y=response_scores.iloc[i,1],
                     s=response_scores.index.values[i],
                     color='red',
                     horizontalalignment='center',
                     verticalalignment='center')

    ## draw the cross
    plt.axvline(0, ls='solid', c='gray')
    plt.axhline(0, ls='solid', c='gray')


def screeplot(self):
    import matplotlib.pyplot as plt
    (self.proportion_explained.iloc[:self.biplot_scores.shape[0]] * 100).plot(kind='bar', color='k')
    plt.ylabel('Proportion explained (%)')


# spider plot
# ellipses (lda)
