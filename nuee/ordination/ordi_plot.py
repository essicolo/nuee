import pandas as pd
from plotnine import *

def ordiplot(self, axes=[0, 1],
         arrow_scale=1,
         sample_scatter='labels',
         group=None,
         level=0.95,
         deviation_ellipses = True,
         error_ellipses = True):

    # checks
    if not len(axes) == 2:
        raise ValueError("axes must have two integers.")

    ordi_columns = self.sample_scores.columns[axes]

    p = ggplot(self.sample_scores.reset_index(),
               aes(x=ordi_columns[0],
                   y=ordi_columns[1]))

    # add ellipses
    from nuee.stats._utils import ellipse
    if group is not None and deviation_ellipses:
        dev_ellipses = []
        for i in pd.unique(group):
            _ellipses_i = ellipse(self.sample_scores.loc[group == i, ordi_columns], level=level, method='deviation')
            _ellipses_i = pd.DataFrame(_ellipses_i,
                                         columns=ordi_columns)
            _ellipses_i['group'] = i
            dev_ellipses.append(_ellipses_i)
        dev_ellipses = pd.concat(dev_ellipses, axis=0)
        p = p + geom_polygon(dev_ellipses, mapping=aes(fill='group'), alpha=0.3)
    if group is not None and error_ellipses:
        err_ellipses = []
        for i in pd.unique(group):
            _ellipses_i = ellipse(self.sample_scores.loc[group == i, ordi_columns], level=level, method='error')
            _ellipses_i = pd.DataFrame(_ellipses_i,
                                         columns=ordi_columns)
            _ellipses_i['group'] = i
            err_ellipses.append(_ellipses_i)
        err_ellipses = pd.concat(err_ellipses, axis=0)
        p = p + geom_polygon(err_ellipses, mapping=aes(fill='group'), colour='white')

    # add sample scores
    if sample_scatter == 'points' and group is None:
        p = p + geom_point()
    elif sample_scatter == 'points' and group is not None:
        p = p + geom_point(aes(colour='group'))
    if sample_scatter == 'labels' and group is None:
        p = p + geom_text(aes(label='ID'))
    elif sample_scatter == 'labels' and group is not None:
        p = p + geom_text(aes(label='ID', colour='group'))

    ## draw species if rda or cca
    if self.ordiobject_type == 'RDA':
        p = p + geom_text(data = self.response_scores.iloc[:, axes].reset_index(),
                          mapping = aes(label='ID'), colour='slategrey')

    # add biplot scores
    p = p +\
        geom_segment(self.biplot_scores * arrow_scale, x=0, y=0,
                     mapping=aes(xend=ordi_columns[0],
                                 yend=ordi_columns[1])) +\
        geom_label((self.biplot_scores * arrow_scale).reset_index(),
                     mapping=aes(x=ordi_columns[0],
                                 y=ordi_columns[1],
                                 label=self.biplot_scores.index.name))

    return p


def screeplot(self):
    df = (self.proportion_explained.iloc[:self.biplot_scores.shape[0]] * 100).reset_index()
    df.columns = ['ID', 'Proportion explainded (%)']
    df["ID"] = pd.Categorical(df["ID"], categories=df["ID"], ordered=True)
    p = ggplot(df, aes(x='ID', y='Proportion explainded (%)')) +\
        geom_col() +\
        xlab('')
    return p


# spider plot
