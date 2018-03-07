import numpy as np
import pandas as pd

from scipy import interpolate
import sys
import matplotlib.pylab as pl
import matplotlib.pyplot as plt

## REMOVE CLASSES



class CloseComp(object):
    """
    Normalize rows to a constant sum of 1.
    This class could be defined as a function. (Should it?)
    """

    def __init__(self, x, total=1):
        self.total=total
        self.x = x
        self.parts = np.array(self.x = x).astype(float) # force parts to be an array of real values
        self.parts = np.atleast_2d(parts)
        condition = self.parts >= 0
        if not condition.all():
            print("Warning. There are negative, infitite of missing values.")

    def closure(self):
        self.closed = self.total * np.apply_along_axis(lambda x: x / np.sum(x), 1, self.parts)
        return closed

    def plot(self):
        if not hasattr(self, 'closed'):
            self.closed = closure(self)
        plt.plot(self.closed[:,0], self.closed[:,1])



class OrthoBasis(object):
    """
    buildBase: Computes the othogonal basis from a sequential binary partition.
    sbpCheck: Checks if the SBP seems OK.
    """

    def __init__(self, sbp):
        self.sbp = np.array(sbp) # force parts to be an array

    def buildBase(self):
        W = self.sbp.transpose()
        dimW = W.shape
        isPos = (W > 0)
        isNeg = (W < 0)
        onesD = np.ones((dimW[0],dimW[0]))
        nPos = np.dot(onesD, isPos)
        nNeg = np.dot(onesD, isNeg)
        isPos*nNeg
        W = (isPos * nNeg - isNeg * nPos)
        nn = []

        for i in np.arange(0,W.shape[1]):
            x = 1/np.sqrt(np.dot(W[:,i], W[:,i]))
            nn.append(x)

        nn = np.array([nn,]*W.shape[0])
        V = W * nn
        return(V)

    def sbpCheck(self, sbpNames):
        sbpCheck = []
        for i in range(0,self.sbp.shape[0]):
            r = np.where(self.sbp[i,:] == 1)
            s = np.where(self.sbp[i,:] == -1)
            if r[0].shape[0] == 1:
                sbpCheck.append(sbpNames[r[0]])
            if s[0].shape[0] == 1:
                sbpCheck.append(sbpNames[s[0]])

        if sorted(sbpNames) == sorted(sbpCheck):
            message = "%s. The SBP seems OK." % sbpCheck
        else:
            message = "%s. The SBP might contain an error." % sbpCheck

        return(message)



class LogRatio(object):
    "Compute the log ratios for compositions"

    def __init__(self, comp, tol=1e-12):
        self.comp = np.atleast_2d(np.array(comp)) # force comp to be a 2D array

        # look if input is CoDa
        condition1 = (self.comp >= 0).all()
        condition2 = (self.comp <= 1).all()
        condition3 = (self.comp.sum(axis=1) >= 1-tol).all()
        condition4 = (self.comp.sum(axis=1) <= 1+tol).all()

        if condition1 and condition2 and condition3 and condition4:
            print("")
        else:
            print("Warning: Input data are not compositional data.")

    def alr(self, commonComp=0):
        t = self.comp.transpose()
        otherRows = list(range(0,t.shape[0])) # rows other than commonComp
        del otherRows[commonComp]
        t_arlMat = np.log(t[otherRows,:]/t[commonComp,:])
        alrTup = (t_arlMat.transpose(), commonComp, self.comp[:,commonComp])
        return(alrTup)

    def clr(self):
        geomean = np.apply_along_axis(lambda x: np.log(np.exp(np.mean(np.log(x)))), 1, self.comp)
        clrMat = np.apply_along_axis(lambda x: np.log(x / np.exp(np.mean(np.log(x)))), 1, self.comp)
        clrTup = (clrMat, geomean)
        return(clrTup)

    def ilr(self, sbp=None):
        if sbp is None: # create default sbp
            sbp = np.zeros(shape=[self.comp.shape[1]-1, self.comp.shape[1]])
            for i in range(0, sbp.shape[0]):
                sbp[i,i] = 1
                sbp[i,(i+1):] = -1
        V = OrthoBasis(sbp).buildBase()
        ilrMat = np.dot(np.log(self.comp), V)
        ilrTup = (ilrMat, sbp)
        return(ilrTup)


class InvLogRatio(object):
    "Inverse log ratios to obtain compositions"
    # attribuer une classe à chacun

    def __init__(self, lr):
        self.lr = lr

    def alr(self):
        self.alr = self.lr
        rat = (np.exp(self.alr[0])).sum(axis=1) + 1
        dat = (np.exp(self.alr[0]).transpose()/rat).transpose()
        lri = np.c_[dat[:,range(0,self.alr[1])],
                    1/rat,
                    dat[:,range(self.alr[1],np.shape(dat)[1])]]
        return(lri)

    def clr(self):
        self.clr = self.lr
        dat = np.exp(self.clr[0])
        lri = CloseComp(dat).closure()
        return(lri)

    def ilr(self, sbp=None):
        self.ilr = self.lr
        if sbp is None:
            sbp = self.ilr[1]
        V = OrthoBasis(sbp).buildBase()
        clr = np.dot(self.ilr[0], V.transpose())
        lri = CloseComp(np.exp(clr)).closure()
        return(lri)

class CodaDendrogram(object):
    "Plot the compositional dendrogram in matplotlib"

    def __init__(self, ilr):
        self.ilr = ilr

    def plot(self, sbp = None, equal_height = True, range_bal=None, show_range=True, fulcrum='median',
             add_observation=None, labels=None, leaf='down'):
        # sbp: matrix, if the sbp is not given, it take the one in the tuple (output of the LogRatio.ilr function)
        # equal_height: boolean value, True: the height of the vertical barrs are all equal, False: the heights
        ## are proportional to the proportion of total variance of the corresponding ilr balance
        # range_bal: list of two floats, the range of all horizontal bars
        # show_range: boolean. if True, the range is plotted on the horizontal bars
        # fulcrum: where the fulcrum is placed: 'median' and 'mean' are the statistics of the self.ilr matrix,
        ## and middle is the middle of the range_bal

        if sbp is not None:
            self.ilr = (self.ilr,sbp)

        sbp = self.ilr[1]
        V = OrthoBasis(sbp).buildBase() # Orthonormal basis
        signary = sbp.transpose()
        #Vzero = np.abs(V) > 1e-15 # non-zero values in the balance # never used

        # ordered matrices by number of parts
        nb_parts_in_bal = np.abs(signary).sum(axis=0) # number of parts implied in each balance
        order_o = nb_parts_in_bal.argsort(axis=0)[::-1] # [::-1] reverse vector
        V_o = V[:,order_o] # reordering the orthon. mat. per nb of parts
        signary_o = np.sign(V_o) # ordered signary (signary is the transposed sbp)
        binary_o = np.abs(signary_o) # ordered binary (binary is the absolute of the signary)
        bal_o = (self.ilr[0][:,order_o],self.ilr[1]) # reorder bal

        if add_observation is not None:
            if len(add_observation) != sbp.shape[1]:
                sys.exit("Error: Invalid add_observation length.")
            add_observation = np.array(add_observation)
            logRatio = LogRatio(add_observation, tol=1e-14)
            debal = logRatio.ilr(sbp)[0]
            if len(debal.shape) == 2:
                 debal = np.mean(debal, axis=0)
            debal_o = debal[order_o]
        signary_df = pd.DataFrame(signary)
        comp_id = np.array(signary_df.sort_index(by=range(0,signary_df.shape[1]), ascending=True).index)

        # hierarchy
        branch_from = np.zeros(shape=V_o.shape[1])
        branch_from[0] = np.nan
        branch_side = np.zeros(shape=V_o.shape[1])
        branch_side[0] = np.nan

        for i in np.arange(1,V_o.shape[1]):
            comp_not_null = V_o[:,i] != 0
            j=i-1
            branch_found = False
            while not branch_found:
                if np.all(V_o[comp_not_null,j]!=0):
                    branch_side[i] = np.sign(V_o[comp_not_null,j][0])
                    branch_from[i] = j
                    branch_found = True
                else:
                    j=j-1

        # balance statistics
        ## horizontal bars' ranges
        if range_bal is None:
            min_bal_o = np.amin(bal_o[0], axis=0)
            max_bal_o = np.amax(bal_o[0], axis=0)
            range_bal = np.vstack([min_bal_o, max_bal_o])
            auto_range = True
        else:
            range_bal_mat = np.array(range_bal)
            if len(range_bal_mat.shape) == 1: # if a list is given, repeat it for each
                for i in range(0,bal_o[0].shape[1]-1):
                    range_bal_mat = np.vstack([range_bal_mat, range_bal])
            range_bal = range_bal_mat.transpose()
            auto_range = False

        add_observation_pt_shape = np.repeat('o', sbp.shape[0])
        add_observation_pt_color = np.repeat('#C83737', sbp.shape[0])
        if add_observation is not None:
            for i in np.arange(0,len(debal_o)):
                if debal_o[i] < range_bal[0,i]:
                    debal_o[i] = range_bal[0,i]
                    add_observation_pt_shape[i] = '<'
                    add_observation_pt_color[i] = '#C83737'
                elif debal_o[i] > range_bal[1,i]:
                    debal_o[i] = range_bal[1,i]
                    add_observation_pt_shape[i] = '>'
                    add_observation_pt_color[i] = '#C83737'
                else:
                    add_observation_pt_shape[i] = 'o'
                    add_observation_pt_color[i] = '#AAD400'

        ## variance and centre
        if len(bal_o[0].shape) == 1:
            fulcrum = 'middle'
            centre_bal_o = range_bal[0,:]+(range_bal[1,:] - range_bal[0,:])/2
            print("Warning: the reference balance is a vector, equal_height is coerced to True and fulcrum is coerced to middle.")
            print("range should be defined") ## argggg
        else:
            var_bal_o = bal_o[0].var(axis=0)# variance per ordered bal
            if fulcrum == 'median':
                centre_bal_o = np.median(bal_o[0], axis=0) # mean per ordered bal
            elif fulcrum == 'mean':
                centre_bal_o = np.mean(bal_o[0], axis=0) # mean per ordered bal
            elif fulcrum == 'middle':
                centre_bal_o = range_bal[0,:]+(range_bal[1,:] - range_bal[0,:])/2
            else:
                sys.exit("Error: Invalid centre value.")

        # x-position of the nodes
        nodes_x = np.zeros(shape=(V_o.shape[1],2))
        is_leaf = np.zeros(shape=(V_o.shape[1],2), dtype=bool)
        for i in np.arange(0,V_o.shape[1])[::-1]:
            nparts = [np.sum(signary_o[:,i] < 0), np.sum(signary_o[:,i] > 0)] # nb of parts with -1 sign, nb of parts with 1 sign in the ith balance
            for j in [0,1]:
                if nparts[j] == 1:  # if there is only one part on the jth side of the ith balance
                    take = signary_o[:,i] == (-1)**(j+1)
                    take = np.arange(0,comp_id.shape[0])[take] == comp_id # to which comp_id does it correspond?
                    nodes_x[i,j] = (np.arange(0,comp_id.shape[0])+1)[take] # +1
                    is_leaf[i, j] = True  # indicate with 1 that the node is a leave (no more sub-balance)
                else:
                    take = signary_o[:,i] == (-1)**(j+1) # which one are they
                    aux = np.dot(np.repeat(1, repeats=np.sum(take)), binary_o[take,:])
                    aux[0:(i+1)] = 0
                    i1 = np.arange(0,aux.shape[0])[aux == np.max(aux)][0]
                    if auto_range:
                        f_interp = interpolate.interp1d(x=range_bal[:,i1], y=nodes_x[i1,:])
                    else:
                        f_interp = interpolate.interp1d(x=range_bal[:,i], y=nodes_x[i1,:])

                    nodes_x[i,j] = f_interp(centre_bal_o[i1])

        # heights of the fulcrums
        if equal_height:
            heights = np.repeat(1,V_o.shape[1])
            maxheight = 1.
            for i in np.arange(0,V_o.shape[1]):
                heights[i:V_o.shape[1]] = heights[i:V_o.shape[1]] - np.sign(np.dot(binary_o[:,i], binary_o[:,i:V_o.shape[1]]))
            inv_heights = np.max(np.abs(heights))+heights + 1 # +1 to elevate lowest from one step, in order to see all leave branches
            scaled_heights = inv_heights/np.max(inv_heights).astype("float64")
            heights = scaled_heights
            dh = np.repeat(heights[0] - heights[1], repeats = heights.shape[0])
        else:
            maxheight = np.max(np.dot(binary_o, var_bal_o))
            heights = np.repeat(maxheight,V_o.shape[1])
            for i in np.arange(0,V_o.shape[1]):
                heights[i:V_o.shape[1]] = heights[i:V_o.shape[1]] - var_bal_o[i] * np.sign(np.dot(binary_o[:,i], binary_o[:,i:V_o.shape[1]]))
            dh = var_bal_o

        # y-position of the nodes
        nodes_y = np.zeros(shape=(V_o.shape[1],2))
        for i in np.arange(0,V_o.shape[1])[::-1]:
                nodes_y[i,:] = np.repeat(heights[i],2)
        if add_observation is not None:
            heights_debal = np.copy(heights)
            range_sc = np.zeros(shape=range_bal.shape)
            nodes_dh_debal = np.zeros(shape=range_bal.shape[1])
            y_offset_debal = np.zeros(shape=range_bal.shape[1])
            for i in np.arange(0,V_o.shape[1])[::-1]:
                dl = np.abs(range_bal[0,i]-centre_bal_o[i])
                du = np.abs(range_bal[1,i]-centre_bal_o[i])
                if du > dl:
                    range_sc[0,i] = centre_bal_o[i] - du
                    range_sc[1,i] = range_bal[1,i]
                else:
                    range_sc[0,i] = range_bal[0,i]
                    range_sc[1,i] = centre_bal_o[i] + dl
                m = 2/(range_sc[1,i]-range_sc[0,i])
                b = 1-range_sc[1,i]*m
                debal_sc = debal_o[i]*m+b
                max_tilt_fac = 0.75#0.5 # 0.25 for visual: to check
                if i == 0:
                    nodes_dh_debal[i] = dh[i]*debal_sc*max_tilt_fac
                else:
                    nodes_dh_debal[i] = dh[branch_from[i]]*debal_sc*max_tilt_fac

                range_temp = (range_bal[1,i] - range_bal[0,i])/2 # range_sc ???
                middle = range_bal[0,i]+(range_bal[1,i] - range_bal[0,i])/2
                centre_offset = middle - centre_bal_o[i]
                perc_x_offset = centre_offset/np.abs(range_temp)
                y_offset = nodes_dh_debal[i]*perc_x_offset
                heights[i] = heights[i]+y_offset # y_offset: small offset because the fulcrum is not always on the center

                centre_offset_debal = middle - debal_o[i]
                perc_x_offset_debal = centre_offset_debal/np.abs(range_temp)
                y_offset_debal[i] = nodes_dh_debal[i]*perc_x_offset_debal

            for i in np.arange(0,V_o.shape[1]):
                isfrom = branch_from[i] # the index of the higher hierarchical fulcrum
                isnow = i # to specify on which side of the higher hierarchical fulcrum is the current fulcrum
                from_debal = 0 # the offset due to the debalancement of higher levels
                while not np.isnan(isfrom):
                    from_debal = from_debal-nodes_dh_debal[isfrom]*branch_side[isnow]
                    isnow = isfrom
                    isfrom = branch_from[isfrom]
                heights[i] = heights[i] + from_debal
                heights_debal[i] = heights_debal[i]+y_offset_debal[i] + from_debal # height of the debal point
                nodes_y[i,0] = nodes_y[i,0] + nodes_dh_debal[i] + from_debal
                nodes_y[i,1] = nodes_y[i,1] - nodes_dh_debal[i] + from_debal


        # plot
        fig = pl.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        if equal_height:
            ax.axes.get_yaxis().set_visible(False)
            ax.spines['left'].set_visible(False)

        ticks = range(1, V.shape[0]+1)
        if labels is None:
            labels = np.array(map(chr, range(65, 65+V.shape[0]))) # en attendant de créer un méthode pour les étiquettes
            labels_id = labels[comp_id]
        else:
            if len(labels) != V.shape[0]:
                sys.exit("Error: The lenght of argument labels should be equal to the number of balances + 1.")
            else:
                labels = np.array(labels)
                labels_id = labels[comp_id]

        if leaf == 'down':
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels_id)
            ax.tick_params(axis='x', length=0)
        elif leaf == 'up':
            ax.axes.get_xaxis().set_visible(False)
            ax.spines['bottom'].set_visible(False)
        else:
            sys.exit("Error: Invalid argument leaf. Should be 'up' or 'down'.")

        margin = 0.05
        if equal_height:
            ax.set_xlim(1-margin, V.shape[0]+margin)
            ax.set_ylim(0-margin, maxheight+dh[0]+margin)
        else:
            ax.set_xlim(0-margin, V.shape[0]+margin)
            ax.set_ylim(np.min(nodes_y)-margin, maxheight+margin)

        ## horizontal segments
        for i in np.arange(0, V.shape[1]):
            ax.plot(nodes_x[i,:], nodes_y[i,:], linestyle='-', color='black')

        ## leaf links
        if leaf == 'down':
            for i in np.arange(0, is_leaf.shape[0]):
                for j in np.arange(0, is_leaf.shape[1]):
                    if is_leaf[i,j]:
                        ax.plot(np.repeat(nodes_x[i,j],2), [nodes_y[i,j],0], linestyle=':', color='black')
        elif leaf == 'up':
            for i in np.arange(0, is_leaf.shape[0]):
                for j in np.arange(0, is_leaf.shape[1]):
                    if is_leaf[i,j]:
                        ax.plot(np.repeat(nodes_x[i,j],2), [nodes_y[i,j],nodes_y[i,j]-np.mean(dh)],
                                linestyle=':', color='black')
                        if j == 0:
                            leaf_label = labels[signary_o[:,i] == -1][0]
                        else:
                            leaf_label = labels[signary_o[:,i] == 1][0]
                        ax.text(nodes_x[i,j], nodes_y[i,j]-np.mean(dh),
                                leaf_label,
                                horizontalalignment='center', verticalalignment='center')
        else:
            sys.exit("Error: Invalid argument leaf. Should be 'up' or 'down'.")

        ## vertical lines
        for i in np.arange(0, V_o.shape[1]):
            f_interp = interpolate.interp1d(x=range_bal[:,i], y=nodes_x[i,:])
            interp_x_centre = f_interp(centre_bal_o[i])
            if i == 0:
                ax.plot(np.repeat(interp_x_centre,2), [heights[i],heights[i]+dh[i]], linestyle='-', color='black')
            else:
                # from the heights[i] (height of the centre) to the height of the node it is branched from
                # wheter it is in left (branch_side[i] = -1 or right = 1)
                if branch_side[i] < 0:
                    ax.plot(np.repeat(interp_x_centre,2), [heights[i], nodes_y[branch_from[i],0]],
                            linestyle='-', color='black')
                else:
                    ax.plot(np.repeat(interp_x_centre,2), [heights[i], nodes_y[branch_from[i],1]],
                            linestyle='-', color='black')
            ax.plot(interp_x_centre, heights[i], 'o', linestyle='-', color='black')
            if add_observation is not None:
                interp_x_centre_debal = f_interp(debal_o[i])
                ax.plot(interp_x_centre_debal, heights_debal[i], marker=add_observation_pt_shape[i], linestyle='-', color=add_observation_pt_color[i])


        ## balance ranges on nodes_x
        if show_range:
            for i in np.arange(0, V.shape[1]):
                for j in [0,1]:
                    label = np.around(range_bal[j,i], decimals=1)
                    ax.text(nodes_x[i,j], nodes_y[i,j]+maxheight/50, label, fontsize=10, va='bottom', ha='center')
