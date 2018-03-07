setwd('/home/essicolo/Documents/GitHub/nuee')


doubs_species = read.csv('data/DoubsSpe.csv', row.names = 1)
doubs_env = read.csv('data/DoubsEnv.csv', row.names = 1)


Y = apply(doubs_species, 2, function(x) x/mean(x))
W = doubs_env[, c('das', 'alt', 'pen', 'deb')]
X = doubs_env[, c('pH', 'dur', 'pho', 'nit', 'amm', 'oxy', 'dbo')]




test.axes.cov <- function(Y, X, W=NULL, nperm=199, seed=NULL, scaleY=FALSE, verbose=FALSE)
    #
    # Test of canonical eigenvalues in simple or partial RDA using the marginal and 
    # forward methods, in the presence of a matrix of covariables W. The function 
    # uses permutation of residuals of the reduced model (Freedman & Lane, 1983).
    #
    # This is a simple explicit program, without shortcuts nor compiled
    # permutation function. Its aim is to unambiguously describe the marginal 
    # and foward testing method in the presence of a matrix of covariables W. 
    # The code is interspersed with comments to explain the computation steps. 
    # This function is not intended for routine testing of canonical eigenvalues, 
    # although it produces correct, publishable results.
# 
# Parameters:
#    Y: response data matrix
#    X: explanatory data matrix
#    W: matrix of covariables
#    nperm: number of permutations
#    seed: seed for random number generator, used by the permutation function 
#          sample(). If seed=NULL (default), a random integer is drawn as the 
#          seed for the run. It will be reset to that value before the test of 
#          each canonical axis. All axes are thus tested using the same set of 
#          permutations. A fixed value, e.g. seed=12345, can be givenn by the 
#          user to compare the results of this function with that of other 
#          functions where the seed can also be set at run time.
#    scaleY = TRUE : standardize the Y variables
#           = FALSE: center the Y variables on their means
#    verbose = TRUE: print messages and intermediate F and F.perm results
#
# License: GPL-2 
# Authors: Pierre Legendre, Cajo J. F. ter Braak and Jari Oksanen, 2010
  {
    ##
    ## BEGIN: Internal functions
    ##
    test.axis1.raw <- function(Y, qr.X, eig.values, n, m, nperm,
                               SS.Y, SS.Y.fit.X, seed)
      # If no covariables: test canonical axis 1 by permutation of the raw data.
      # No residuals are computed on W or on previously tested axes.
    {
      # Compute F statistics: f.m for marginal method, F.f for forward method
      F.m <- eig.values[1] / (SS.Y - SS.Y.fit.X)       # F for marginal test
      F.f <- eig.values[1] / (SS.Y - eig.values[1])    # F for forward test
      sum.eigval <- eig.values[1]
      #
      set.seed(seed)
      nGE.m <- 1   # Hope correction: count 'Fstat' in the reference distribution
      nGE.f <- 1   # Hope correction: count 'Fstat' in the reference distribution
      if(verbose) cat('\n')
      for(iperm in 1:nperm) {
        Y.perm <- Y[sample(n),]
        SS.Y.perm <- SS.Y     # Same SS for Y and Y.perm after permuting rows of Y
        # 
        # Re-use the qr.X: does not change during permutations
        Y.fit.perm <- qr.fitted(qr.X, Y.perm)
        SS.Y.fit.perm <- sum(Y.fit.perm^2)
        Y.fit.perm.eig <- svd(Y.fit.perm, nv=0, nu=0)$d^2
        eig.value.perm <- Y.fit.perm.eig[1]
        #
        # Marginal method: F statistic under permutation
        den.m = (SS.Y.perm - SS.Y.fit.perm)
        F.perm.m <- eig.value.perm / den.m
        if(F.perm.m >= F.m) nGE.m <- nGE.m+1
        #
        # Forward method: F statistic under permutation
        den.f = (SS.Y.perm - eig.value.perm)
        F.perm.f <- eig.value.perm / den.f
        if(F.perm.f >= F.f) nGE.f <- nGE.f+1
        #
        if(verbose) cat("Axis 1    :  F.m =",F.m,"  F.perm.m =",F.perm.m,
                        "  F.f =",F.f,"  F.perm.f =",F.perm.f,'\n')
      }
      out <- c(F.m*(n-1-m), F.f*(n-1-m), nGE.m/(nperm+1), nGE.f/(nperm+1))
      out
    }
    
    test.following <- function(Y,X.resid,W,qr.W,qr.XW,begin,eig.values,n,mq,k,axes,
                               nperm,SS.Y,SS.Y.fit.XW,out,seed)
      # Test the following canonical axes by permuting residuals of reduced model
    {
      for(j in begin:k) {
        qr.W.axes1j <- qr(cbind(W, axes[,1:j]))
        Y.fit.W.axes1j <- qr.fitted(qr.W.axes1j, Y)
        SS.W.axes1j <- sum(Y.fit.W.axes1j^2) 
        #
        # Compute F statistics: F.m for marginal method, F.f for forward method
        F.m <- eig.values[j] / (SS.Y - SS.Y.fit.XW)    # F for marginal test
        F.f <- eig.values[j] / (SS.Y - SS.W.axes1j)    # F for forward test
        # F.f for forward method: from Canoco manual, eq. 3.12
        # Conformity with Canoco: (n-1-mq) d.f. for the denominator of F
        #
        # Compute reduced model fitted and residuals of Y on W and previous axes
        if(j==1) {
          qr.W.prev  <- qr.W
        } else {
          qr.W.prev  <- qr(cbind(W, axes[,1:(j-1)]))
        }
        Y.fit <- qr.fitted(qr.W.prev, Y)
        Y.res <- qr.resid(qr.W.prev, Y)
        qr.X.res <- qr(qr.resid(qr.W.prev, X.resid))
        #
        set.seed(seed)
        nGE.m <- 1   # Hope correction
        nGE.f <- 1   # Hope correction
        if(verbose) cat('\n')
        for(iperm in 1:nperm) {
          # Create permuted Y and compute its sum of squares (SS)
          Y.perm <- Y.fit + Y.res[sample(n),]
          SS.Y.perm <- sum(Y.perm^2)   # Not same SS as in the unpermuted data
          #
          # The j-th eigenvalue is the first eigenvalue of the partial RDA
          # of Y.perm by X in the presence of the previous axes[,1:(j-1)].
          # Prior to the permutation loop, X has been residualized on W and  
          # the previous axes.
          Y.fit.perm <- qr.fitted(qr.X.res, Y.perm)
          Y.fit.perm.eig1 <- svd(Y.fit.perm, nv=0, nu=0)$d[1]^2
          #
          # Marginal method:
          # For denominator of the F.m statistic: RDA of Y.perm on X.resid.
          # qr.X was computed in the main function.
          Y.fit.Tot.perm <- qr.fitted(qr.XW, Y.perm)
          SS.Y.fit.Tot.perm <- sum(Y.fit.Tot.perm^2)
          #
          # Forward method:
          # The fitted values of Y.perm on the previous axes [1:(j-1)]
          # provide the first [1:(j-1)] constrained eigenvalues for the 
          # denominator of F.f. We sum them, then we add the j-th eigenvalue 
          # 'Y.fit.perm.eig1' computed above.
          sum.eig.1toj.perm<-sum(qr.fitted(qr.W.prev, Y.perm)^2)+Y.fit.perm.eig1
          #
          # Marginal method: F statistic under permutation
          F.perm.m <- Y.fit.perm.eig1 / (SS.Y.perm - SS.Y.fit.Tot.perm)
          if(F.perm.m >= F.m) nGE.m <- nGE.m+1
          #
          # Forward method: F statistic under permutation
          F.perm.f <- Y.fit.perm.eig1 / (SS.Y.perm - sum.eig.1toj.perm)
          if(F.perm.f >= F.f) nGE.f <- nGE.f+1
          #
          if(verbose) cat("Axis [",j,"]:  F.m =",F.m,"  F.perm.m =",F.perm.m,
                          "  F.f =",F.f,"  F.perm.f =",F.perm.f,'\n')
        }
        vec.out <- c(F.m*(n-1-mq), F.f*(n-1-mq), nGE.m/(nperm+1), nGE.f/(nperm+1))
        out <- rbind(out, vec.out)         # Attach the new values to 'out'
      }
      out
    }
    ##
    ## END: Internal functions
    ##
    if(nperm > 5) verbose <- FALSE   # Modify this limit as needed #
    if(length(seed)==0) seed <- ceiling(runif(1,max=10000))
    cat("seed =",seed,'\n')
    a <- system.time({             # How much time for the permutation test?
      epsilon <- sqrt(.Machine$double.eps)
      #
      # Center or standardize the Y variables, standardize the X variables
      Y <- scale(as.matrix(Y), center=TRUE, scale=scaleY)
      X <- scale(as.matrix(X), center=TRUE, scale=TRUE)
      n <- nrow(Y)
      SS.Y <- sum(Y^2)
      #
      # Compute the rank of X
      m <- sum(svd(cov(X), nv = 0, nu = 0)$d > epsilon)
      
      # Is there a matrix W containing covariables?
      if(length(W)==0) {
        covar <- FALSE
        W <- rep(1,n)
        if(verbose) cat("\nThere is no matrix of covariables\n")
      } else {
        covar <- TRUE
        W <- scale(as.matrix(W), center=TRUE, scale=FALSE)
        if(verbose) cat("\nThere is a matrix of covariables\n")
      }
      # Compute the rank of cbind(X, W) 
      mq <- sum(svd(cov(cbind(X, W)), nv = 0, nu = 0)$d > epsilon)
      
      if(verbose) cat('n =',n,' m =',m,' mq =',mq,'\n')
      qr.W <- qr(W)
      if(length(W)!=0) {
        X.resid <- qr.resid(qr.W, X)  # Same as X.resid <- residuals(lm(X ~ W))
      } else {
        X.resid <- X
      }
      #
      # RDA consists of 2 steps: regression, then PCA of the table of fitted 
      # values by eigen decomposition of the matrix of Sums of Squares and Cross 
      # Products (SSCP), which is cov*(n-1)
      qr.X <- qr(X.resid)              # We use QR decomposition for efficiency
      Y.fit.X <- qr.fitted(qr.X, Y)    # Faster than Yhat<-fitted(lm(Y~X.resid))
      SS.Y.fit.X <- sum(Y.fit.X^2)     # fraction [a]
      Y.fit.eig <- svd(Y.fit.X, nv=0)  # SVD of Y.fit
      eig.values <- Y.fit.eig$d^2      # = eigen(cov(t(Y.fit.X)%*%Y.fit.X))$values
      k <- sum(eig.values > epsilon)   # Number of canonical eigenvalues
      axes <- Y.fit.eig$u[,1:k]        # Canonical axes = f(X.resid)
      
      qr.XW <- qr(cbind(X.resid,W))    #
      Y.fit.XW <- qr.fitted(qr.XW, Y)  # 
      SS.Y.fit.XW <- sum(Y.fit.XW^2)   # fraction [a+b+c]
      
      # cat("k =",k,"  SS.Y =",SS.Y," SS.Y.fit.X =",SS.Y.fit.X,
      # " SS.Y.fit.XW =",SS.Y.fit.XW,"\n")
      
      rsquare <- SS.Y.fit.X/SS.Y      # Semipartial R-square
      if(!covar) {
        adj.rsquare <- 1-((1-rsquare)*(n-1)/(n-m-1))
      } else {
        adj.rsquare <- NA
      }
      #
      # A section could be added to test the significance of the R-square (F-test).
      # A section could be added to produce a biplot with scaling type 1 or 2.
      #
      # Test the significance of the first canonical axis
      if(!covar) {
        temp <- test.axis1.raw(Y,qr.X,eig.values,n,m,nperm,SS.Y,SS.Y.fit.X,seed)
        out <- temp
        begin <- 2
      } else {
        begin <- 1
        out <- NULL
      }
      
      # Test the following canonical axes by permuting residuals of the reduced model
      if((covar) | ((!covar) & (k > 1))) {
        temp <- test.following(Y,X.resid,W,qr.W,qr.XW,begin,eig.values,n,mq,k,axes,
                               nperm,SS.Y,SS.Y.fit.XW,out,seed)
      }
      #
    })
    a[3] <- sprintf("%2f",a[3])
    cat("Computation time =",a[3]," sec",'\n')
    #
    # Output the results
    # The eigenvalues of the covariance matrix = (eigenvalues computed here)/(n-1)
    out <- cbind(eig.values[1:k]/(n-1), temp)
    rownames(out) <- NULL
    rownames(out) <- rownames(out, do.NULL=FALSE, prefix="Axis.")	
    colnames(out) <- c("Eigenvalue", "F.marginal", "F.forward", "P-marginal", "P-forward")
    res <- list(rsquare=rsquare, adj.rsquare=adj.rsquare, test.axes=out, axes=axes)
    res
  }
