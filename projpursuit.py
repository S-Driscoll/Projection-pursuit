from scipy.linalg import orth, sqrtm, lstsq
from numpy import *
from kurtosis import kurtosis
from scipy.linalg.lapack import dgesv


def rdiv(a, b):
    """Equivalent to matlab's a/b"""
    return matmul(a, linalg.inv(b))


def ldiv(a, b):
    """Equivalent to matlab's a\b"""
    if a.shape[0] == a.shape[1] and a.shape[0] == b.shape[0]:
        lu, piv, x, info = dgesv(a, b, False, False)
        return x
    else:
        return lstsq(a, b, rcond=None)[0]


def mean_(X: ndarray) -> ndarray:
    """Numpy's mean by default takes mean of all elements, while matlab's
    only takes mean across first dimension of size >1. In practice, matlab's mean is the
     mean of each column, so that's what this emulates"""
    return mean(X, axis=0).reshape(1, X.shape[1])


def sum_(X: ndarray, axis: int = 0) -> ndarray:
    """Equivalent of matlab's sum"""
    s = sum(X, axis=axis, keepdims=True)
    if max(s.shape) == 1:
        return float(s)
    else:
        return s


def svd(X: ndarray, full_matrices=True) -> (ndarray, ndarray, ndarray):
    """Singular value decomposition, matlab-style"""
    U, S_diag, W_h = linalg.svd(X, full_matrices=full_matrices)
    W = W_h.T
    S = zeros((X.shape[1], X.shape[1]))
    fill_diagonal(S, S_diag)
    return U, S, W


def diag_(X: ndarray) -> ndarray:
    # diag returns (n,) array, this converts to (n,1)
    return atleast_2d(diag(X)).T


def projpursuit(X: ndarray = None, **kwargs) -> [ndarray, ndarray, dict]:
    # PROJPURSUIT  Projection Pursuit Analysis
    #   T = PROJPURSUIT(X) performs projection pursuit analysis on the
    #   matrix X, using default algorithmic parameters (see below) and
    #   returns the scores in T.  The matrix X is mxn (objects x variables)
    #   and T is mxp (objects x scores), where the default value of p is 2.
    #
    #   Projection pusuit (PP) is an exploratory data analysis technique that
    #   seeks to optimize a projection index to find "interesting" projections
    #   of objects in a lower dimensional space.  In this algorithm, kurtosis
    #   (fourth statistical moment) is used as the projection index.
    #
    #   T = PROJPURSUIT(X,P) returns the first P projection pursuit scores.
    #   Usually P is 2 or 3 for data visualization (default = 2).
    #
    #   T= PROJPURSUIT(X,P,GUESS) uses GUESS initial random starting points for
    #   the optimization.  Larger values of GUESS decrease the likelihood of a
    #   local optimum, but increase computation time.  The default value is
    #   GUESS=100.
    #
    #   T = PROJPURSUIT(X,...,S1,S2,...) specifies algorithmic variation of
    #   the PP analysis, where S1, S2, etc. are character strings as specified
    #   with the options below.
    #
    #      Meth:    Stepwise Unvariate ('Uni') or Multivariate ('Mul') Kurtosis
    #      CenMeth: Ordinary ('Ord') or Recentered ('Rec') Kurtosis
    #      VSorth:  Orthogonal Scores ('SO') or Orthogonal Loadings ('VO')
    #      MaxMin:  Minimization ('Min') or Maximization ('Max') of Kurtosis
    #      StSh:    Shifted ('Sh') or Standard ('St') Optimization Method
    #
    #   In each case, the default option is the first one.  These variations
    #   are discussed in more detail below under the heading 'Algorithms'.
    #
    #   [T,V] = PROJPURSUIT(...) returns the P loading vectors in V (nxp).
    #
    #   [T,V,PPOUT] = PROJPURSUIT(...) returns additional outputs from the PP
    #   analysis in the structured variable PPOUT. These vary with the
    #   algorithm selected, as indicated below.
    #        PPOUT.K:  Kurtosis value(s) for the optimum subspace. Can
    #                  otherwise be found by searching for the max/min of
    #                  PPOUT.kurtObj. For multivariate methods, this is a
    #                  scalar; for univariate methods, it is a 1xP vector
    #                  corresponding to the optimum value in each step.
    #        PPOUT.kurtObj: Kurtosis values for different initial guesses.
    #        PPOUT.convFlag: Convergence status for different initial guesses.
    #        PPOUT.W:  If the scores are made orthogonal for univariate
    #                  methods, W and P are intermediate matrices in the
    #                  calculation of deflated matrices. The loadings are not
    #                  orthogonal in this case and are given by V=W*inv(P'*W).
    #                  If the projection vectors are set to be orthogonal, or
    #                  multivariate algorithms are used, these are not
    #                  calculated.
    #        PPOUT.P:  See PPOUT.W.
    #        PPOUT.Mu: The estimated row vector subtracted from the data
    #                  set, X, for re-centered methods.
    #
    #   Algorithms:
    #
    #   Univariate vs. Multivariate
    #      In the stepwise univariate PP algorithm, univariate kurtosis is
    #      optimized as the projection vectors are extracted sequentially,
    #      with deflation of the original matrix at each step. In the
    #      multivariate algorithm, multivariate kurtosis is optimized as
    #      all of the projection vectors are calculated simultaneously.
    #      Univariate is best for small numbers of balanced clusters that can
    #      be separated in a binary fashion and runs faster than the
    #      multivariate algorithm.
    #
    #   Minimization vs Maximization
    #      Minimization of kurtosis is most often used to identify clusters.
    #      Maximization may be useful in identifying outliers. Maximization
    #      is not an option for recentered algorithms.
    #
    #   Orthogonal Scores vs. Orthogonal Loadings
    #      This option is only applicable to stepwise univariate algorithms
    #      for P>1 and relates to the deflation of the data matrix in the
    #      stepwise procedure. Orthogonal scores are generally preferred,
    #      since these avoid correlated scores in multiple dimensions.
    #      However, the projection vectors (loadings) will not be orthogonal
    #      in this case.  For multivariate methods, the loadings are always
    #      orthogonal.
    #
    #   Ordinary vs. Recentered Algorithms
    #      For data sets that are unbalanced (unequal number of members in each
    #      class, the recentered algorithms may provide better results than
    #      ordinary PP.
    #
    #   Shifted vs. Standard Algorithms
    #      This refers to the mathematics of the quasi-power method. The
    #      shifted algorithm should be more stable, but the option for the
    #      standard algorithm has been retained. The choice is not available
    #      for recentered algorithms, and the shifted algorithm may still be
    #      implemented if solutions become unstable.

    #
    #                             Version 1.0
    #
    # Original algorithms written by Siyuan Hou.
    # Additional modifications made by Peter Wentzell, Steve Drisoll, Bonnie Russell, Chelsi Wicks.
    # Python conversion by Bonnie Russell
    #

    # Set Default Parameters
    MaxMin = kwargs.get("MaxMin", "min").lower()
    StSh = kwargs.get("StSh", "sh").lower()
    VSorth = kwargs.get("VSorth", "so").lower()
    Meth = kwargs.get("Meth", "uni").lower()
    CenMeth = kwargs.get("CenMeth", "ord").lower()
    p = kwargs.get("p", 2)
    guess = kwargs.get("guess", 100)
    convlimit = kwargs.get("convlimit", 1e-10)
    ppout = {"W": [], "P": [], "Mu": []}

    # Check for valid inputs and parse as required
    if X is None:
        raise Exception("PP:DefineVar:X: Provide data matrix X")
    elif type(X) != ndarray or X.dtype != dtype(float):
        raise Exception("PP:InvalVar:X: Invalid data matrix X")

    # Check numeric variables
    m, n = X.shape
    if p < 1:
        raise Exception("PP:InvalVar:p: Invalid value for subspace dimension.")
    if guess < 1:
        raise Exception("PP:InvalVar:guess: Invalid value for number of guesses")
    if m < p + 1 or n < p + 1:
        raise Exception("PP:InvalVar:X: Insufficient size of data matrix.")

    # Set options for algorithm
    if MaxMin not in ["min", "max"]:
        raise Exception("PP:InvMode:MaxMin: Choose either to minimize or maximize.")
    if MaxMin == "max" and CenMeth == "rec":
        raise Exception(
            "PP:InvMode:MaxMin: Maximization not available for recentered PP."
        )
    if StSh not in ["st", "sh"]:
        raise Exception("PP:InvMode:StSh: Choose either the standard or shifted method")
    if VSorth not in ["vo", "so"]:
        raise Exception(
            "PP:InvMode:VSorth: Choose for either the scores or the projection vectors to be orthogonal"
        )
    if Meth not in ["mul", "uni"]:
        raise Exception(
            "PP:InvMode:UniMul: Choose either univariate or multivariate method"
        )
    if CenMeth not in ["rec", "ord"]:
        raise Exception(
            "PP:InvMode:OrdRec: Choose either the ordinary or recentred method"
        )

    # Carry out PP using appropriate algorithm
    if Meth == "mul":
        if CenMeth == "rec":
            print("Performing recentered multivariate PP")  # Diagnostic
            T, V, R, K, Vall, kurtObj, convFlag = rcmulkurtpp(
                X, p, guess, convlimit=convlimit
            )
            ppout["K"] = K
            ppout["kurtObj"] = kurtObj
            ppout["convFlag"] = convFlag
            ppout["Mu"] = R
        else:
            print("Performing ordinary multivariate PP({})".format(StSh))  # Diagnostic
            T, V, Vall, kurtObj, convFlag = mulkurtpp(
                X, p, guess, MaxMin, StSh, convlimit=convlimit
            )
            ppout["K"] = kurtObj.min(0)
            ppout["kurtObj"] = kurtObj
            ppout["convFlag"] = convFlag
    else:
        if CenMeth == "rec":
            print(
                "Performing recentered univariate PP({})".format(VSorth)
            )  # Diagnostic
            T, V, R, W, P, kurtObj, convFlag = rckurtpp(
                X, p, guess, VSorth, convlimit=convlimit
            )
            ppout["K"] = kurtObj.min(0)
            ppout["kurtObj"] = kurtObj
            ppout["convFlag"] = convFlag
            ppout["W"] = W
            ppout["P"] = P
            ppout["Mu"] = R
        else:
            print(
                "Performing ordinary univariate PP({},{})".format(StSh, VSorth)
            )  # Diagnostic
            T, V, W, P, kurtObj, convFlag = okurtpp(
                X, p, guess, MaxMin, StSh, VSorth, convlimit=convlimit
            )
            ppout["K"] = kurtObj.min(0)
            ppout["kurtObj"] = kurtObj
            ppout["convFlag"] = convFlag
            ppout["W"] = W
            ppout["P"] = P
    return T, V, ppout


# Original Univariate Kurtosis Projection Pursuit Algorithm
def okurtpp(X, p, guess, MaxMin, StSh, VSorth="SO", convlimit=1e-10):
    """Returns (T,V,W,P,kurtObj,convFlag)"""
    # Quasi-power methods to optimize univariate kurtosis
    #
    #
    # Input:
    #       X:       The data matrix. Rows denote samples, and columns denote variables.
    #       p:       The number of projection vectors to be extracted.
    #       guess:   The number of initial guesses for optimization,e.g. 100.
    #                The more dimensions, the better to have more initial guesses.
    #       MaxMin:  A string indicating to search for maxima or minima of kurtosis.
    #                The available choices are "Max" and "Min".
    #                   "Max": To search for maxima of kurtosis
    #                   "Min": To search for minima of kurtosis
    #                Projections revealing outliers tend to have a maximum
    #                kurtosis, while projections revealing clusters tend to
    #                have a minimum kurtosis. Maximization seems more important
    #                in ICA to look for independent source signals, while
    #                minimization appears useful in PP to looks for clusters.
    #       StSh:    A string indicating if the standard or the shifted algorithm
    #                is used. The available choices are "St" and "Sh".
    #                   "St": To use the standard quasi-power method.
    #                   "Sh": To use the shifted quasi-power method.
    #       VSorth:  A string indicating whether the scores or projection
    #                vectors are orthogonal. The available choices are
    #                   "VO": The projection vectors are orthogonal, but
    #                         scores are not, in general.
    #                   "SO": The scores are orthogonal, but the projection
    #                         vectors are not, in general.
    #                If not specified (empty), the scores are made orthogonal.
    # Output:
    #       T:        Scores.
    #       V:        Projection vectors.
    #       W & P:    If the scores are made orthogonal, they appear in the
    #                 deflation steps. They can be used to calculate the final
    #                 projection vectors with respect to the original matrix.
    #                 If the projection vectors are set to be orthogonal, they
    #                 are not needed.
    #       kurtObj:  Kurtosis values for different initial guesses.
    #       convFlag: Convergence status for different initial guesses.

    # Note:
    #
    # The scores orthogonality is based on mean-centered data. If the data
    # are not mean-centered, the mean scores are added to the final scores and
    # therefore the final scores may not be not orthogonal.
    #
    # For minimization of kurtosis, the standard method (st) may not be stable
    # when the number of samples is only slightly larger than the number of
    # variables. Thus, the shifted method (sh) is recommended.

    # Author:
    # S. Hou, University of Prince Edward Island, Charlottetown, PEI, Canada, 2012.
    #
    # Version, Nov. 2012. This is the updated version. The original version was
    # reported in the literature: S. Hou, and P. D. Wentzell, Fast and Simple
    # Methods for the Optimization of Kurtosis Used # as a Projection Pursuit
    # Index, Analytica Chimica Acta, 704 (2011) 1-15.
    if VSorth.upper() not in ["VO", "SO"]:
        raise Exception(
            "Please correctly choose the orthogonality of scores or projection vectors."
        )

    if StSh.upper() in ["ST", "SH"]:
        StSh0 = StSh
    else:
        raise Exception('Please correctly choose "St" or "Sh" method.')

    #  Mean center the data and reduce the dimensionality of the data
    # if the number of variables is larger than the number of samples.
    Morig = ones([X.shape[0], 1]) @ mean_(X)
    X = X - Morig
    rk = linalg.matrix_rank(X)
    if p > rk:
        p = rk
        print("The component number larger than the data rank is ignored.")
    Uorig, Sorig, Worig = svd(X, full_matrices=False)
    X = Uorig @ Sorig
    X = X[:, 0:rk]
    Worig = Worig[:, 0:rk]
    X0 = X.copy()
    # Initial settings
    r, c = X.shape
    maxcount = 10000
    convFlag = empty((guess, p), dtype=object)
    kurtObj = zeros((guess, p))
    T = zeros((r, p))
    W = zeros((c, p))
    P = zeros((c, p))

    for j in range(1, p + 1):
        cc = c + 1 - j  # todo: may need to adjust off by one
        convlimit = convlimit * cc  # Set convergence limit
        wall = zeros((cc, guess))
        U, S, Vj = svd(X)
        Vj = Vj[:, 0:cc]  # This reduces the dimensionality of the data
        X = X @ Vj  # when deflation is performed.
        if MaxMin.upper() == "MAX":  # Option to search for maxima.
            invMat2 = atleast_2d(1 / diag(X.T @ X)).T
        elif MaxMin.upper() == "MIN":  # Option to search for minima.
            Mat2 = diag_(X.T @ X)
            Mat2 = Mat2.reshape(Mat2.shape[0], 1)
            VM = zeros((cc * cc, r))  # This is used to calculate "Mat1a" later
            for i in range(r):
                tem = X[i : i + 1, :].T @ X[i : i + 1, :]
                VM[:, i : i + 1] = tem.reshape(cc * cc, 1, order="F").copy()
        else:
            raise Exception(
                "Please correctly choose to maximize or minimize the kurtosis."
            )

        # optimize some flag checks
        MAX = MaxMin.upper() == "MAX"
        MIN = MaxMin.upper() == "MIN"

        # Loop for different initial guesses of w
        for k in range(guess):
            w = random.randn(cc, 1)  # Random initial guess of w for real numbers
            w = w / linalg.norm(w)
            oldw1 = w.copy()
            oldw2 = oldw1.copy()
            StSh = StSh0
            count = 0
            while count <= maxcount:
                count += 1
                x = X @ w
                # Maximum or minimum search
                if MAX:  # Option to search for maxima.
                    w = invMat2 * (X.T @ (x * x * x))
                elif MIN:  # Option to search for minima.
                    sq_x = square(x)
                    tmp = VM @ sq_x
                    Mat1 = sum_(tmp, 1)
                    Mat1 = Mat1.reshape(cc, cc, order="F").copy()
                    tmp = Mat2 * w
                    w = ldiv(Mat1, tmp)
                # Test convergence
                w = w / linalg.norm(w)
                L1 = (w.T @ oldw1) ** 2
                if (1 - L1) < convlimit:
                    convFlag[k, j - 1] = "Converged"
                    break  # Exit the "while ... end" loop if converging
                # Continue the interation if "break" criterion is not reached
                if StSh.upper() == "SH":  # Shifted method
                    w = w + 0.5 * oldw1
                    w = w / linalg.norm(w)
                elif MIN:  # "St" method & minimization
                    L2 = (w.T @ oldw2) ** 2  # If "St" method is not stable,
                    if L2 > L1 and L2 > 0.99:  # change to shifted method
                        StSh = "Sh"
                        print(
                            'Warning: "St" method is not stable. Change to shifted method.'
                        )
                    oldw2 = oldw1
                    # "St" method & maximization: do nothing
                oldw1 = w
            if count > maxcount:
                convFlag[k, j - 1] = "Not converged"
            # Save the projection vectors for different initial guesses
            wall[:, k] = w[:, 0]
        # Find the best solution from different initial guesses
        kurtObj[:, j - 1] = kurtosis(X @ wall, 1, 0)
        if (
            MaxMin.upper() == "MAX"
        ):  # Find the best projection vector for maximum search.
            tem = max(kurtObj[:, j - 1])
        elif (
            MaxMin.upper() == "MIN"
        ):  # Find the best projection vector for minimum search.
            tem = min(kurtObj[:, j - 1])
        else:
            raise Exception("Must set MIN or MAX")
        ind = where(kurtObj[:, j - 1] == tem)[0]
        Wj = wall[:, ind]  # Take the best projection vector as the solution.

        # Deflation of matrix
        if VSorth.upper() == "VO":  # This deflation method makes the
            t = X @ Wj  # projection vectors orthogonal.
            T[:, j - 1] = t[:, 0]
            W[:, j - 1] = (Vj @ Wj)[:, 0]
            X = X0 - X0 @ W @ W.T
        elif (
            VSorth.upper() == "SO"
        ):  # This deflation method makes the scores orthogonal.
            t = (
                X @ Wj
            )  # This follows the deflation method used in the non-linear partial
            T[:, j - 1] = t[
                :, 0
            ]  # least squares (NIPALS), which is well-known in chemometrics.
            W[:, j - 1] = (Vj @ Wj)[:, 0]
            Pj = rdiv(X.T @ t, (t.T @ t))
            P[:, j - 1] = (Vj @ Pj)[:, 0]
            X = (
                X0 - T @ P.T
            )  # This uses the Gram-Schmidt process for complex-valued vectors
    # Transform back into original space
    W = Worig @ W  # The projection vector(s) are tranformed into original space.
    if VSorth.upper() == "VO":
        V = W.copy()
        W = array([])
        P = array([])
        T = T + Morig @ V  # Adjust the scores. Mean scores are added.
    else:
        P = Worig @ P  # Vectors in P are tranformed into original space.
        V = W @ linalg.inv(P.T @ W)  # Calculate the projection vectors by V=W*inv(P'*W)
        T = T + Morig @ V  # Adjust the scores. Mean scores are added.
        tem = sqrt(sum_(abs(V) ** 2))
        V = V / (
            ones((V.shape[0], 1)) @ tem
        )  # Make the projection vectors be unit length
        T = T / (ones((T.shape[0], 1)) @ tem)  # Adjust T with respect to V
        P = P * (ones((P.shape[0], 1)) @ tem)  # Adjust P with respect to V
    return T, V, W, P, kurtObj, convFlag


# Original Multivariate Kurtosis Projection Pursuit Algorithm
def mulkurtpp(X, p, guess, MaxMin, StSh, convlimit=1e-10):
    """Returns [T,V,Vall,kurtObj,convFlag]"""
    #
    # Quasi-power method to optimize multivariate kurtosis.
    #
    # Input:
    #       X:      The data matrix.
    #       p:      The dimension of the plane or heperplane (Normally, 2 or 3).
    #       guess:  The number of initial guesses for optimization.
    #               The more dimension, the better to have more initial guesses.
    #       MaxMin: A string indicating to search for maxima or minima of kurtosis.
    #               The available choices are "Max" and "Min".
    #                   "Max": To search for maxima of kurtosis
    #                   "Min": To search for minima of kurtosis
    #               Projections revealing outliers tend to have a maximum
    #               kurtosis, while projections revealing clusters tend to
    #               have a minimum kurtosis.
    #       StSh:   A string indicating if the standard or the shifted algorithm
    #               is used. The available choices are "St" and "Sh".
    #                   "St": To use the standard quasi-power method.
    #                   "Sh": To use the shifted quasi-power method.
    # Output:
    #       T:        Scores.
    #       V:        Projection vectors.
    #       Vall:     All the projection vectors found based on different initial guesses. The
    #                 best projection vectors are chosen as the solutions and put in V
    #       kurtObj:  Kurtosis values for different projection vectors.
    #       convFlag: Convergence status for the initial guesses..
    #
    #  Mean center the data and reduce the dimensionality of the data if the number
    #  of variables is larger than the number of samples.
    Morig = ones((X.shape[0], 1)) @ mean_(X)
    X = X - Morig
    rk = linalg.matrix_rank(X)
    Uorig, Sorig, Vorig = svd(X, full_matrices=False)
    X = Uorig @ Sorig
    X = X[:, 0:rk]
    Vorig = Vorig[:, 0:rk]
    r, c = X.shape

    # Initial settings
    maxcount = 10000
    Vall = empty((1, guess), dtype=object)
    kurtObj = zeros((1, guess))
    convFlag = empty((1, guess), dtype=object)

    for k in range(guess):
        V = random.randn(c, p)  # Random initial guess of V
        V = orth(V)
        oldV = V.copy()
        count = 0
        while True:
            count += 1
            A = V.T @ X.T @ X @ V
            Ainv = linalg.inv(A)
            scal = sqrtm(Ainv) @ V.T @ X.T
            scal = sqrt(sum_(scal ** 2))
            Mat = (ones((c, 1)) @ scal) * X.T
            Mat = Mat @ Mat.T
            if MaxMin.upper() == "MAX":  # Option to search for maxima.
                M = linalg.inv(X.T @ X) @ Mat
                if StSh.upper() == "ST":
                    V = M @ V
                elif StSh.upper() == "SH":
                    V = (M + eye(c) * trace(M) / c) @ V
                else:
                    raise Exception(
                        "Please correctly choose to standard or shifted method."
                    )
            elif MaxMin.upper() == "MIN":  # Option to search for minima.
                M = linalg.inv(Mat) @ (X.T @ X)
                if StSh.upper() == "ST":
                    V = M @ V
                elif StSh.upper() == "SH":
                    V = (M + eye(c) * trace(M) / c) @ V
                else:
                    raise Exception(
                        "Please correctly choose to standard or shifted method."
                    )
            else:
                raise Exception(
                    "Please correctly choose to maximize or minimize the kurtosis."
                )

            V, TemS, TemV = svd(
                V, full_matrices=False
            )  # Apply SVD to find an orthonormal basis.
            if all(sum_((oldV - V) ** 2) / (c * p) < convlimit):  # Test convergence.
                convFlag[0, k] = "Converged"
                break
            elif count > maxcount:
                convFlag[0, k] = "Not converged"
                break
            oldV = V.copy()
        kurtObj[0, k] = r * sum_(
            (sum_((sqrtm(Ainv) @ V.T @ X.T) ** 2, 0)) ** 2, 1
        )  # Calculate kurtosis.

        U, S, V = svd(X @ V @ V.T)
        Vall[0, k] = Vorig @ V[:, 0:p]

    if MaxMin.upper() == "MAX":  # Find the best projection vector for maximum search.
        tem = max(kurtObj[0, :])
    elif MaxMin.upper() == "MIN":  # Find the best projection vector for minimum search.
        tem = min(kurtObj[0, :])
    ind = where(kurtObj[0, :] == tem)[0][0]
    V = Vall[0, ind]  # Store the projection vectors
    T = X @ Vorig.T @ V + Morig @ V  # Calculate the scores.
    return T, V, Vall, kurtObj, convFlag


# Recentered Univariate Kurtosis Projection Pursuit Algorithm
def rckurtpp(X, p, guess, VSorth="SO", convlimit=1e-10):
    """Returns [T,V,R,W,P,kurtObj,convFlag]"""
    #
    # Algorithms for minimization of recentered kurtosis. recentered kurtosis
    # is proposed as a projection pursuit index in this work, aiming to deal with
    # unbalanced clusters.
    #
    # Input:
    #       X:        The data matrix.
    #       p:        The number of projection vectors to be extracted.
    #       guess:    The number of initial guesses for optimization.
    #                 The more dimensions, the better to have more initial guesses.
    #       VSorth:   A string indicating whether the scores or projection
    #                 vectors are orthogonal. The available choices are
    #                   "VO": The projection vectors are orthogonal, but
    #                         scores are not, in general.
    #                   "SO": The scores are orthogonal, but the projection
    #                         vectors are not, in general.
    #                If not specified (empty), the scores are made orthogonal.
    # Output:
    #       T:        Scores.
    #       V:        Projection vectors.
    #       R:       The estimated row vector subtracted from the data set X.
    #       W & P:    If users choose scores are orthogonal, they appear in the
    #                 deflation steps. They can be used to calculate the final
    #                 projection vectors with respect to the original matrix X.
    #                 If the projection vectors are set to be orthogonal, they
    #                 are not needed.
    #       kurtObj:  Kurtosis values for different initial guesses.
    #       convFlag: Convergence status for different initial guesses.

    # Note:
    # Users have the option to make the projection vectors or scores orthogonal.
    # The scores orthogonality is based on mean-centered data. If the data
    # are not mean-centered, the mean scores are added to the final scores and
    # therefore the final scores may not be not orthogonal.
    # Author:
    # S. Hou, University of Prince Edward Island, Charlottetown, PEI, Canada, 2012.
    #
    # This algorithm is based on the Quasi-Power methods. The Quasi-power
    # methods were reported in the literature: S. Hou, and P. D. Wentzell,
    # Fast and Simple Methods for the Optimization of Kurtosis Used as a
    # Projection Pursuit Index, Analytica Chimica Acta, 704 (2011) 1-15.
    #
    #
    if VSorth.upper() not in ["VO", "SO"]:
        raise Exception(
            "Please correctly choose the orthogonality of scores or projection vectors."
        )
    #
    #  Mean center the data and reduce the dimensionality of the data
    # if the number of variables is larger than the number of samples.
    Morig = mean_(X)
    X = X - ones((X.shape[0], 1)) * Morig
    rk = linalg.matrix_rank(X)
    if p > rk:
        p = rk
        print("The component number larger than the data rank is ignored.")
    #
    Uorig, Sorig, Worig = svd(
        X, full_matrices=False
    )  # the matlab code also passed 'econ', which is equivalent to full_matrices=False
    X = Uorig @ Sorig
    X = X[:, 0:rk]
    Worig = Worig[:, 0:rk]
    X0 = X.copy()
    # Initial settings
    r, c = X.shape
    maxcount = 10000
    convFlag = empty((guess, p), dtype=object)
    kurtObj = zeros((guess, p))
    T = zeros((r, p))
    W = zeros((c, p))
    P = zeros((c, p))
    ALPH = zeros((1, p))

    for j in range(1, p + 1):
        cc = c + 1 - j
        convlimit = convlimit * cc  # Set convergence limit
        wall = zeros((cc, guess))
        alphall = zeros((1, guess))
        U, S, Vj = svd(X)
        Vj = Vj[:, 0:cc]  # This reduces the dimensionality of the data
        X = X @ Vj  # when deflation is performed.
        for k in range(guess):
            w = random.randn(cc, 1)  # Random initial guess of w for real numbers
            w = w / linalg.norm(w)
            alph = mean_(X @ w)
            oldw1 = w.copy()
            oldw2 = oldw1.copy()
            count = 0
            while True:
                count += 1
                x = X @ w
                xalph = x - alph
                alph = alph + sum_(xalph ** 3) / (
                    3 * sum_(xalph ** 2)
                )  # Update alpha (alph) value
                mu = alph @ w.T  # Updata mu, given w and alpha (alph)
                tem = (x - alph) ** 2
                dalph_dv = (X.T @ tem) / sum_(tem)  # Calculate dalpha/dv
                tem1 = X.T - dalph_dv @ ones((1, r))
                tem2 = X - ones((r, 1)) @ mu
                Mat1 = ((ones((cc, 1)) @ tem.T) * (tem1)) @ (tem2)
                Mat2 = tem1 @ tem2
                w = ldiv(Mat1, (Mat2 @ w))  # updata w
                # Test convergence
                w = w / linalg.norm(w)
                L1 = (w.T @ oldw1) ** 2
                if (1 - L1) < convlimit:
                    convFlag[k, j - 1] = "Converged"
                    break  # Exit the "while ... end" loop if converging
                elif count > maxcount:
                    convFlag[k, j - 1] = "Not converged"
                    break  # Exit if reaching the maximum iteration number
                # Continue the interation if "break" criterion is not reached
                L2 = (w.T @ oldw2) ** 2
                if L2 > L1 and L2 > 0.95:
                    w = w + (random.rand() / 5 + 0.8) * oldw1
                    w = w / linalg.norm(w)
                oldw2 = oldw1.copy()
                oldw1 = w.copy()
            # Save the projection vectors for different initial guesses
            wall[:, k] = w[:, 0]
            alphall[0, k] = alph
        # Take the best projection vector as the solution
        kurtObj[:, j - 1] = (
            r
            * sum_((X @ wall - ones((r, 1)) @ alphall) ** 4, axis=0)
            / ((sum_((X @ wall - ones((r, 1)) @ alphall) ** 2, axis=0)) ** 2)
        ).T[:, 0]
        tem = min(kurtObj[:, j - 1])
        ind = where(kurtObj[:, j - 1] == tem)[0]
        Wj = wall[:, ind]  # Take the best projection vector as the solution.
        for i in range(cc):
            if Wj[i] != 0:
                signum = sign(Wj[i])  # Change the sign of w to make it unique
                break
        Wj = Wj * signum
        ALPH[0, j - 1] = alphall[0, ind] @ signum
        # Deflation of matrix
        if VSorth.upper() == "VO":  # This deflation method makes the
            t = X @ Wj  # projection vectors orthogonal.
            T[:, j - 1] = t[:, 0]
            W[:, j - 1] = (Vj @ Wj)[:, 0]
            X = X0 - X0 @ W @ W.T
        elif (
            VSorth.upper() == "SO"
        ):  # This deflation method makes the scores orthogonal.
            t = (
                X @ Wj
            )  # This follows the deflation method used in the non-linear partial
            T[:, j - 1] = t[
                :, 0
            ]  # least squares (NIPALS), which is well-known in chemometrics.
            W[:, j - 1] = (Vj @ Wj)[:, 0]
            Pj = X.T @ t / (t.T @ t)
            P[:, j - 1] = (Vj @ Pj)[:, 0]
            X = X0 - T @ P.T
    # Transform back into original space
    W = Worig @ W  # The projection vector(s) are tranformed into original space.
    if VSorth.upper() == "VO":
        V = W.copy()
        W = array([])
        P = array([])
        T = T + ones((r, 1)) @ Morig @ V  # Adjust the scores. Mean scores are added.
        R = ALPH @ V.T + Morig
    else:
        P = Worig @ P  # Vectors in P are tranformed into original space.
        V = W @ linalg.inv(P.T @ W)  # Calculate the projection vectors by V=W*inv(P'*W)
        T = T + ones((r, 1)) @ Morig @ V  # Adjust the scores. Mean scores are added.
        R = ALPH @ (P.T @ W) @ W.T + Morig
        tem = sqrt(sum_(abs(V) ** 2))
        V = V / (
            ones((V.shape[0], 1)) * tem
        )  # Make the projection vectors be unit length
        T = T / (ones((T.shape[0], 1)) * tem)  # Adjust T with respect to V
        P = P * (ones((P.shape[0], 1)) * tem)  # Adjust P with respect to V
    return T, V, R, W, P, kurtObj, convFlag


# Recentered Multivariate Kurtosis Projection Pursuit Algorithm
def rcmulkurtpp(X, p, guess, convlimit=1e-10):
    """Returns [T,V,R,K,Vall,kurtObj,convFlag]"""
    #
    # Algorithms for minimization of re-centered multivariate kurtosis that is
    # used as a project pursuit index. This algorithm aims to deal with
    # unbalanced clusters (multivariate version). The effect of dimension is
    # taken into account by introducing a dimension term in the constraint.
    #
    # Input:
    #       X:      The data matrix. X cannot be singular.
    #       p:      The dimensionality of the plane or heperplane (Normally, 2 or 3).
    #       guess:  The number of initial guesses for optimization.
    # Output:
    #       T:      Scores of the chosen subspace (with the lowest multivariate
    #               kurtosis value).
    #       V:      Projection vectors for the chosen subspace.
    #       R:      The estimated row vector subtracted from the data set X.
    #       K:      Multivariate kurtosis value for the chosen subspace.
    #       Vall:   All the projection vectors found based on different initial guesses. The
    #               best projection vectors are chosen as the solutions and put in V.
    #       kurtObj:   Kurtosis values for the projection vectors of different initial guesses.
    #       convFlag: Convergence status for the different initial guesses.
    #
    #
    # This algorithm extends the Quasi-Power methods reported in two papers:
    # (1) S. Hou, and P. D. Wentzell, Fast and Simple Methods for the Optimization
    #     of Kurtosis Used as a Projection Pursuit Index, Analytica Chimica Acta,
    #     704 (2011) 1-15. (featured article)
    # (2) S. Hou, and P. D. Wentzell,Re-centered Kurtosis as a Projection Pursuit
    #     Index for Multivariate Data Analysis, Journal of Chemometrics, 28
    #     (2014) 370-384.   (Special issue article)
    #
    # Author:
    # S. Hou, University of Prince Edward Island, Charlottetown, PEI, Canada, 2014.

    # Mean-center the data
    n, m = X.shape
    Morig = mean_(X)
    X = X - ones((n, 1)) * Morig

    # Initial settings
    maxcount = 10000
    Vall = empty((1, guess), dtype=object)
    rall = empty((1, guess), dtype=object)
    kurtObj = zeros((1, guess))
    convFlag = empty((1, guess), dtype=object)

    # Loop
    for i in range(guess):
        count = 0
        V = random.randn(m, p)  # Random initial guess of V
        V = orthbasis(V)
        oldV1 = V.copy()
        R = mean_(X).T
        while True:
            count += 1

            # Update r
            Y = (X - ones((n, 1)) @ R.T / p) @ V  # Note p is in the denominator
            invPsi = linalg.inv(Y.T @ Y)
            gj = diag_(Y @ invPsi @ Y.T)
            Yj = Y @ invPsi @ (sum_(Y, axis=0)).T
            J = (
                2 * Y.T @ ((Yj @ ones((1, p))) * Y) @ invPsi - eye(p) * (sum_(gj) + 2)
            ) / p  # Jacobian matrix
            f = sum_(Y.T * (ones((p, 1)) @ gj.T), 1)
            R = R - V @ (ldiv(J, f))  # Newton' method

            # Update V
            # Calculate b1 and b2
            XX = X - ones((n, 1)) * R.T  # Note p is not in the denominator
            Z = XX @ V
            S = Z.T @ Z
            invS = linalg.inv(S)
            ai = diag_(Z @ invS @ Z.T)
            Z_ai = (ai @ ones((1, p))) * Z
            Si_ai = Z.T @ Z_ai

            b1 = ldiv(-J.T, (invS @ Si_ai @ invS @ (sum_(Z)).T))
            b2 = ldiv(-J.T, (invS @ (sum_(Z_ai)).T))

            # Calculate the 8 matrices
            Yj_b1_Yj = (Y @ b1 @ ones((1, p))) * Y
            Yj_b2_Yj = (Y @ b2 @ ones((1, p))) * Y
            Xj_gj = sum_((gj @ ones((1, m))) * X)

            M1 = X.T @ Z @ invS @ Si_ai
            M2 = -Xj_gj.T @ b1.T @ S
            M3 = (
                2 * X.T @ Y @ (invPsi @ Y.T @ Yj_b1_Yj @ invPsi @ S)
            )  # Parentheses added to speed up
            M4 = -2 * X.T @ Yj_b1_Yj @ invPsi @ S

            M5 = (X.T * (ones((m, 1)) @ ai.T)) @ XX  # Full rank
            M6 = -Xj_gj.T @ b2.T @ Z.T @ XX  # Not full rank
            M7 = (
                2 * X.T @ Y @ (invPsi @ Y.T @ Yj_b2_Yj @ invPsi @ Z.T @ XX)
            )  # Parentheses added to speed up
            M8 = -2 * X.T @ Yj_b2_Yj @ invPsi @ Z.T @ XX

            # Calculate new V
            V = ldiv((M5 + M6 + M7 + M8), (M1 + M2 + M3 + M4))
            V = orthbasis(V)

            # Test convergence
            L = abs(V) - abs(oldV1)
            L = trace(L.T @ L)
            if L < convlimit * p:
                convFlag[0, i] = "Converged"
                break
            elif count > maxcount:
                convFlag[0, i] = "Not converged"
                break
            oldV1 = V.copy()

        # Save the subspaces for different initial guesses. Note the basis of the
        # subspace has been changed in accordance with PCA (mean-centered) criterion.
        kurtObj[0, i] = n * sum_(diag_(Z @ linalg.inv(Z.T @ Z) @ Z.T) ** 2)
        Utem, Stem, Vtem = svd(X @ V, full_matrices=False)  # X has been mean-centered.
        Vtem = V @ Vtem
        Vall[0, i] = Vtem
        rall[0, i] = R.T @ Vtem @ Vtem.T  # r is saved as a row vector now.

    # Take the best projection vector as the solution
    tem = kurtObj.min()
    ind = where(kurtObj == tem)[1][0]
    V = Vall[0][ind]
    R = rall[0][ind]
    T = X @ V
    K = kurtObj[0][ind]

    # Add mean value
    T = (
        T + ones((n, 1)) @ Morig @ V
    )  # Adjust the scores (The scores of mean vector are added).
    R = R + Morig  # Adjust r (mean vector is added).

    return T, V, R, K, Vall, kurtObj, convFlag


def orthbasis(A):
    """Returns V"""
    # Calculate an orthonormal basis for matix A using Gram-Schimdt process
    # Reference: David Poole, Linear Algebra - A Modern Introduction,
    # Brooks/Cole, 2003. pp.376.
    #
    # Input:
    #   A: a matrix
    # Output:
    #   V: an orthonormal matrix

    c = A.shape[1]
    V = zeros(A.shape)
    V[:, 0] = A[:, 0] / linalg.norm(A[:, 0])
    for i in range(1, c):
        tem = A[:, i] - V @ V.T @ A[:, i]
        V[:, i] = tem / linalg.norm(tem)
    return V
