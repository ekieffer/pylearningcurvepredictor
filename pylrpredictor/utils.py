"""
Functions which are common and require SciPy Base and Level 1 SciPy
(special, linalg)
"""

from __future__ import division, print_function, absolute_import

from numpy import exp, log, asarray, arange, newaxis, hstack, product, array, \
                  zeros, eye, poly1d, r_, rollaxis, sum, fromstring



def logsumexp(a, axis=None, b=None):
    """Compute the log of the sum of exponentials of input elements.
    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis over which the sum is taken. By default `axis` is None,
        and all elements are summed.
        .. versionadded:: 0.11.0
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`.
        .. versionadded:: 0.12.0
    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.
    See Also
    --------
    numpy.logaddexp, numpy.logaddexp2
    Notes
    -----
    Numpy has a logaddexp function which is very similar to `logsumexp`, but
    only handles two arguments. `logaddexp.reduce` is similar to this
    function, but may be less stable.
    Examples
    --------
    >>> from scipy.misc import logsumexp
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107
    With weights
    >>> a = np.arange(10)
    >>> b = np.arange(10, 0, -1)
    >>> logsumexp(a, b=b)
    9.9170178533034665
    >>> np.log(np.sum(b*np.exp(a)))
    9.9170178533034647
    """
    a = asarray(a)
    if axis is None:
        a = a.ravel()
    else:
        a = rollaxis(a, axis)
    a_max = a.max(axis=0)
    if b is not None:
        b = asarray(b)
        if axis is None:
            b = b.ravel()
        else:
            b = rollaxis(b, axis)
        out = log(sum(b * exp(a - a_max), axis=0))
    else:
        out = log(sum(exp(a - a_max), axis=0))
    out += a_max
    return out
