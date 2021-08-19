"""
    Data object module
    ===================

    The :class:`.Data` is the backbone of the library that allows one to compute easily the different quantities needed to
    postprocess forecasts.

    Basically, it is a Numpy :class:`~numpy.ndarray` with the first 5 dimensions allocated to a
    dedicated and fixed meaning. These first axis of the data represent:

    * Axis 0th: predictor number (:math:`p`)
    * Axis 1st: observation/realization number (:math:`n`)
    * Axis 2nd: ensemble member number (:math:`m`)
    * Axis 3rd: variable or label number (:math:`v`) **[Not used/implemented for the moment!]**
    * Axis 4th: lead time (:math:`t`)

    These 5 first dimensions are called the *data index* (see :attr:`Data.index_shape`).
    As such, they represent the data as a multi-dimensional array :math:`\\mathcal{D}_{p,n,m,v} (t)` where :math:`t`
    is the lead time.

    The extra dimensions possibly trailing in the array are the intrinsic dimensions of the data itself.
    For instance, an array of total dimension 7 represents 2D data (e.g. fields).
    If only 5 dimensions are present on total, then the data is a scalar.
    The main operations of the :class:`.Data` are broadcasted over these extra-dimension, making the data object directly compliant
    with multi-dimensional forecast data.

    Examples
    --------

    Here is an example showing how the :class:`Data` object works:

    >>> import numpy as np
    >>> from core.data import Data
    >>> a = np.random.randn(2, 3, 10, 1, 60, 20, 20)
    >>> data = Data(a)
    >>> data.number_of_predictors
    2
    >>> data.number_of_observations
    3
    >>> data.number_of_members
    10
    >>> data.number_of_variables
    1
    >>> data.number_of_time_steps
    60
    >>> data.shape
    (20, 20)
    >>> data.index_shape
    (2, 3, 10, 1, 60)

    Notes
    -----

    * The methods of the :class:`Data` object return as much as possible another :class:`Data` object. If it is not possible
      to format the ouptut according to the shape described above, a :class:`~numpy.ndarray` is returned. For example, matrix
      derived from the data are returned as NumPy arrays.
    * By convention, if a method or operation reduces or returns a :class:`Data` with one of the index missing, the corresponding index of
      the object is set to zero to preserve index shape of the object. For example, for a :class:`Data` object
      :math:`\\mathcal{D}_{p,n,m,v} (t)` of :attr:`~Data.index_shape` `(P, N, M, V, T)`, the :attr:`Data.ensemble_max`
      method returns :math:`\\max_m \\mathcal{D}_{p,n,m,v} (t)` as a :class:`Data` object of shape `(P, N, 1, V, T)`.
      E.g.:

      >>> import numpy as np
      >>> from core.data import Data
      >>> a = np.random.randn(2, 3, 10, 1, 60, 20, 20)
      >>> data = Data(a)
      >>> data.index_shape
      (2, 3, 10, 1, 60)
      >>> maxi = data.ensemble_max
      >>> maxi.index_shape
      (2, 3, 1, 1, 60)

      In the following, in such a case we will use the notation :math:`\\mathcal{D}_{p,n,v} (t) \\equiv \\mathcal{D}_{p,n,0,v} (t)`.
    * Missing values in :class:`Data` objects can be marked as :data:`numpy.nan`. The various averages, summation and methods will automatically **ignore** the missing values.
      As a consequence, it means that these averages and summations will include less terms. For example, if at one lead time, an ensemble member value is missing,
      the ensemble mean is done on the rest of the ensemble at this precise lead time and obviously does not include this member.

    References
    ----------

    .. bibliography:: ../ref.bib
        :labelprefix: DATA-
        :keyprefix: data-

    Warnings
    --------

    Several properties and definitions inside the Data object are not yet fixed or well-defined. Usages and standards might still
    evolve.


    .. _quantiles: https://en.wikipedia.org/wiki/Quantile

"""
# TODO : - Timestamps as a _total_shape array of np.datetime64 ?
#        - Metadata management
#        - data consistency check


import numpy as np
from scipy.stats import norm
import pickle
import warnings
import datetime

try:
    import matplotlib.pyplot as plt
    from core.plotting import std_plot, minmax_plot
except ModuleNotFoundError:
    warnings.warn('Unable to import matplotlib, plotting methods will not work.', ImportWarning)
    plt = None

try:
    import pandas as pd
except ModuleNotFoundError:
    warnings.warn('Unable to import pandas, loading pandas DataFrame will not work.', ImportWarning)
    pd = None


class Data(object):
    """Main data structure of the library.

    Parameters
    ----------
    data: None or ~numpy.ndarray
        The data array. If not `None`, should be an array of shape

         (:attr:`~.Data.number_of_predictors`, :attr:`~.Data.number_of_observations`, :attr:`~.Data.number_of_members`, :attr:`~.Data.number_of_variables`,  :attr:`~.Data.number_of_time_steps`)

        Default to `None`.
    timestamps: None or ~numpy.ndarray(~datetime.datetime) or list(~numpy.ndarray(~datetime.datetime))
        The timestamps of the forecast data.
        Can be a 1D :class:`~numpy.ndarray` of :class:`~datetime.datetime` timestamps (one per lead time). In that case, the same timestamps vector is
        attributed to all the predictors and the observations provided by `data`.
        Can also be a list of 1D :class:`~numpy.ndarray` of :class:`~datetime.datetime` timestamps (one list entry per observation). It allows one to set a different timestamps per
        observation/realization.
        If None, no timestamp is set. Default to `None`.
    metadata: object or ~numpy.ndarray(object) or list(~numpy.ndarray(object))
        Object(s) describing the metadata of the data (not implemented yet). Can be an object, a :class:`~numpy.ndarray` of objects
        (one per observation/realization), or a list of 1D :class:`~numpy.ndarray` of objects (one list entry per predictor, one array component
        per observation/realization). If a single array is provided, it can be of shape (:attr:`~.Data.number_of_predictors`, :attr:`~.Data.number_of_observations`) to
        specify the metadata of each predictor and observation/realization separately. It can also be a 1D array for which each component corresponds
        to an observation/realization. In this case, the same metadata object is used for each predictor. Default to the `None` object.
    dtype: ~numpy.dtype
        The data type of the data being stored. Default to :class:`numpy.float64`.

    Attributes
    ----------

    data: ~numpy.ndarray
        The data array.
    timestamps: (~numpy.ndarray(~numpy.ndarray(~datetime.datetime))
        The timestamps of the data, stored as :class:`~numpy.ndarray` of :class:`~datetime.datetime` and with shape corresponding to (:attr:`~.Data.number_of_predictors`, :attr:`~.Data.number_of_observations`).
    metadata: ~numpy.ndarray(object)
        Object describing the metadata of the data (not specified yet).
    """

    # First, some builtin standards

    _axis_name_dict = {"predictor":   0,
                       "observation": 1,
                       "obs":         1,
                       "realization": 1,
                       "member":      2,
                       "ensemble":    2,
                       "label":       3,
                       "variable":    3,
                       "time":        4,
                       "lead":        4}

    def __init__(self, data=None, metadata=None, timestamps=None, dtype=np.float64):

        if data is not None:
            self.data = data.astype(dtype)
            self._dtype = dtype
            self.metadata = self._create_metadata(data, metadata)
            self.timestamps = self._create_timestamps(data, data, timestamps)
        else:
            self.data = data
            self._dtype = dtype
            self.metadata = metadata
            self.timestamps = timestamps

    def load_timestamps(self, timestamps):
        """Load timestamps data.

        Parameters
        ----------
        timestamps: ~numpy.ndarray(~datetime.datetime) or list(~numpy.ndarray(~datetime.datetime))
            The timestamps of the forecast data.
            Can be a 1D :class:`~numpy.ndarray` of :class:`~datetime.datetime` timestamps (one per lead time). In that case, the same timestamps vector is
            attributed to all the predictors and the observations provided by `data`.
            Can also be a list of 1D :class:`~numpy.ndarray` of :class:`~datetime.datetime` timestamps (one list entry per observation). It allows one to set a different timestamps per
            observation/realization.
        """
        self.timestamps = self._create_timestamps(self.data, self.data, timestamps)

    def clear_data(self):
        """Reset the Data object."""
        self.data = None
        self.metadata = None
        self.timestamps = None

    def set_dtype(self, dtype):
        """Set the data type.

        Parameters
        ----------
        dtype: ~numpy.dtype
            The Numpy data type of the data.
        """
        self.data = self.data.astype(dtype)
        self._dtype = dtype

    @property
    def dtype(self):
        """~numpy.dtype: The data type."""
        return self._dtype

    @property
    def number_of_observations(self):
        """int: The number of observations stored in the data object."""
        if self.data is None:
            return None
        else:
            return self.index_shape[1]

    @property
    def number_of_variables(self):
        """int: The number of variables stored in the data object."""
        if self.data is None:
            return None
        else:
            return self.index_shape[3]

    @property
    def number_of_time_steps(self):
        """int: The number of time steps stored in the data object."""
        if self.data is None:
            return None
        else:
            return self.index_shape[4]

    @property
    def number_of_predictors(self):
        """int: The number of predictors stored in the data object."""
        if self.data is None:
            return None
        else:
            return self.index_shape[0]

    @property
    def number_of_members(self):
        """int: The number of ensemble members stored in the data object."""
        if self.data is None:
            return None
        else:
            return self.index_shape[2]

    @property
    def ndim(self):
        """int: The dimension of the data."""
        if self.data is None:
            return None
        else:
            return self.data.ndim-5

    @property
    def index_shape(self):
        """tuple(int): The shape of the data index."""
        if self.data is None:
            return None
        else:
            return self.data.shape[:5]

    @property
    def _total_shape(self):
        """tuple(int): The total shape of the data array."""
        if self.data is None:
            return None
        else:
            return self.data.shape

    @property
    def shape(self):
        """tuple(int): The shape of the data."""
        if self.data is None:
            return None
        else:
            return self.data.shape[5:]

    @property
    def _scalars(self):
        if self.data is None:
            return False
        else:
            return self.ndim == 0

    @property
    def _vectors(self):
        if self.data is None:
            return False
        else:
            return self.ndim == 1

    @property
    def _fields(self):
        if self.data is None:
            return False
        else:
            return self.ndim == 2

    def is_scalar(self):
        """bool: Return true if the data stored are scalars."""
        return self._scalars

    def is_vector(self):
        """bool: Return true if the data stored are vectors."""
        return self._vectors

    def is_field(self):
        """bool: Return true if the data stored are fields."""
        return self._fields

    def is_empty(self):
        """bool: Return true if there is no data stored."""
        return self.data is None

    def __repr__(self):
        return self.data.__repr__()

    def __str(self):
        return self.data.__str()

    def __getitem__(self, index):
        return self.data[index[:5]]

    def __setitem__(self, index, value):
        self.data[index[:5]] = value

    def __add__(self, other):

        try:
            res = self.data + other.data
        except:
            try:
                res = self.data + other
            except:
                res = None

        return Data(res, self.metadata, self.timestamps, self.dtype)

    def __radd__(self, other):
        self.__add__(other)

    def __sub__(self, other):

        try:
            res = self.data - other.data
        except:
            try:
                res = self.data - other
            except:
                res = None

        return Data(res, self.metadata, self.timestamps, self.dtype)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        try:
            res = self.data * other.data
        except:
            try:
                res = self.data * other
            except:
                res = None

        return Data(res, self.metadata, self.timestamps, self.dtype)

    def __truediv__(self, other):
        try:
            res = self.data / other.data
        except:
            try:
                res = self.data / other
            except:
                res = None

        return Data(res, self.metadata, self.timestamps, self.dtype)

    def __pow__(self, power, modulo=None):
        res = self.data.__pow__(power, modulo)
        return Data(res, self.metadata, self.timestamps, self.dtype)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __abs__(self):
        return Data(np.abs(self.data), self.metadata, self.timestamps, self.dtype)

    def get_value(self, index):
        """Get the value(s) of a particular data index.

        Parameters
        ----------
        index: tuple(int)
            The data index of the value.

        Returns
        -------
        values: ~numpy.ndarray
            The values corresponding to the index.
        """
        return self.data[index]

    def get_data(self):
        """Return the whole data array.

        Returns
        -------
        ~numpy.ndarray
            The whole data array.
        """
        return self.data

    def get_metadata(self):
        """Return the meta data."""
        return self.metadata

    def copy(self):
        """Return a (shallow) copy of the Data object.

        Returns
        -------
        Data
            A copy of the Data object.
        """
        return Data(self.data.copy(), self.metadata, self.timestamps, self.dtype)

    @property
    def observational_mean(self):
        """Data: Mean over the observation index :math:`n`:
        :math:`\\mu^{\\rm  obs}_{p,m,v} (t) = \langle \mathcal{D}_{p,n,m,v} (t) \\rangle_n`."""
        return Data(np.nanmean(self.data, axis=1)[:, np.newaxis, ...], self.metadata, None, self.dtype)

    @property
    def observational_median(self):
        """Data: Median over the observation index :math:`n`."""
        return Data(np.nanmedian(self.data, axis=1)[:, np.newaxis, ...], self.metadata, None, self.dtype)

    @property
    def observational_var(self):
        """Data: Variance over the observation index :math:`n`:
        :math:`\\sigma^{\\rm obs}_{p,m,v} (t)^2 = \\left\\langle (\\mathcal{D}_{p,n,m,v} (t) - \\mu^{\\rm obs}_{p,m,v} (t))^2 \\right\\rangle_n`."""
        return Data(np.nanvar(self.data, axis=1, ddof=1)[:, np.newaxis, ...], self.metadata, None, self.dtype)

    @property
    def observational_std(self):
        """Data: Standard deviation over the observation index :math:`n`:
        :math:`\sigma^{\\rm obs}_{p,m,v} (t)`."""
        return Data(np.nanstd(self.data, axis=1, ddof=1)[:, np.newaxis, ...], self.metadata, None, self.dtype)

    @property
    def observational_min(self):
        """Data: Observational minimum over the ensemble index :math:`n`:
        :math:`\min_n \mathcal{D}_{p,n,m,v} (t)`."""
        return Data(np.nanmin(self.data, axis=1)[:, :, np.newaxis, ...], self.metadata, None, self.dtype)

    @property
    def observational_max(self):
        """Data: Observational maximum over the ensemble index :math:`n`:
        :math:`\max_n \mathcal{D}_{p,n,m,v} (t)`."""
        return Data(np.nanmax(self.data, axis=1)[:, :, np.newaxis, ...], self.metadata, None, self.dtype)

    def observational_quantiles(self, q, interpolation='linear'):
        """Return the observational `quantiles`_ of the data.

        Parameters
        ----------
        q: array_like(float)
            Quantile or sequence of quantiles to compute, which must be between 0 and 1 inclusive.
        interpolation: str, optional
            This optional parameter specifies the interpolation method to use when the desired quantile lies between
             two data points. See :func:`numpy.quantile` for more information.

        Returns
        -------
        Data
            The observational quantiles, stored along the observation axis (1st axis).
        """
        return Data(np.moveaxis(np.nanquantile(self.data, q, axis=1, interpolation=interpolation), 0, 1),
                    self.metadata, None, self.dtype)

    @property
    def ensemble_mean(self):
        """Data: Ensemble mean. Mean over the ensemble index :math:`m`:
        :math:`\\mu^{\\rm{ens}}_{p,n,v} (t) = \\langle \\mathcal{D}_{p,n,m,v} (t) \\rangle_m`."""
        return Data(np.nanmean(self.data, axis=2)[:, :, np.newaxis, ...], self.metadata, self.timestamps, self.dtype)

    @property
    def ensemble_median(self):
        """Data: Ensemble median. Median over the ensemble index :math:`m`."""
        return Data(np.nanmedian(self.data, axis=2)[:, :, np.newaxis, ...], self.metadata, self.timestamps, self.dtype)

    @property
    def ensemble_var(self):
        """Data: Ensemble variance over the ensemble index :math:`m`:
        :math:`\\sigma^{\\rm{ens}}_{p,n,v} (t)^2 = \\left\\langle \\left( \\mathcal{D}_{p,n,m,v} (t) - \\mu^{\\rm{ens}}_{p,n,v} (t) \\right)^2 \\right\\rangle_m`."""
        return Data(np.nanvar(self.data, axis=2, ddof=1)[:, :, np.newaxis, ...], self.metadata, self.timestamps, self.dtype)

    @property
    def ensemble_std(self):
        """Data: Ensemble standard deviation over the ensemble index :math:`m`:
        :math:`\\sigma^{\\rm{ens}}_{p,n,v} (t)`."""
        return Data(np.nanstd(self.data, axis=2, ddof=1)[:, :, np.newaxis, ...], self.metadata, self.timestamps, self.dtype)

    @property
    def ensemble_min(self):
        """Data: Ensemble minimum over the ensemble index :math:`m`:
        :math:`\\min_m \\mathcal{D}_{p,n,m,v} (t)`."""
        return Data(np.nanmin(self.data, axis=2)[:, :, np.newaxis, ...], self.metadata, self.timestamps, self.dtype)

    @property
    def ensemble_max(self):
        """Data: Ensemble maximum over the ensemble index :math:`m`:
        :math:`\\max_m \\mathcal{D}_{p,n,m,v} (t)`."""
        return Data(np.nanmax(self.data, axis=2)[:, :, np.newaxis, ...], self.metadata, self.timestamps, self.dtype)

    def ensemble_quantiles(self, q, interpolation='linear'):
        """Return the ensemble `quantiles`_ of the data.


        Parameters
        ----------
        q: array_like(float)
            Quantile or sequence of quantiles to compute, which must be between 0 and 1 inclusive.
        interpolation: str, optional
            This optional parameter specifies the interpolation method to use when the desired quantile lies between
            two data points. See :func:`numpy.quantile` for more information.

        Returns
        -------
        Data
            The ensemble quantiles, stored along the ensemble member number axis (1st axis).
        """
        return Data(np.moveaxis(np.nanquantile(self.data, q, axis=2, interpolation=interpolation), 0, 2),
                    self.metadata, self.timestamps, self.dtype)

    @property
    def ensemble_members_distance(self):
        """~numpy.ndarray: Distance between ensemble members:
        :math:`d^{\\rm{MBM}}_{p,n,m_1,m_2,v} (t) = |\mathcal{D}_{p,n,m_1,v} (t)- \mathcal{D}_{p,n,m_2,v} (t) |`"""
        return np.abs(self.data[:, :, :, np.newaxis, ...] - self.data[:, :, np.newaxis, :, ...])

    @property
    def observational_distance(self):
        """~numpy.ndarray: Distance between observations:
        :math:`d^{\\rm{obs}}_{p,n_1,n_2,m,v} (t) = |\\mathcal{D}_{p,n_1,m,v} (t)- \\mathcal{D}_{p,n_2,m,v} (t)|`"""
        return np.abs(self.data[:, :, np.newaxis, ...] - self.data[:, np.newaxis, :, ...])

    @property
    def delta(self):
        """Data: Average over the ensemble members of the :attr:`ensemble_members_distance`:
        :math:`\\delta_{p,n,v} (t) = \\left\\langle d^{\\rm{MBM}}_{p,n,m_1,m_2,v} (t) \\right\\rangle_{m_1, m_2}`
        """
        return Data(np.nanmean(np.nanmean(self.ensemble_members_distance, axis=2), axis=2)[:, :, np.newaxis, ...], self.metadata, self.timestamps, self.dtype)

    @property
    def uncertainty(self):
        """Data: Average over the observations of the :attr:`observational_distance` divided by 2:
        :math:`\\langle d^{\\rm{obs}}_{p,n_1,n_2,m,v} (t)\\rangle_{n_1,n_2} / 2`.
        Sometimes called the uncertainty contribution of the CRPS.
        See :cite:`data-hersbach2000decomposition` for more details."""
        return Data(np.nanmean(np.nanmean(self.observational_distance, axis=1), axis=1)[:, np.newaxis, ...] / 2, self.metadata, self.timestamps, self.dtype)

    @property
    def centered_ensemble(self):
        """Data: Returns an ensemble centered on its :attr:`ensemble_mean`:
        :math:`\\bar{\\mathcal{D}}^{\\rm ens}_{p,n,m,v} (t) = \\mathcal{D}_{p,n,m,v} (t) - \\mu^{\\rm ens}_{p,n,v} (t)`."""
        return self - self.ensemble_mean

    @property
    def centered_observation(self):
        """Data: Returns an ensemble centered on its :attr:`observational_mean`:
        :math:`\\bar{\\mathcal{D}}^{\\rm obs}_{p,n,m,v} (t) = \\mathcal{D}_{p,n,m,v} (t) - \\mu^{\\rm obs}_{p,n,v} (t)`."""
        return self - self.observational_mean

    def ensemble_distance(self, other):
        """Data: Averaged distance between ensemble member and another Data object:
        :math:`d^{\\rm{ens}}_{p,n,v} [\\mathcal{O}] (t) = \\langle|\\mathcal{D}_{p,n,m,v} (t)- \\mathcal{O}_{p,n,m,v} (t)|\\rangle_m` where :math:`\\mathcal{O}` is the
        other Data object."""
        return abs(self - other).ensemble_mean

    def _CRPS_ab(self, other):

        data = np.sort(self.data, axis=2)
        d = np.diff(data, axis=2)
        a = np.where(data[:, :, 1:, ...] < other.data, d, 0)
        b = np.where(data[:, :, :-1, ...] > other.data, d, 0)
        a = np.where(np.logical_and(data[:, :, :-1, ...] < other.data, other.data < data[:, :, 1:, ...]),
                     other.data - data[:, :, :-1, ...], a)
        b = np.where(np.logical_and(data[:, :, :-1, ...] < other.data, other.data < data[:, :, 1:, ...]),
                     data[:, :, 1:, ...] - other.data, b)

        outb = np.where(other.data < self.ensemble_min.data, self.ensemble_min.data - other.data, 0)
        outa = np.where(other.data > self.ensemble_max.data, other.data - self.ensemble_max.data, 0)

        a = np.concatenate((a, outa), axis=2)
        b = np.concatenate((outb, b), axis=2)

        return Data(a, self.metadata, self.timestamps, self.dtype), Data(b, self.metadata, self.timestamps, self.dtype)

    def _CRPS_go(self, other):

        a, b = self._CRPS_ab(other)

        aa = a.observational_mean
        bb = b.observational_mean

        g = aa + bb
        o = bb / g

        o.data[:, 0, 0, ...] = np.nanmean(np.where(self.ensemble_min.data - other.data > 0, 1, 0), axis=(1, 2))
        o.data[:, 0, -1, ...] = np.nanmean(np.where(self.ensemble_max.data - other.data > 0, 1, 0), axis=(1, 2))

        g.data[:, 0, 0, ...] = bb.data[:, 0, 0, ...] / o.data[:, 0, 0, ...]
        g.data[:, 0, -1, ...] = aa[:, 0, -1, ...] / (1 - o.data[:, 0, -1, ...])

        return g, o

    def CRPS_relipot(self, other):
        """Return the decomposition of CRPS scores with another Data object :math:`\\mathcal{O}` (typically containing
        observations) according to the fomula:

        :math:`{\\rm CRPS}_{p,v} (t) = {\\rm Reli}_{p,v} (t) + {\\rm CRPS}^{\\rm pot}_{p,v} (t)`

        where :math:`{\\rm Reli}_{p,v} (t)` and :math:`{\\rm CRPS}^{\\rm pot}_{p,v} (t)` are
        respectively the reliability and potential CRPS, i.e. the CRPS one would obtain with a perfectly reliable ensemble.
        See :cite:`data-hersbach2000decomposition`, pp. 564 for more details.

        Parameters
        ----------
        other: Data
            Another Data object with observations in it.

        Returns
        -------
        tuple(Data)
            The decomposition of the CRPS score into the reliability and the potential CRPS.
        """

        p = np.arange(1, self.number_of_members+1) / self.number_of_members
        g, o = self._CRPS_go(other)

        reli = np.moveaxis(np.moveaxis(g.data, 2, -1) * (np.moveaxis(o.data, 2, -1) - p)**2, -1, 2)
        pot = g.data * o.data * (1 - o.data)

        return Data((np.nansum(reli, axis=2))[:, :, np.newaxis, ...], self.metadata, self.timestamps, self.dtype), \
               Data((np.nansum(pot, axis=2))[:, :, np.newaxis, ...], self.metadata, self.timestamps, self.dtype)

    def CRPS_decomposition(self, other):
        """Return the decomposition of CRPS scores with another Data object :math:`\\mathcal{O}` (typically containing
        observations) according to the fomula:

        :math:`{\\rm CRPS}_{p,v} (t) = {\\rm Reli}_{p,v} (t) - {\\rm Resol}_{p,v} (t) + {\\rm Unc}_{p,v} (t)`

        where :math:`{\\rm Reli}_{p,v} (t)`, :math:`{\\rm Resol}_{p,v} (t)` and :math:`{\\rm Unc}_{p,v} (t)` are
        respectively the reliability, the resolution and the uncertainty contribution to the CRPS.
        See :cite:`data-hersbach2000decomposition`, pp. 565 for more details.

        Parameters
        ----------
        other: Data
            Another Data object with observations in it.

        Returns
        -------
        tuple(Data)
            The decomposition of the CRPS score into the reliability, the resolution and the uncertainty contribution.
        """

        reli, pot = self.CRPS_relipot(other)

        uncert = other.uncertainty

        resol = uncert - pot

        return reli, resol, uncert

    def CRPS(self, other):
        """Return the CRPS scores with another Data object :math:`\\mathcal{O}` (typically containing
        observations). This score is computed according to :cite:`data-hersbach2000decomposition` (see pp. 563-564).

        Parameters
        ----------
        other: Data
            Another Data object with observations in it.

        Returns
        -------
        Data
            The CRPS score.
        """

        p = np.arange(1, self.number_of_members+1) / self.number_of_members
        a, b = self._CRPS_ab(other)

        c = np.moveaxis(np.moveaxis(a.data, 2, -1) * p**2 + np.moveaxis(b.data, 2, -1) * (1 - p)**2, -1, 2)

        return Data((c.sum(axis=2))[:, :, np.newaxis, ...], self.metadata, self.timestamps, self.dtype).observational_mean

    def Abs_CRPS(self, other):
        """Return the Absolute norm CRPS scores with another Data object :math:`\\mathcal{O}` (typically containing
        observations). This score is computed with the analytical formula:

        :math:`{\\rm CRPS}^{\\rm Abs}_{p,v} (t) = \\left\\langle d^{\\rm ens}_{p,n,v} [\\mathcal{O}] (t) - \\delta_{p,n,v} (t) /2 \\right\\rangle_n`

        where :math:`d^{\\rm{ens}}_{p,n,v} [\\mathcal{O}]` is the :meth:`~.Data.ensemble_distance` with the observations object :math:`\\mathcal{O}`,
        and :math:`\\delta_{p,n,v}` is :attr:`delta`, obtained by taking the average of the :attr:`ensemble_members_distance` over the ensemble members.
        See :cite:`data-van2015ensemble` and :cite:`data-gneiting2007strictly` for more details.

        Parameters
        ----------
        other: Data
            Another Data object with observations in it.

        Returns
        -------
        Data
            The Absolute norm CRPS score.
        """
        return (self.ensemble_distance(other) - 0.5 * self.delta).observational_mean

    def Ngr_CRPS(self, other):
        """Return the Non-homogeneous Gaussian Regression (NGR) CRPS scores with another Data object :math:`\\mathcal{O}`
        (typically containing observations). This score is computed with the analytical formula:

        :math:`{\\rm CRPS}^{\\rm Ngr}_{p,v} (t) = \\left\\langle\\sigma^{\\rm ens}_{p,n,v} (t) \\left(z_{p,n,v}(t)(2\\Phi(z_{p,n,v}(t)) -1) + 2\\phi(z_{p,n,v}(t)) - \\pi^{-1/2}\\right)\\right\\rangle_n`

        where :math:`\\phi` is the normal distribution, :math:`\\Phi` is its cumulative distribution function and
        :math:`z_{p,n,v} = \\left(\\mathcal{O}_{p,n,v} (t) -\\mu^{\\rm{ens}}_{p,n,v} (t)\\right)/\\sigma^{\\rm{ens}}_{p,n,v} (t)` is the
        standardized error with respect to the `other` data (where :math:`\\mu^{\\rm{ens}}` and :math:`\\sigma^{\\rm{ens}}` are respectively the :attr:`ensemble_mean` and the :attr:`ensemble_std`).
        See :cite:`data-van2015ensemble` and :cite:`data-gneiting2005calibrated` for more details.

        Parameters
        ----------
        other: Data
            Another Data object with observations in it. Must have the same shape as the Data object, except along the members :math:`m` dimensions (2nd axis).

        Returns
        -------
        Data
            The NGR CRPS score.
        """
        z = (other - self).ensemble_mean.data / self.ensemble_std.data
        return (self.ensemble_std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - np.pi**(-0.5))).observational_mean

    def bias(self, other):
        """Return the bias :math:`\\left\\langle\\mu^{\\rm{ens}}_{p,n,v} (t) - \\mathcal{O}_{p,n,v} (t)\\right\\rangle_n`
        with another Data object :math:`\\mathcal{O}` (typically containing observations).

        Parameters
        ----------
        other: Data
            Another Data object with observations in it. Must have the same shape as the Data object, except along the members :math:`m` dimensions (2nd axis).

        Returns
        -------
        Data
            The bias.
        """
        return (self - other).ensemble_mean.observational_mean

    def ensemble_mean_MSE(self, other):
        """Return the Mean Square Error of the ensemble mean :math:`\\left\\langle\\left(\\mu^{\\rm{ens}}_{p,n,v} (t) - \\mathcal{O}_{p,n,v} (t)\\right)^2\\right\\rangle_n`
        with another Data object :math:`\\mathcal{O}` (typically containing observations).

        Parameters
        ----------
        other: Data
            Another Data object with observations in it. Must have the same shape as the Data object, except along the members :math:`m` dimensions (2nd axis).

        Returns
        -------
        Data
            The ensemble mean Mean Square Error.
        """
        return ((self.ensemble_mean - other)**2).observational_mean

    def ensemble_mean_RMSE(self, other):
        """Return the Root Mean Square Error of the ensemble mean :math:`\\sqrt{\\left\\langle\\left(\\mu^{\\rm{ens}}_{p,n,v} (t) - \\mathcal{O}_{p,n,v} (t)\\right)^2\\right\\rangle_n}`
        with another Data object :math:`\\mathcal{O}` (typically containing observations).

        Parameters
        ----------
        other: Data
            Another Data object with observations in it. Must have the same shape as the Data object, except along the members :math:`m` dimensions (2nd axis).

        Returns
        -------
        Data
            The ensemble mean Root Mean Square Error.
        """
        return self.ensemble_mean_MSE(other)**0.5

    def plot(self, predictor=0, variable=0, ax=None, timestamps=None, global_label=None, grid_point=None, **kwargs):
        """Plot the data as a function of time.

        Parameters
        ----------
        predictor: int, optional
            The predictor index to use. Default is 0.
        variable: int, optional
            The variable index to use.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
        timestamps: None or ~numpy.ndarray(~datetime.datetime), optional
            An array containing the timestamp of the data. If None, try to use the data `timestamps` and in last resort a numbered time index.
            Default to `None`.
        global_label: None or str or list(str), optional
            Label to represent all the data (str), or all the data of one observation (list of str) in the legend.
        grid_point: tuple(int, int), optional
            If the data are fields, specifies which grid point to plot.
        kwargs: dict
            Argument to be passed to the plotting routine.

        Returns
        -------
        ax: ~matplotlib.axes.Axes
            An axes where the data were plotted.

        """

        if plt is None:
            warnings.warn('Matplotlib not loaded, cannot plot !', ImportWarning)
            return None

        p = predictor
        v = variable
        if grid_point is None:
            ni = None
            nj = None
        else:
            ni = grid_point[0]
            nj = grid_point[1]

        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        if timestamps is None:
            timestamps = self._get_timestamps(predictor)
        elif not isinstance(timestamps, list):
            timestamps = [timestamps] * self.number_of_observations

        opo = False
        lab = False

        if isinstance(global_label, list):
            opo = True
            lab = True
        elif isinstance(global_label, str):
            opo = False
            lab = True

        if self._scalars:
            first = True
            for i in range(self.number_of_observations):
                if opo:
                    first = True
                for j in range(self.number_of_members):
                    if lab and first:
                        if opo:
                            ax.plot(timestamps[i], self.data[p, i, j, v, :], label=global_label[i], **kwargs)
                        else:
                            ax.plot(timestamps[i], self.data[p, i, j, v, :], label=global_label, **kwargs)
                        first = False
                    else:
                        ax.plot(timestamps[i], self.data[p, i, j, v, :], **kwargs)
        elif self._fields:
            if ni is None or nj is None:
                warnings.warn('You must specify which grid point to plot, cannot plot !', UserWarning)
                return None
            first = True
            for i in range(self.number_of_observations):
                if opo:
                    first = True
                for j in range(self.number_of_members):
                    if lab and first:
                        if opo:
                            ax.plot(timestamps[i], self.data[p, i, j, v, :, ni, nj], label=global_label[i], **kwargs)
                        else:
                            ax.plot(timestamps[i], self.data[p, i, j, v, :, ni, nj], label=global_label, **kwargs)
                        first = False
                    else:
                        ax.plot(timestamps[i], self.data[p, i, j, v, :, ni, nj], **kwargs)

        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=30, horizontalalignment='center')
        return ax

    def plot_ensemble_mean(self, predictor=0, variable=0, ax=None, timestamps=None, grid_point=None, **kwargs):
        """Plot the data ensemble mean :attr:`ensemble_mean` as a function of time.

        Parameters
        ----------
        predictor: int, optional
            The predictor index to use. Default is 0.
        variable: int, optional
            The variable index to use.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
        timestamps: None or ~numpy.ndarray(~datetime.datetime), optional
            An array containing the timestamp of the data. If None, try to use the data `timestamps` and in last resort a numbered time index.
            Default to `None`.
        grid_point: tuple(int, int)
            If the data are fields, specifies which grid point to plot.
        kwargs: dict
            Argument to be passed to the plotting routine.

        Returns
        -------
        ax: ~matplotlib.axes.Axes
            An axes where the data were plotted.

        """

        em = self.ensemble_mean
        ax = em.plot(predictor, variable, ax, timestamps, grid_point=grid_point, **kwargs)
        return ax

    def plot_ensemble_median(self, predictor=0, variable=0, ax=None, timestamps=None, grid_point=None, **kwargs):
        """Plot the data ensemble median :attr:`ensemble_median` as a function of time.

        Parameters
        ----------
        predictor: int, optional
            The predictor index to use. Default is 0.
        variable: int, optional
            The variable index to use.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
        timestamps: None or ~numpy.ndarray(~datetime.datetime), optional
            An array containing the timestamp of the data. If None, try to use the data `timestamps` and in last resort a numbered time index.
            Default to `None`.
        grid_point: tuple(int, int)
            If the data are fields, specifies which grid point to plot.
        kwargs: dict
            Argument to be passed to the plotting routine.

        Returns
        -------
        ax: ~matplotlib.axes.Axes
            An axes where the data were plotted.

        """

        em = self.ensemble_median
        ax = em.plot(predictor, variable, ax, timestamps, grid_point=grid_point, **kwargs)
        return ax

    def plot_ensemble_std(self, predictor=0, variable=0, ax=None, timestamps=None, grid_point=None, **kwargs):
        """Plot the data ensemble standard deviation :attr:`ensemble_std` as a function of time.

        Parameters
        ----------
        predictor: int, optional
            The predictor index to use. Default is 0.
        variable: int, optional
            The variable index to use.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
        timestamps: None or ~numpy.ndarray(~datetime.datetime), optional
            An array containing the timestamp of the data. If None, try to use the data `timestamps` and in last resort a numbered time index.
            Default to `None`.
        grid_point: tuple(int, int)
            If the data are fields, specifies which grid point to plot.
        kwargs: dict
            Argument to be passed to the plotting routine.

        Returns
        -------
        ax: ~matplotlib.axes.Axes
            An axes where the data were plotted.

        """

        if plt is None:
            warnings.warn('Matplotlib not loaded, cannot plot !', ImportWarning)
            return None

        if grid_point is None:
            ni = None
            nj = None
        else:
            ni = grid_point[0]
            nj = grid_point[1]

        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        if timestamps is None:
            timestamps = self._get_timestamps(predictor)
        elif not isinstance(timestamps, list):
            timestamps = [timestamps] * self.number_of_observations

        em = self.ensemble_mean
        std = np.sqrt(self.ensemble_var.data)

        if self._scalars:
            for i in range(self.number_of_observations):
                std_plot(timestamps[i], em[predictor, i, 0, variable, :], std[predictor, i, 0, variable, :], ax=ax, **kwargs)
        elif self._fields:
            if ni is None or nj is None:
                warnings.warn('You must specify which grid point to plot, cannot plot !', UserWarning)
                return None
            for i in range(self.number_of_observations):
                std_plot(timestamps[i], em[predictor, i, 0, variable, :, ni, nj], std[predictor, i, 0, variable, :, ni, nj], ax=ax, **kwargs)

        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=30, horizontalalignment='center')
        return ax

    def plot_ensemble_quantiles(self, q, low_interpolation='linear', high_interpolation='linear',
                                predictor=0, variable=0, ax=None, timestamps=None, grid_point=None, alpha=0.1, **kwargs):
        """Plot the data ensemble quantiles :attr:`ensemble_quantiles` as a function of time.

        Parameters
        ----------
        q: array_like(float)
            Quantile or sequence of quantiles to compute, which must be between 0 and 0.5 exclusive.
            A symmetric quantile with respect to 0.5 will also be computed.
        low_interpolation: str, optional
            This optional parameter specifies the interpolation method to use when the desired lower quantile (q<0.5)
            lies between two data points. See :func:`numpy.quantile` for more information.
        high_interpolation: str, optional
            This optional parameter specifies the interpolation method to use when the desired higher quantile (q>0.5)
            lies between two data points. See :func:`numpy.quantile` for more information.
        predictor: int, optional
            The predictor index to use. Default is 0.
        variable: int, optional
            The variable index to use.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
        timestamps: None or ~numpy.ndarray(~datetime.datetime), optional
            An array containing the timestamp of the data. If None, try to use the data `timestamps` and in last resort a numbered time index.
            Default to `None`.
        grid_point: tuple(int, int)
            If the data are fields, specifies which grid point to plot.
        alpha: float
            Base level of transparency for the highest and lowest quantiles.
        kwargs: dict
            Argument to be passed to the plotting routine.

        Returns
        -------
        ax: ~matplotlib.axes.Axes
            An axes where the data were plotted.
        """

        if plt is None:
            warnings.warn('Matplotlib not loaded, cannot plot !', ImportWarning)
            return None

        if grid_point is None:
            ni = None
            nj = None
        else:
            ni = grid_point[0]
            nj = grid_point[1]

        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        if timestamps is None:
            timestamps = self._get_timestamps(predictor)
        elif not isinstance(timestamps, list):
            timestamps = [timestamps] * self.number_of_observations

        qq = list()
        for qi in q:
            qq.append(1.-qi)
        low_quant = self.ensemble_quantiles(q, interpolation=low_interpolation)
        high_quant = self.ensemble_quantiles(qq, interpolation=high_interpolation)

        if self._scalars:
            for i in range(self.number_of_observations):
                for qi in range(len(q)):
                    ax.fill_between(timestamps[i], low_quant[predictor, i, qi, variable, :],
                                    high_quant[predictor, i, qi, variable, :],
                                    alpha=alpha+q[qi], **kwargs)
        elif self._fields:
            if ni is None or nj is None:
                warnings.warn('You must specify which grid point to plot, cannot plot !', UserWarning)
                return None
            for i in range(self.number_of_observations):
                for qi in range(len(q)):
                    ax.fill_between(timestamps[i], low_quant[predictor, i, qi, variable, :, ni, nj],
                                    high_quant[predictor, i, qi, variable, :, ni, nj],
                                    alpha=alpha+q[qi], **kwargs)

        return ax

    def plot_ensemble_minmax(self, predictor=0, variable=0, ax=None, timestamps=None, grid_point=None, **kwargs):
        """Plot the data ensemble minimum :attr:`ensemble_min` and maximum :attr:`ensemble_max` as a function of time.

        Parameters
        ----------
        predictor: int, optional
            The predictor index to use. Default is 0.
        variable: int, optional
            The variable index to use.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
        timestamps: None or ~numpy.ndarray(~datetime.datetime), optional
            An array containing the timestamp of the data. If None, try to use the data `timestamps` and in last resort a numbered time index.
            Default to `None`.
        grid_point: tuple(int, int)
            If the data are fields, specifies which grid point to plot.
        kwargs: dict
            Argument to be passed to the plotting routine.

        Returns
        -------
        ax: ~matplotlib.axes.Axes, optional
            An axes where the data were plotted.

        """

        if plt is None:
            warnings.warn('Matplotlib not loaded, cannot plot !', ImportWarning)
            return None

        if grid_point is None:
            ni = None
            nj = None
        else:
            ni = grid_point[0]
            nj = grid_point[1]

        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        if timestamps is None:
            timestamps = self._get_timestamps(predictor)
        elif not isinstance(timestamps, list):
            timestamps = [timestamps] * self.number_of_observations

        mini = self.ensemble_min
        maxi = self.ensemble_max

        if self._scalars:
            for i in range(self.number_of_observations):
                minmax_plot(timestamps[i], mini[predictor, i, 0, variable, :], maxi[predictor, i, 0, variable, :], ax=ax, **kwargs)
        elif self._fields:
            if ni is None or nj is None:
                warnings.warn('You must specify which grid point to plot, cannot plot !', UserWarning)
                return None
            for i in range(self.number_of_observations):
                minmax_plot(timestamps[i], mini[predictor, i, 0, variable, :], maxi[predictor, i, 0, variable, :, ni, nj], ax=ax, **kwargs)

        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=30, horizontalalignment='center')
        return ax

    def plot_CRPS(self, other, predictor=0, variable=0, ax=None, timestamps=None, grid_point=None, **kwargs):
        """Plot the data CRPS :meth:`CRPS` score with respect to observation data (other) a function of time.

        Parameters
        ----------
        other: Data
            Another data structure holding the observations.
        predictor: int, optional
            The predictor index to use. Default is 0.
        variable: int, optional
            The variable index to use.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
        timestamps: None or ~numpy.ndarray(~datetime.datetime), optional
            An array containing the timestamp of the data. If None, try to use the data `timestamps` and in last resort a numbered time index.
            Default to `None`.
        grid_point: tuple(int, int)
            If the data are fields, specifies which grid point to plot.
        kwargs: dict
            Argument to be passed to the plotting routine.

        Returns
        -------
        ax: ~matplotlib.axes.Axes
            An axes where the CRPS data were plotted.

        """

        crps = self.CRPS(other)
        ax = crps.plot(predictor, variable, ax, timestamps, grid_point=grid_point, **kwargs)
        return ax

    def plot_Abs_CRPS(self, other, predictor=0, variable=0, ax=None, timestamps=None, grid_point=None, **kwargs):
        """Plot the data Absolute norm CRPS :meth:`Abs_CRPS` score with respect to observation data (other) a function of time.

        Parameters
        ----------
        other: Data
            Another data structure holding the observations.
        predictor: int, optional
            The predictor index to use. Default is 0.
        variable: int, optional
            The variable index to use.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
        timestamps: None or ~numpy.ndarray(~datetime.datetime), optional
            An array containing the timestamp of the data. If None, try to use the data `timestamps` and in last resort a numbered time index.
            Default to `None`.
        grid_point: tuple(int, int)
            If the data are fields, specifies which grid point to plot.
        kwargs: dict
            Argument to be passed to the plotting routine.

        Returns
        -------
        ax: ~matplotlib.axes.Axes
            An axes where the CRPS data were plotted.

        """

        abs_crps = self.Abs_CRPS(other)
        ax = abs_crps.plot(predictor, variable, ax, timestamps, grid_point=grid_point, **kwargs)
        return ax

    def plot_Ngr_CRPS(self, other, predictor=0, variable=0, ax=None, timestamps=None, grid_point=None, **kwargs):
        """Plot the data Non-homogeneous Gaussian Regression (NGR) CRPS  :meth:`Ngr_CRPS` score with respect to observation data (other) a function of time.

        Parameters
        ----------
        other: Data
            Another data structure holding the observations.
        predictor: int, optional
            The predictor index to use. Default is 0.
        variable: int, optional
            The variable index to use.
        timestamps: None or ~numpy.ndarray(~datetime.datetime), optional
            An array containing the timestamp of the data. If None, try to use the data `timestamps` and in last resort a numbered time index.
            Default to `None`.
        grid_point: tuple(int, int)
            If the data are fields, specifies which grid point to plot.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
        kwargs: dict
            Argument to be passed to the plotting routine.

        Returns
        -------
        ax: ~matplotlib.axes.Axes
            An axes where the CRPS data were plotted.

        """

        if timestamps is None:
            timestamps = self.timestamps

        ngr_crps = self.Ngr_CRPS(other)
        ax = ngr_crps.plot(predictor, variable, ax, timestamps, grid_point=grid_point, **kwargs)
        return ax

    def plot_observational_mean(self, predictor=0, variable=0, ax=None, timestamps=None, grid_point=None, **kwargs):
        """Plot the data observational mean :attr:`observational_mean` as a function of time.

        Parameters
        ----------
        predictor: int, optional
            The predictor index to use. Default is 0.
        variable: int, optional
            The variable index to use.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
        timestamps: None or ~numpy.ndarray(~datetime.datetime), optional
            An array containing the timestamp of the data. If None, try to use the data `timestamps` and in last resort a numbered time index.
            Default to `None`.
        grid_point: tuple(int, int)
            If the data are fields, specifies which grid point to plot.
        kwargs: dict
            Argument to be passed to the plotting routine.

        Returns
        -------
        ax: ~matplotlib.axes.Axes
            An axes where the data were plotted.

        """

        om = self.observational_mean
        ax = om.plot(predictor, variable, ax, timestamps, grid_point=grid_point, **kwargs)
        return ax

    def plot_observational_median(self, predictor=0, variable=0, ax=None, timestamps=None, grid_point=None, **kwargs):
        """Plot the data observational median :attr:`observational_median` as a function of time.

        Parameters
        ----------
        predictor: int, optional
            The predictor index to use. Default is 0.
        variable: int, optional
            The variable index to use.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
        timestamps: None or ~numpy.ndarray(~datetime.datetime), optional
            An array containing the timestamp of the data. If None, try to use the data `timestamps` and in last resort a numbered time index.
            Default to `None`.
        grid_point: tuple(int, int)
            If the data are fields, specifies which grid point to plot.
        kwargs: dict
            Argument to be passed to the plotting routine.

        Returns
        -------
        ax: ~matplotlib.axes.Axes
            An axes where the data were plotted.

        """

        om = self.observational_median
        ax = om.plot(predictor, variable, ax, timestamps, grid_point=grid_point, **kwargs)
        return ax

    def plot_observational_std(self, predictor=0, variable=0, ax=None, timestamps=None, grid_point=None, **kwargs):
        """Plot the data observational standard deviation :attr:`observational_std` as a function of time.

        Parameters
        ----------
        predictor: int, optional
            The predictor index to use. Default is 0.
        variable: int, optional
            The variable index to use.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
        timestamps: None or ~numpy.ndarray(~datetime.datetime), optional
            An array containing the timestamp of the data. If None, try to use the data `timestamps` and in last resort a numbered time index.
            Default to `None`.
        grid_point: tuple(int, int)
            If the data are fields, specifies which grid point to plot.
        kwargs: dict
            Argument to be passed to the plotting routine.

        Returns
        -------
        ax: ~matplotlib.axes.Axes
            An axes where the data were plotted.

        """

        if plt is None:
            warnings.warn('Matplotlib not loaded, cannot plot !', ImportWarning)
            return None

        if grid_point is None:
            ni = None
            nj = None
        else:
            ni = grid_point[0]
            nj = grid_point[1]

        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        if timestamps is None:
            timestamps = self._get_timestamps(predictor)
        elif not isinstance(timestamps, list):
            timestamps = [timestamps] * self.number_of_observations

        em = self.observational_median
        std = np.sqrt(self.observational_var.data)

        if self._scalars:
            for i in range(self.number_of_observations):
                std_plot(timestamps[i], em[predictor, i, 0, variable, :], std[predictor, i, 0, variable, :], ax=ax, **kwargs)
        elif self._fields:
            if ni is None or nj is None:
                warnings.warn('You must specify which grid point to plot, cannot plot !', UserWarning)
                return None
            for i in range(self.number_of_observations):
                std_plot(timestamps[i], em[predictor, i, 0, variable, :, ni, nj], std[predictor, i, 0, variable, :, ni, nj], ax=ax, **kwargs)

        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=30, horizontalalignment='center')
        return ax

    def plot_observational_quantiles(self, q, low_interpolation='linear', high_interpolation='linear',
                                     predictor=0, variable=0, ax=None, timestamps=None, grid_point=None, alpha=0.1, **kwargs):
        """Plot the data observational quantiles :attr:`observational_quantiles` as a function of time.

        Parameters
        ----------
        q: array_like(float)
            Quantile or sequence of quantiles to compute, which must be between 0 and 0.5 exclusive.
            A symmetric quantile with respect to 0.5 will also be computed.
        low_interpolation: str, optional
            This optional parameter specifies the interpolation method to use when the desired lower quantile (q<0.5)
            lies between two data points. See :func:`numpy.quantile` for more information.
        high_interpolation: str, optional
            This optional parameter specifies the interpolation method to use when the desired higher quantile (q>0.5)
            lies between two data points. See :func:`numpy.quantile` for more information.
        predictor: int, optional
            The predictor index to use. Default is 0.
        variable: int, optional
            The variable index to use.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
        timestamps: None or ~numpy.ndarray(~datetime.datetime), optional
            An array containing the timestamp of the data. If None, try to use the data `timestamps` and in last resort a numbered time index.
            Default to `None`.
        grid_point: tuple(int, int)
            If the data are fields, specifies which grid point to plot.
        alpha: float
            Base level of transparency for the highest and lowest quantiles.
        kwargs: dict
            Argument to be passed to the plotting routine.

        Returns
        -------
        ax: ~matplotlib.axes.Axes
            An axes where the data were plotted.
        """

        if plt is None:
            warnings.warn('Matplotlib not loaded, cannot plot !', ImportWarning)
            return None

        if grid_point is None:
            ni = None
            nj = None
        else:
            ni = grid_point[0]
            nj = grid_point[1]

        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        if timestamps is None:
            timestamps = self._get_timestamps(predictor)
        elif not isinstance(timestamps, list):
            timestamps = [timestamps] * self.number_of_observations

        qq = list()
        for qi in q:
            qq.append(1.-qi)
        low_quant = self.observational_quantiles(q, interpolation=low_interpolation)
        high_quant = self.observational_quantiles(qq, interpolation=high_interpolation)

        if self._scalars:
            for i in range(self.number_of_observations):
                for qi in range(len(q)):
                    ax.fill_between(timestamps[i], low_quant[predictor, i, qi, variable, :],
                                    high_quant[predictor, i, qi, variable, :],
                                    alpha=alpha+q[qi], **kwargs)
        elif self._fields:
            if ni is None or nj is None:
                warnings.warn('You must specify which grid point to plot, cannot plot !', UserWarning)
                return None
            for i in range(self.number_of_observations):
                for qi in range(len(q)):
                    ax.fill_between(timestamps[i], low_quant[predictor, i, qi, variable, :, ni, nj],
                                    high_quant[predictor, i, qi, variable, :, ni, nj],
                                    alpha=alpha+q[qi], **kwargs)

        return ax

    def plot_observational_minmax(self, predictor=0, variable=0, ax=None, timestamps=None, grid_point=None, **kwargs):
        """Plot the data observational minimum :attr:`observational_min` and maximum :attr:`observational_max` as a function of time.

        Parameters
        ----------
        predictor: int, optional
            The predictor index to use. Default is 0.
        variable: int, optional
            The variable index to use.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
        timestamps: None or ~numpy.ndarray(~datetime.datetime), optional
            An array containing the timestamp of the data. If None, try to use the data `timestamps` and in last resort a numbered time index.
            Default to `None`.
        grid_point: tuple(int, int)
            If the data are fields, specifies which grid point to plot.
        kwargs: dict
            Argument to be passed to the plotting routine.

        Returns
        -------
        ax: ~matplotlib.axes.Axes, optional
            An axes where the data were plotted.

        """

        if plt is None:
            warnings.warn('Matplotlib not loaded, cannot plot !', ImportWarning)
            return None

        if grid_point is None:
            ni = None
            nj = None
        else:
            ni = grid_point[0]
            nj = grid_point[1]

        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        if timestamps is None:
            timestamps = self._get_timestamps(predictor)
        elif not isinstance(timestamps, list):
            timestamps = [timestamps] * self.number_of_observations

        mini = self.observational_min
        maxi = self.observational_max

        if self._scalars:
            for i in range(self.number_of_observations):
                minmax_plot(timestamps[i], mini[predictor, i, 0, variable, :], maxi[predictor, i, 0, variable, :], ax=ax, **kwargs)
        elif self._fields:
            if ni is None or nj is None:
                warnings.warn('You must specify which grid point to plot, cannot plot !', UserWarning)
                return None
            for i in range(self.number_of_observations):
                minmax_plot(timestamps[i], mini[predictor, i, 0, variable, :], maxi[predictor, i, 0, variable, :, ni, nj], ax=ax, **kwargs)

        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=30, horizontalalignment='center')
        return ax

    def load_scalars(self, data, metadata=None, timestamps=None, load_axis=1, concat_axis=1, columns=0, replace_timestamps=False):
        """Load scalar data in the Data object. For the moment, only Pandas dataframe and NumPy arrays are accepted.

        Parameters
        ----------
        data: ~pandas.DataFrame or ~numpy.ndarray or list(~pandas.DataFrame) or list(~numpy.ndarray)
            The data to load in the object, packed along the load_axis.
            If ~numpy.ndarray are provided, they can be at most 2-dimensional and their last axis is always identified with the lead time.
            The remaining axis will be identified to an axis of the Data object given by `load_axis`
            If ~pandas.DataFrame are provided, there row axis is expected to be identified with the lead time,
            while the columns axis will be identified to an axis of the Data object given by `load_axis`
            In both cases, a list of data can be provided instead, the list items will be loaded along the axis provided by the first element of `load_axis`,
            which must thus be a 2-tuple. If the list elements are 2D, their first axis are loaded along the axis of the Data object corresponding to the second element of
            `load_axis`, the last axis being loaded along the lead time axis. If the list elements are 1D, they are loaded along the lead time axis.
            Finally, if there are already data inside the object, the data provided will be appended to them along the `concat_axis`.
        metadata: object or list(object)
            The metadata of the provided data. If a list of data is provided, then a list of metadata object can be provided. Otherwise, the same
            metadata object will be used for all the data items in the list.
        timestamps: ~numpy.ndarray(~datetime.datetime) or list(~numpy.ndarray(~datetime.datetime))
            The timestamps array(s) of the provided data, as ~datetime.datetime object. If a list of data is provided, then a list of timestamps arrays can be provided.
            Otherwise, the same timestamps array will be used for all the data items in the list.
        load_axis: int or str or tuple(int) or tuple(str)
            Axis over which the provided data are loaded in the Data object.
            Equal to 1 by default to match the observation index.
            Can be a number if `data` is a ~numpy.ndarray or a ~pandas.DataFrame. Have to be 2-tuple if `data` is a list (see above).
            Can also be a string like i.e. 'obs' to load along the observation index, or 'members' to load along the ensemble member index.
        concat_axis: int or str
            Axis over which the data have to be concatenated. Can be a number or
            a string like i.e. 'obs' to concatenate along the observation index, or 'members' to concatenate along the ensemble member index.
        columns: int or str or list(int) or list(str), optional
            Allow to specify the column of the ~pandas.DataFrame to load along `load_axis`. Only works with pandas ~pandas.DataFrame.
        replace_timestamps: bool
            Replace the timestamps possibly already present in the Data object if `concat_axis` is not 1. Default to `False`.
        """

        # checking axis input argument
        load_axis = self._determine_axis(load_axis)
        concat_axis = self._determine_axis(concat_axis)

        if load_axis is None:
            warnings.warn('Wrong axis argument. Data not loaded.', SyntaxWarning)
            return
        if isinstance(load_axis, int):
            if load_axis == 0:
                warnings.warn('Cannot load a predictor with this method. Data not loaded.', UserWarning)
                return
        else:
            if 0 in load_axis:
                warnings.warn('Cannot load a predictor with this method. Data not loaded.', UserWarning)
                return

        # checking the data type of the data input argument
        data_type = self._determine_data_type(data)
        if data_type in ["unknown", "none"]:
            s = ""
            if pd is None:
                s += "\nPlease note that pandas is not loaded. That might cause the error."
            warnings.warn('Wrong data type for the data argument. Data not loaded.' + s, UserWarning)
            return

        # checking the columns input argument
        column_type, columns = self._determine_columns(columns)
        if column_type == "unknown":
            warnings.warn('Wrong columns argument. Data not loaded.', SyntaxWarning)
            return

        # considering the case where the input data are pandas dataframes
        if "pandas" in data_type:

            # loading the data array first
            # case where only one dataframe is provided
            if data_type == "pandas":
                # loading the input data in a numpy array
                new_data = self._create_scalars_data_from_pandas(data, load_axis, columns, column_type, self._dtype)

            # case where a list of data frame is provided
            elif data_type == "pandas_list":

                # loading the input data in a list of numpy array
                new_data = self._create_scalars_data_from_pandas_list(data, load_axis, columns, column_type, self._dtype)

            else:
                warnings.warn('Wrong pandas type. Data not loaded.', UserWarning)
                new_data = None

            if new_data is not None:
                # loading metadata
                new_metadata = self._create_metadata(new_data, metadata)

                # loading timestamps
                new_timestamps = self._create_timestamps(new_data, data, timestamps)

        # considering the case where the input data are numpy arrays
        elif "numpy" in data_type:
            # loading the data array first
            # case where only one array is provided
            if data_type == "numpy":
                new_data = self._create_scalars_data_from_ndarray(data, load_axis)
            # case where a list of array is provided
            elif data_type == "numpy_list":
                new_data = self._create_scalars_data_from_ndarray_list(data, load_axis)
            else:
                warnings.warn('Wrong numpy type. Data not loaded.', UserWarning)
                new_data = None

            if new_data is not None:
                # loading metadata
                new_metadata = self._create_metadata(new_data, metadata)

                # loading timestamps
                new_timestamps = self._create_timestamps(new_data, data, timestamps)

        else:
            new_data = None
            new_timestamps = None
            new_metadata = None

        if new_data is not None:
            # case if the data object is empty
            if self.data is None:
                self.data = new_data
                self.timestamps = new_timestamps
                self.metadata = new_metadata
            # otherwise try to concatenate with current data
            else:
                if not self._scalars:
                    warnings.warn('Existing Data object are not scalars. Data not loaded.', UserWarning)
                    return

                try:
                    self.data = np.concatenate((self.data, new_data), axis=concat_axis)
                except:
                    warnings.warn('Problem to concatenate the new data with one already present in the Data object. Data not loaded', UserWarning)
                    return

                if load_axis == 1:
                    try:
                        self.timestamps = np.concatenate((self.timestamps, new_timestamps), axis=concat_axis)
                    except:
                        warnings.warn('Problem to concatenate the new timestamps with one already present in the Data object. Timestamps not loaded', UserWarning)

                    try:
                        self.metadata = np.concatenate((self.metadata, new_metadata), axis=concat_axis)
                    except:
                        warnings.warn('Problem to concatenate the new metadata with one already present in the Data object. Metadata not loaded', UserWarning)
                elif replace_timestamps:
                    self.timestamps = new_timestamps
                    self.metadata = new_metadata
                else:
                    warnings.warn('Warning. No timestamps or metadata loaded with the new data!', UserWarning)

    def _create_scalars_data_from_pandas(self, data, axis, columns, column_type, output_dtype):

        if column_type == 'label':
            data_np = data.loc[:, columns].values.T.astype(dtype=output_dtype)
        elif column_type == 'number':
            data_np = data.iloc[:, columns].values.T.astype(dtype=output_dtype)
        elif column_type == "all":
            data_np = data.values.T.astype(dtype=output_dtype)
        else:
            warnings.warn('Wrong columns argument. Data not loaded.', SyntaxWarning)
            return None

        new_data = self._create_scalars_data_from_ndarray(data_np, axis)

        return new_data

    @staticmethod
    def _create_scalars_data_from_ndarray(data_np, axis):

        if not isinstance(axis, int):
            warnings.warn('Wrong axis argument. Data not loaded.', SyntaxWarning)
            return None

        if len(data_np.shape) == 1:
            data_np = data_np[np.newaxis, :]

        # determining the shape of the data object being created
        shape = [1 for i in range(axis)] + [data_np.shape[0]] + [1 for i in range(axis + 1, 4)] + [data_np.shape[1]]

        try:
            ret_data = data_np.reshape(shape)
        except:
            warnings.warn('Impossible to reshape properly the input data. Data not loaded.', UserWarning)
            ret_data = None

        return ret_data

    def _create_scalars_data_from_pandas_list(self, data, axis, columns, column_type, output_dtype):

        data_np = list()
        if column_type == 'label':
            for d in data:
                data_np.append(d.loc[:, columns].values.T.astype(dtype=output_dtype))
                if len(data_np[-1].shape) == 1:
                    data_np[-1] = data_np[-1][np.newaxis, :]
        elif column_type == 'number':
            for d in data:
                data_np.append(d.iloc[:, columns].values.T.astype(dtype=output_dtype))
                if len(data_np[-1].shape) == 1:
                    data_np[-1] = data_np[-1][np.newaxis, :]
        elif column_type == "all":
            for d in data:
                data_np.append(d.values.T.astype(dtype=output_dtype))
                if len(data_np[-1].shape) == 1:
                    data_np[-1] = data_np[-1][np.newaxis, :]
        else:
            warnings.warn('Wrong columns argument. Data not loaded.', SyntaxWarning)
            return None

        # check the shape of the provided arrays
        for d in data_np[1:]:
            if d.shape != data_np[0].shape:
                warnings.warn('Incompatible pandas dataframes in the list provided. Data not loaded.', SyntaxWarning)
                return None

        new_data = self._create_scalars_data_from_ndarray_list(data_np, axis)

        return new_data

    @staticmethod
    def _create_scalars_data_from_ndarray_list(data_np, axis):

        # determining the shape of the data object being created
        if not isinstance(axis, (list, tuple)):
            warnings.warn('Wrong axis argument. Data not loaded.', SyntaxWarning)
            return None
        elif len(axis) > 2:
            warnings.warn('Too many axis provided. Only considering the first 2.', SyntaxWarning)
            axis = axis[:2]

        sh = [1, data_np[0].shape[0]]
        if axis[0] == axis[1]:
            warnings.warn('Wrong axis argument. Data not loaded.', SyntaxWarning)
            return None
        elif axis[0] < axis[1]:
            sa = axis
        else:
            sa = axis[::-1]
            sh = sh[::-1]

        shape = [1 for i in range(sa[0])] + [sh[0]] + [1 for i in range(sa[0] + 1, sa[1])] + [sh[1]] + \
                [1 for i in range(sa[1] + 1, 4)] + [data_np[0].shape[1]]

        try:
            new_data = data_np[0].reshape(shape)
            for d in data_np[1:]:
                new_data = np.concatenate((new_data, d.reshape(shape)), axis=axis[0])
        except:
            warnings.warn('Impossible to reshape and concatenate properly the input data. Data not loaded.', UserWarning)
            new_data = None

        return new_data

    @staticmethod
    def _create_metadata(new_data, metadata):

        if metadata is False:  # private way to disable the metadata
            return None
        new_metadata = np.empty(new_data.shape[:2], dtype=object)
        if not isinstance(metadata, (list, tuple)):
            # warnings.warn('List of data provided, but only one metadata object provided. Using the same metadata for each member of the list', UserWarning)
            for i in range(new_metadata.shape[1]):
                new_metadata[0, i] = metadata
        else:
            if len(metadata) == new_metadata.shape[1]:
                for i in range(new_metadata.shape[1]):
                    new_metadata[0, i] = metadata[i]
            else:
                warnings.warn('Incoherent number of data elements and metadata elements. Metadata not loaded.', UserWarning)
        return new_metadata

    @staticmethod
    def _create_timestamps(new_data, data, timestamps):

        if timestamps is False:  # private way to disable the timestamps
            return None
        # check if a timestamp vector list has been provided
        new_timestamps = np.empty(new_data.shape[:2], dtype=object)
        if not isinstance(timestamps, (list, tuple)):
            # warnings.warn('List of data provided, but only one timestamps vectors provided. Using the same vector for each member of the list', UserWarning)
            if timestamps is not None:

                # check the provided timestamp
                if not isinstance(timestamps, np.ndarray):
                    warnings.warn('The provided timestamps do not form a numpy array. Timestamps not loaded.', UserWarning)
                    timestamps = None
                else:
                    # if a valid timestamps array is provided just pass it
                    if timestamps.shape == new_data.shape[:2] and timestamps.dtype is np.dtype(object):
                        return timestamps

                    for t in timestamps:
                        if not isinstance(t, datetime.datetime):
                            raise Exception  # TODO: Debug
                            warnings.warn('The provided timestamps are not datetime object. Timestamps not loaded.', UserWarning)
                            timestamps = None
                            break
                for i in range(new_timestamps.shape[1]):
                    new_timestamps[0, i] = timestamps
            else:
                if isinstance(data, (list, tuple)):
                    timestamps = list()
                    for d in data:
                        try:
                            timestamps.append(d.index.to_pydatetime())
                        except:
                            timestamps.append(None)

                    if len(timestamps) != new_timestamps.shape[1]:
                        timestamps = [None] * new_timestamps.shape[1]

                    for i in range(new_timestamps.shape[1]):
                        new_timestamps[0, i] = timestamps[i]
                else:
                    try:
                        timestamps = data.index.to_pydatetime()
                    except:
                        timestamps = None

                    for i in range(new_timestamps.shape[1]):
                        new_timestamps[0, i] = timestamps

        else:
            if len(timestamps) == new_timestamps.shape[1]:
                for i in range(new_timestamps.shape[1]):
                    new_timestamps[0, i] = timestamps[i]
            else:
                warnings.warn('Incoherent number of data elements and timestamps elements. Timestamps not loaded.', UserWarning)
        return new_timestamps

    def append_predictors(self, data):
        """Append a predictors Data object to the current ones (i.e. along the 0th axis).

        Parameters
        ----------
        data: Data
            The data object of the predictors to append. Must be compatible/broadcastable.
            If the initial Data object is empty, simply copy the `data` object.
        """

        try:
            if isinstance(data, Data):
                if self.data is not None:
                    self.data = np.concatenate((self.data, data.data))
                    if self.metadata is not None:
                        self.metadata = np.concatenate((self.metadata, data.metadata))
                    if self.timestamps is not None:
                        self.timestamps = np.concatenate((self.timestamps, data.timestamps))
                else:
                    self.data = data.data.copy()
                    self.metadata = data.metadata.copy()
                    self.timestamps = data.timestamps.copy()
        except:
            warnings.warn('Unable to append the provided predictors.', UserWarning)

    def append_realizations(self, data):
        """Append a realizations Data object to the current ones (i.e. along the 1st axis).

        Parameters
        ----------
        data: Data
            The data object of the realizations to append. Must be compatible/broadcastable.
            If the initial Data object is empty, simply copy the `data` object.
        """

        try:
            if isinstance(data, Data):
                if self.data is not None:
                    self.data = np.concatenate((self.data, data.data), axis=1)
                    if self.metadata is not None:
                        self.metadata = np.concatenate((self.metadata, data.metadata), axis=1)
                    if self.timestamps is not None:
                        self.timestamps = np.concatenate((self.timestamps, data.timestamps), axis=1)
                else:
                    self.data = data.data.copy()
                    self.metadata = data.metadata.copy()
                    self.timestamps = data.timestamps.copy()
        except:
            warnings.warn('Unable to append the provided realizations.', UserWarning)

    def append_observations(self, data):
        """Append a observations Data object to the current ones (i.e. along the 1st axis). Alias for
        :meth:`.append_realization`.

        Parameters
        ----------
        data: Data
            The data object of the observations to append. Must be compatible/broadcastable.
            If the initial Data object is empty, simply copy the `data` object.
        """

        try:
            if isinstance(data, Data):
                if self.data is not None:
                    self.data = np.concatenate((self.data, data.data), axis=1)
                    if self.metadata is not None:
                        self.metadata = np.concatenate((self.metadata, data.metadata), axis=1)
                    if self.timestamps is not None:
                        self.timestamps = np.concatenate((self.timestamps, data.timestamps), axis=1)
                else:
                    self.data = data.data.copy()
                    self.metadata = data.metadata.copy()
                    self.timestamps = data.timestamps.copy()
        except:
            warnings.warn('Unable to append the provided observations.', UserWarning)

    def append_members(self, data):
        """Append a members Data object to the current ones (i.e. along the 2nd axis).

        Parameters
        ----------
        data: Data
            The data object of the members to append. Must be compatible/broadcastable.
            If the initial Data object is empty, simply copy the `data` object.
        """

        try:
            if isinstance(data, Data):
                if self.data is not None:
                    self.data = np.concatenate((self.data, data.data), axis=1)
                    if self.metadata is not None:
                        self.metadata = np.concatenate((self.metadata, data.metadata), axis=1)
                    if self.timestamps is not None:
                        self.timestamps = np.concatenate((self.timestamps, data.timestamps), axis=1)
                else:
                    self.data = data.data.copy()
                    self.metadata = data.metadata.copy()
                    self.timestamps = data.timestamps.copy()
        except:
            warnings.warn('Unable to append the provided members.', UserWarning)

    def append_variable(self, data):
        warnings.warn('Method not yet implemented.', UserWarning)
        # if isinstance(data, np.ndarray):
        #     if self.data is None:
        #         self.data = data
        #     else:
        #         self.data = np.concatenate((self.data, data), axis=3)
        # elif isinstance(data, Data):
        #     if self.data is None:
        #         if data.data is not None:
        #             self.data = data.data
        #     else:
        #         if data.data is not None:
        #             self.data = np.concatenate((self.data, data.data), axis=3)

    @property
    def ensemble_mean_observational_self_covariance(self):
        """numpy.ndarray: Ensemble mean observational covariance matrix:
        :math:`{\\rm Cov}^{\\rm obs}_{p_1, p_2, v} [\\mu^{\\rm ens}, \\mu^{\\rm ens}] (t)= \\left\\langle \\bar{\\mu}^{\\rm ens}_{p_1,n,v} (t) \, \\bar{\\mu}^{\\rm ens}_{p_2,n,v} (t) \\right\\rangle_n`

        where :math:`\\bar{\\mu}^{\\rm ens}_{p,n,v} (t) =  \\mu^{\\rm ens}_{p,n,v}(t) - \\langle \\mu^{\\rm ens}_{p,n',v}(t) \\rangle_{n'}` and
        where :math:`\\mu^{\\rm ens}_{p,n,v}(t)` is the :attr:`ensemble_mean`.
        """
        em = self.ensemble_mean.centered_observation.data
        res = np.nanmean(em[:, np.newaxis, ...] * em[np.newaxis, ...], axis=2)[:, :, np.newaxis, ...]
        return res

    def ensemble_mean_observational_covariance(self, other):
        """Observational covariance matrix of the ensemble mean with another Data object :math:`\\mathcal{O}`:
        :math:`{\\rm Cov}^{\\rm obs}_{p_1, p_2, v} [\\bar{\\mathcal{O}}^{\\rm obs}, \\mu^{\\rm ens}] (t)= \\left\\langle \\left\\langle\\bar{\\mathcal{O}}^{\\rm obs}_{p_1,n,m,v} (t)\\right\\rangle_m \, \, \\bar{\\mu}^{\\rm ens}_{p_2,n,v} (t) \\right\\rangle_n`

        where :math:`\\bar{\\mu}^{\\rm ens}_{p,n,v} (t) =  \\mu^{\\rm ens}_{p,n,v}(t) - \\langle \\mu^{\\rm ens}_{p,n',v}(t) \\rangle_{n'}` and
        where :math:`\\mu^{\\rm ens}_{p,n,v}(t)` is the :attr:`ensemble_mean`. :math:`\\bar{\\mathcal{O}}^{\\rm obs}_{p,n,m,v} (t)` is the :attr:`centered_observation` of the other Data object.

        Parameters
        ----------
        other: Data
            Another Data object with observations in it.

        Returns
        -------
        Data
            The variance vector.
        """
        em = self.ensemble_mean.centered_observation.data
        om = other.ensemble_mean.centered_observation.data
        return np.nanmean(om[:, np.newaxis, ...] * em[np.newaxis, ...], axis=2)[:, :, np.newaxis, ...]

    def load_from_file(self, filename, **kwargs):
        """Function to load previously saved data with the method :meth:`save_to_file`.

        Parameters
        ----------
        filename: str
            The file name where the Data object was saved.
        kwargs: dict
            Keyword arguments to pass to the pickle module method.
        """
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f, **kwargs)

        self.__dict__.clear()
        self.__dict__.update(tmp_dict)

    def save_to_file(self, filename, **kwargs):
        """Function to save the data to a file with the :mod:`pickle` module.

        Parameters
        ----------
        filename: str
            The file name where to save the Data object.
        kwargs: dict
            Keyword arguments to pass to the pickle module method.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f, **kwargs)

    def _get_timestamps(self, predictor):
        timestamps = list()
        for n in range(self.number_of_observations):
            if self.timestamps[predictor, n] is not None:
                timestamps.append(self.timestamps[predictor, n])
            else:
                timestamps.append(np.array(range(self.number_of_time_steps)))
        return timestamps

    def _determine_axis(self, axis):

        if isinstance(axis, str):
            if axis in self._axis_name_dict.keys():
                axis = self._axis_name_dict[axis]
                return axis
            else:
                return None
        elif isinstance(axis, int):
            return axis
        elif isinstance(axis, (tuple, list)):
            a = list()
            for axe in axis:
                if isinstance(axe, str):
                    if axe in self._axis_name_dict.keys():
                        a.append(self._axis_name_dict[axe])
                    else:
                        return None
                elif isinstance(axe, int):
                    a.append(axe)
                else:
                    return None
            return a
        else:
            return None

    @staticmethod
    def _determine_data_type(data):

        if data is None:
            return "none"

        data_type = ""
        if pd is not None:
            if isinstance(data, pd.DataFrame):
                data_type = "pandas"

        if isinstance(data, np.ndarray):
            data_type = "numpy"

        if isinstance(data, (list, tuple)):
            if pd is not None:
                if isinstance(data[0], pd.DataFrame):
                    data_type = "pandas_list"

            if isinstance(data[0], np.ndarray):
                data_type = "numpy_list"

        if data_type:
            return data_type
        else:
            return "unknown"

    @staticmethod
    def _determine_columns(columns):

        column_type = ""

        if isinstance(columns, int):
            column_type = "number"
        elif isinstance(columns, slice):
            column_type = "number"
        elif isinstance(columns, str):
            if columns == "all":
                column_type = "all"
            else:
                column_type = "label"
        elif isinstance(columns, (list, tuple)):
            if len(columns) > 0:
                if isinstance(columns[0], int):
                    column_type = "number"
                elif isinstance(columns[0], bool):
                    column_type = "label"
                elif isinstance(columns[0], str):
                    column_type = "label"
                else:
                    column_type = "unknown"
            else:
                column_type = "unknown"
        else:
            column_type = "unknown"

        return column_type, columns

    def full_like(self, value, **kwargs):
        """Like :func:`numpy.full_like`, returns a full :class:`Data` object with the same :attr:`index_shape` and
        :attr:`shape` and type as the initial one.

        Parameters
        ----------
        value:
            The fill value to use.
        kwargs: dict
           The argument to pass to :func:`numpy.full_like`.

        Returns
        -------
        Data:
            The full :class:`Data` object
        """
        data = np.full_like(self.data, value, self._dtype, **kwargs)
        return Data(data, self.metadata, self.timestamps, self._dtype)

    def zeros_like(self, **kwargs):
        """Like :func:`numpy.zeros_like`, returns a  :class:`Data` object with the same :attr:`index_shape` and
        :attr:`shape` and type as the initial one, but filled with zeros.

        Parameters
        ----------
        kwargs: dict
           The argument to pass to :func:`numpy.zeros_like`.

        Returns
        -------
        Data:
            The zeros :class:`Data` object
        """
        data = np.zeros_like(self.data, self._dtype, **kwargs)
        return Data(data, self.metadata, self.timestamps, self._dtype)
