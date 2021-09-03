"""
    Member-By-Member postprocessor module
    =====================================

    This module contains :class:`.Data` postprocessors classes applying the postprocessing algorithm detailed in

    * Bert Van Schaeybroeck and Stéphane Vannitsem. Ensemble post-processing using member-by-member approaches: theoretical aspects. *Quarterly Journal
      of the Royal Meteorological Society*, **141** (688):807–818, 2015. URL: `<https://doi.org/10.1002/qj.2397>`_.

    Basically, it corrects a provided :class:`.Data` object :math:`\\mathcal{D}_{p,n,m,v} (t)` provided by applying the formula:

    .. math::

        \\mathcal{D}^C_{n,m,v} (t) = \\alpha_{v} (t) + \\sum_{p=1}^P \\beta_{p,v} (t) \\mu^{\\rm ens}_{p,n,v} (t) + \\tau_{v} (t) \\bar{\\mathcal{D}}^{\\rm ens}_{0,n,m,v} (t)

    where :math:`P` is the total number of predictors (given by the attribute :attr:`~.Data.number_of_predictors` of :math:`\\mathcal{D}`).
    In the formula above:

    * The Data object :math:`\\mathcal{D}_{p,n,m,v} (t)` is assumed to contain all the predictors used to correct the
      variable needed, the first predictor being by convention the variable itself: :math:`\\mathcal{D}_{0,n,m,v} (t)`.
    * :math:`\\mu^{\\rm ens}_{p,n,v} (t)` is the :attr:`.Data.ensemble_mean` of the Data object :math:`\\mathcal{D}_{p,n,m,v} (t)`.
    * :math:`\\tau_{v} (t)` is a multiplicative correction applied to each member of the first predictor of the the :attr:`Data.centered_ensemble` :math:`\\bar{\\mathcal{D}}^{\\rm ens}_{0,n,m,v} (t)`.
    * :math:`\\mathcal{D}^C_{n,m,v} (t)` is the corrected Data object, with an ensemble member-by-member correction.

    Content
    -------

    Several different postprocessors are available, with different strategy to determine the coefficients :math:`\\alpha_{v} (t)`, :math:`\\beta_{p,v} (t)` and :math:`\\tau_{v}`:

    * :class:`BiasCorrection`: A simple correction of the bias.
    * :class:`EnsembleMeanCorrection`: A correction of the ensemble mean.
    * :class:`EnsembleSpreadScalingCorrection`: A correction of the ensemble mean and a correction of the ensemble spread through a multiplicative scaling.
    * :class:`EnsembleSpreadScalingAbsCRPSCorrection`: A correction of the ensemble mean and a correction of the ensemble spread, with a tuning of the parameters to
      minimize the Absolute norm CRPS score :meth:`.Data.Abs_CRPS`.
    * :class:`EnsembleSpreadScalingNgrCRPSCorrection`: A correction of the ensemble mean and a correction of the ensemble spread, with a tuning of the parameters to
      minimize the Non-homogeneous Gaussian Regression (NGR) CRPS score :meth:`.Data.Ngr_CRPS`.
    * :class:`EnsembleAbsCRPSCorrection`: A correction of the ensemble mean and a correction of the ensemble spread with spread scaling and nudging.
      In addition, the parameters are tuned to minimize the Absolute norm CRPS score :meth:`.Data.Abs_CRPS`.
    * :class:`EnsembleNgrCRPSCorrection`: A correction of the ensemble mean and a correction of the ensemble spread with spread scaling and nudging.
      In addition, the parameters are tuned to minimize the Non-homogeneous Gaussian Regression (NGR) CRPS score :meth:`.Data.Ngr_CRPS`.

    Postprocessors
    --------------

    We now detail these postprocessors:

"""
# TODO : - parameters as Data struct, ?
#        - diagnostic tools + plot (DONE)
#        - timestamps in plot (DONE)
#        - 3d diagnostic plot (2 predictors)
#        - add timestamps option in plot_parameters

import numpy as np
import warnings
import datetime
import multiprocessing
from core.utils import map_times_to_int_array
from core.data import Data
from core.postprocessor import PostProcessor
import scipy.optimize as optimize
from scipy.interpolate import interp1d
from itertools import product

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    warnings.warn('Unable to import matplotlib, plotting method will not work.', ImportWarning)
    plt = None

try:
    import pandas as pd
except ModuleNotFoundError:
    warnings.warn('Unable to import pandas, loading pandas DataFrame will not work.', ImportWarning)
    pd = None


class EnsembleMeanCorrection(PostProcessor):
    """A postprocessor to correct the ensemble mean. Correct according to the formula:

    .. math::

        \\mathcal{D}^C_{n,m,v} (t) = \\alpha_{v} (t) + \\sum_{p=1}^P \\beta_{p,v} (t) \\, \\mu^{\\rm ens}_{p,n,v} (t) + \\tau_{v} (t) \\bar{\\mathcal{D}}^{\\rm ens}_{0,n,m,v} (t)

    where :math:`\\alpha_{v} (t) = \\left\\langle \\mathcal{O}_{n,v} (t) - \\sum_{p=1}^P \\beta_{p,v} (t) \\, \\mu^{\\rm ens}_{p,n,v} (t) \\right\\rangle_n` with :math:`\\mathcal{O}_{n,v} (t)`
    the observation training set and :math:`\\mu^{\\rm ens}_{p,n,v} (t)` the ensemble mean of the predictors of the training set.
    :math:`\\bar{\\mathcal{D}}^{\\rm ens}_{0,n,m,v} (t)` is the :attr:`~.Data.centered_ensemble` of the first predictor.
    We have also

    .. math::

        \\beta_{p,v} (t) = \\sum_{p_1=1}^P \\, {\\rm Cov}^{\\rm obs}_{p_1, v} [\\bar{\\mathcal{O}}^{\\rm obs}, \\mu^{\\rm ens}] (t) \\, {\\rm Cov}^{\\rm obs}_{p_1, p_2, v} [\\mu^{\\rm ens}, \\mu^{\\rm ens}] (t)

    where :math:`{\\rm Cov}^{\\rm obs}_{p_1, v} [\\bar{\\mathcal{O}}^{\\rm obs}, \\mu^{\\rm ens}]` is the :meth:`~.Data.ensemble_mean_observational_covariance` with
    the observation training set :math:`\\mathcal{O}_{n,v} (t)`, and :math:`{\\rm Cov}^{\\rm obs}_{p_1, p_2, v} [\\mu^{\\rm ens}, \\mu^{\\rm ens}]` is
    the :attr:`~.Data.ensemble_mean_observational_self_covariance`. Finally, we have :math:`\\tau_{v} (t) = 1` for all :math:`v` and :math:`t`.

    Attributes
    ----------
    parameters_list: list(Data)
        The list of training parameters :math:`\\alpha_{v} (t)`, :math:`\\beta_{n,v} (t)`, :math:`\\tau_{v} (t)`, as :class:`.Data` objects.
        This list is empty until the first :meth:`~.BiasCorrection.train` method call.

    Examples
    --------

    .. plot::
        :format: doctest
        :include-source: 1

        >>> from pp_test.test_data import test_sin
        >>> import postprocessors.MBM as MBM
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Loading random sinusoidal observations and reforecasts
        >>> past_observations, reforecasts = test_sin(60., 3., number_of_observation=200)
        >>> past_observations.index_shape
        (1, 200, 1, 1, 21)
        >>> reforecasts.index_shape
        (1, 200, 10, 1, 21)
        >>>
        >>> # Training the postprocessor
        >>> postprocessor = MBM.EnsembleMeanCorrection()
        >>> postprocessor.train(past_observations, reforecasts)
        >>>
        >>> # Loading new random sinusoidal observations and forecasts
        >>> observations, forecasts = test_sin(60., 3.)
        >>> corrected_forecasts = postprocessor(forecasts)  # Correcting the new forecasts
        >>>
        >>> # Plotting the forecasts
        >>> ax = observations.plot(global_label='observations', color='tab:green')
        >>> ax = forecasts.plot(ax=ax, global_label='raw forecasts', mfc=None, mec='tab:blue',
        ... ls='', marker='o', ms=3.0)
        >>> ax = corrected_forecasts.plot(ax=ax, global_label='corrected forecasts', mfc=None,
        ... mec='tab:orange', ls='', marker='o', ms=3.0)
        >>> t = ax.set_ylabel('temperature [C°]')
        >>> t = ax.set_xlabel('date')
        >>> t = ax.legend()
        >>> t = ax.set_title('Example of corrected forecasts')
        >>>
        >>> # Plotting the postprocessing parameters
        >>> ax = postprocessor.plot_parameters()
        >>> t = ax.set_title('Postprocessing parameters \\n'+r'($\\gamma_{1,0}$ is $\\tau_{0}$)')
        >>>
        >>> # Computing the CRPS score
        >>> raw_crps = forecasts.CRPS(observations)
        >>> corrected_crps = corrected_forecasts.CRPS(observations)
        >>>
        >>> # Plotting the CRPS score
        >>> ax = raw_crps.plot(timestamps=postprocessor.parameters_list[0].timestamps[0, 0],
        ... global_label='Raw forecast CRPS', color='tab:blue')
        >>> ax = corrected_crps.plot(timestamps=postprocessor.parameters_list[0].timestamps[0, 0], ax=ax,
        ... global_label='Corrected forecast CRPS', color='tab:orange')
        >>> t = ax.set_ylabel('CRPS [C°]')
        >>> t = ax.set_xlabel('time [hour]')
        >>> t = ax.set_title('CRPS score')
        >>> t = ax.legend()
        >>> plt.show()

    """

    def __init__(self):

        PostProcessor.__init__(self)

    @staticmethod
    def _compute_em_coefficients(observations, predictors, forecast_init_time=None):

        sigma_v = predictors.ensemble_mean_observational_self_covariance
        sigma_v = np.moveaxis(sigma_v, [0, 1], [-2, -1])
        sigma_ov = predictors.ensemble_mean_observational_covariance(observations)
        sigma_ov = np.moveaxis(sigma_ov, [0, 1], [-2, -1])
        beta = sigma_ov @ np.linalg.inv(sigma_v)
        beta = np.moveaxis(beta, -1, 0)
        index_shape = list(predictors.index_shape)
        index_shape[1] = index_shape[2] = 1
        beta = np.squeeze(beta).reshape(tuple(index_shape) + predictors.shape)

        alpha = observations.observational_mean.data
        alpha -= np.mean(np.sum(beta * predictors.ensemble_mean.data, axis=0), axis=0)[np.newaxis, np.newaxis, ...]

        if predictors.timestamps is not None:
            timestamps = predictors.timestamps[0, 0]
        elif observations.timestamps is not None:
            timestamps = observations.timestamps[0, 0]
        else:
            parameters_time = None
            return Data(alpha, timestamps=parameters_time), Data(beta, timestamps=parameters_time)

        timedeltas = np.diff(timestamps)
        if isinstance(forecast_init_time, int):
            first_dt = timestamps[0].hour - forecast_init_time
            ptime = np.concatenate(([first_dt], first_dt + map_times_to_int_array(timedeltas.cumsum())))
        elif isinstance(forecast_init_time, datetime.datetime):
            first_dt = timestamps[0] - forecast_init_time
            ptime = np.concatenate(([first_dt], map_times_to_int_array((first_dt + timedeltas).cumsum())))
        else:
            first_dt = timestamps[0].hour
            ptime = np.concatenate(([first_dt], first_dt + map_times_to_int_array(timedeltas.cumsum())))
        beta_parameters_time = np.empty((predictors.number_of_predictors, 1), dtype=object)
        alpha_parameters_time = np.empty((1, 1), dtype=object)
        alpha_parameters_time[0, 0] = ptime
        for p in range(predictors.number_of_predictors):
            beta_parameters_time[p, 0] = ptime

        return Data(alpha, timestamps=alpha_parameters_time), Data(beta, timestamps=beta_parameters_time)

    def _corrected_forecast(self, predictors):
        return self._corrected_forecast_mod(predictors, self.parameters_list)

    @staticmethod
    def _corrected_forecast_mod(predictors, parameters_list):
        cem = predictors.centered_ensemble.data[0]
        res = parameters_list[0].data
        res = res + np.sum(parameters_list[1].data * predictors.ensemble_mean.data, axis=0)[np.newaxis, ...]
        res = res + parameters_list[2].data * cem[np.newaxis, ...]
        if predictors.timestamps is not None:
            timestamps = predictors.timestamps[0][np.newaxis, ...]
        else:
            timestamps = None
        return Data(res, metadata=predictors.metadata, timestamps=timestamps, dtype=predictors.dtype)

    def train(self, observations, predictors, forecast_init_time=None, **kwargs):
        """Method to train the postprocessor with an observation and predictors training set. Once trained, the postprocessor parameters
        are stored in the :attr:`~.EnsembleMeanCorrection.parameter_list`.

        Parameters
        ----------
        observations: Data
            A `Data` object containing the observation training set.
        predictors: Data
            A `Data` object containing the predictors training set. Must be broadcastable with the `observations`.
        forecast_init_time: int or ~datetime.datetime, optional
            The time at which the predictors generation begin (the (re)forecasts initial time). If None, assume that this time is zero (00Z).
        """

        alpha, beta = self._compute_em_coefficients(observations, predictors, forecast_init_time)

        self.parameters_list.clear()
        self.parameters_list.append(alpha)
        self.parameters_list.append(beta)
        self.parameters_list.append(alpha.full_like(1.))
        self.parameters_list.append(alpha.zeros_like())

    def __call__(self, predictors, predictor_offset=0, proc_time_offset=0, shift_parameters=0, interpolate_offset=None,
                 init_params=None):
        """Actual postprocessing of new predictors. Return the corrected first predictor.

        Parameters
        ---------
        predictors: Data
            Data object with the predictors to postprocess. The method will get their timestamps and interpolate the parameters for the timestamps that
            are missing.
        predictor_offset: int, optional
            Allow to skip the first `predictor_offset` values of the given predictors. Default to zero.
        proc_time_offset: int, optional
            Indicate at which time in hours the processor start with respect to the start of the forecast. Default to zero.
        interpolate_offset: None or int, optional
            Allow to start the interpolation before the first data point, and specify at which time (in hour after the start of the forecast).
            `None` to disable. Default to `None`.
        init_params: None or list(int) or list(Data), optional
            Specify the initial parameters of the interpolation if `interpolate_offset` is set. `None` to disable.
            Must be set if `interpolate_offset` is set.
            Default to `None`.
        """
        if self.parameters_list:
            if predictors.timestamps is None:
                return self._corrected_forecast(predictors)
            else:
                parameters_list, ppt, t = self.get_interpolated_parameters(predictors, predictor_offset,
                                                                           proc_time_offset, interpolate_offset,
                                                                           init_params)

                if parameters_list is not None:

                    data_time = predictors.timestamps.copy()
                    for i in range(data_time.shape[0]):
                        for j in range(data_time.shape[1]):
                            data_time[i, j] = data_time[i, j][predictor_offset:]

                    offset_predictors = predictors.copy()
                    offset_predictors.data = offset_predictors.data[:, :, :, :, predictor_offset:, ...]
                    offset_predictors.timestamps = data_time

                    return self._corrected_forecast_mod(offset_predictors, parameters_list)
                else:
                    return self._corrected_forecast(predictors)
        else:
            warnings.warn('Postprocessor is not trained. ' +
                          'Impossible to postprocess the predictors.', UserWarning)
            return None

    def plot_parameters(self, timestamps=None, variable='all', ax=None, grid_point=None, **kwargs):

        if plt is None:
            warnings.warn('Matplotlib not loaded, cannot plot !', UserWarning)
            return None

        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        if variable == 'all':
            selected_var = range(self.parameters_list[0].number_of_variables)
        elif isinstance(variable, int):
            selected_var = [variable]
        elif isinstance(variable, slice):
            selected_var = range(variable.stop)[variable]
        elif isinstance(variable, (list, tuple)):
            selected_var = variable
        else:
            warnings.warn('Wrong variable argument, cannot plot !', UserWarning)
            return None

        leg = list()

        for var in selected_var:
            self.parameters_list[0].plot(variable=var, timestamps=timestamps, grid_point=grid_point, ax=ax, **kwargs)
            leg.append(r'$\alpha_{'+str(var)+r'}$')

        for var in selected_var:
            for p in range(self.parameters_list[1].index_shape[0]):
                self.parameters_list[1].plot(predictor=p, variable=var, timestamps=timestamps, grid_point=grid_point, ax=ax, **kwargs)
                leg.append(r'$\beta_{'+str(p)+r','+str(var)+'}$')

        for var in selected_var:
            self.parameters_list[2].plot(variable=var, timestamps=timestamps, grid_point=grid_point, ax=ax, **kwargs)
            leg.append(r'$\gamma_{1,' + str(var) + r'}$')

        for var in selected_var:
            self.parameters_list[3].plot(variable=var, timestamps=timestamps, grid_point=grid_point, ax=ax, **kwargs)
            leg.append(r'$\gamma_{2,' + str(var) + r'}$')

        ax.legend(leg)
        ax.set_xlabel('Timestep')

        return ax

    def _plot_diagnostics(self, observations, predictors, variable=0, predictor=0, timesteps=None, olabel=None, plabel=None,
                          timestamps=None,
                          scatter_kwargs=None,
                          line_kwargs=None):

        if plt is None:
            warnings.warn('Matplotlib not loaded, cannot plot !', UserWarning)
            return None

        if scatter_kwargs is None:
            scatter_kwargs = dict()

        if line_kwargs is None:
            line_kwargs = dict()

        if olabel is None:
            olabel = ''
        else:
            olabel = '(' + olabel + ')'

        if plabel is None:
            plabel = ''
        else:
            plabel = '(' + plabel + ')'

        index = list(range(observations.number_of_time_steps))
        if timesteps is None:
            timesteps = slice(observations.number_of_time_steps)
            index = index[timesteps]
        elif isinstance(timesteps, int):
            timesteps = slice(timesteps, timesteps+1)
            index = index[timesteps]
        elif isinstance(timesteps, (list, tuple)):
            index = timesteps
        elif isinstance(timesteps, slice):
            index = index[timesteps]
        else:
            warnings.warn('Wrong timesteps argument, cannot plot !', UserWarning)
            return None

        if len(index) > 1:
            n_panels = len(observations.data[predictor, 0, 0, variable, index])
            nr = int(np.ceil(n_panels/2))
            fig, axs = plt.subplots(ncols=2, nrows=nr, figsize=(15, 5*nr))
            plt.subplots_adjust(hspace=0.4)
            axsl = axs.flatten()
        else:
            fig = plt.figure()
            axsl = [fig.gca()]
            n_panels = 1

        for i in range(n_panels):
            obs1 = (observations[0, :, 0, variable, index[i]])[..., np.newaxis]
            obs = obs1.copy()
            for m in range(predictors.number_of_members-1):
                obs = np.concatenate((obs, obs1), axis=-1)
            axsl[i].scatter(obs.flatten(), predictors[predictor, :, :, variable, index[i]].flatten(), **scatter_kwargs)
            maxi = np.nanmax(predictors[predictor, :, :, variable, index[i]])
            mini = np.nanmin(predictors[predictor, :, :, variable, index[i]])
            y = np.linspace(mini, maxi)
            axsl[i].plot(self.parameters_list[0][0, 0, 0, variable, index[i]]
                         + self.parameters_list[1][predictor, 0, 0, variable, index[i]] * y, y, **line_kwargs)
            if timestamps is not None:
                try:
                    time = str(timestamps[index[i]])
                except:
                    time = str(index[i] + 1)
            else:
                time = str(index[i]+1)

            if i % 2 == 0:
                axsl[i].set_ylabel('Predictors '+str(predictor)+' '+plabel)
            axsl[i].set_title('time:'+time)
            axsl[i].legend()

        if len(axsl) > 1:
            axsl[-2].set_xlabel('Observations '+olabel)
        axsl[-1].set_xlabel('Observations '+olabel)

        return fig, axsl

    def _plot_interpolated_diagnostics(self, actual_predictors, past_observations, past_predictors, variable=0, predictor=0,
                                       timesteps=None, olabel=None, plabel=None,
                                       scatter_kwargs=None,
                                       line_kwargs=None,
                                       proc_kwargs=None):

        if plt is None:
            warnings.warn('Matplotlib not loaded, cannot plot !', UserWarning)
            return None

        if scatter_kwargs is None:
            scatter_kwargs = dict()

        if line_kwargs is None:
            line_kwargs = dict()

        if olabel is None:
            olabel = ''
        else:
            olabel = '(' + olabel + ')'

        if plabel is None:
            plabel = ''
        else:
            plabel = '(' + plabel + ')'

        parameters_list, pp_time, timestamps = self.get_interpolated_parameters(actual_predictors, **proc_kwargs)

        index = list(range(parameters_list[0].number_of_time_steps))
        if timesteps is None:
            timesteps = slice(parameters_list[0].number_of_time_steps)
            index = index[timesteps]
        elif isinstance(timesteps, int):
            timesteps = slice(timesteps, timesteps+1)
            index = index[timesteps]
        elif isinstance(timesteps, (list, tuple)):
            index = timesteps
        elif isinstance(timesteps, slice):
            index = index[timesteps]
        else:
            warnings.warn('Wrong timesteps argument, cannot plot !', UserWarning)
            return None

        if len(index) > 1:
            n_panels = len(parameters_list[0].data[predictor, 0, 0, variable, index])
            nr = int(np.ceil(n_panels/2))
            fig, axs = plt.subplots(ncols=2, nrows=nr, figsize=(15, 5*nr))
            plt.subplots_adjust(hspace=0.4)
            axsl = axs.flatten()
        else:
            fig = plt.figure()
            axsl = [fig.gca()]
            n_panels = 1

        for i in range(n_panels):
            if timestamps[index[i]] in pp_time:
                pp_index = np.where(pp_time == timestamps[index[i]])[0][0]
                obs1 = (past_observations[0, :, 0, variable, pp_index])[..., np.newaxis]
                obs = obs1.copy()
                for m in range(past_predictors.number_of_members - 1):
                    obs = np.concatenate((obs, obs1), axis=-1)
                axsl[i].scatter(obs.flatten(), past_predictors[predictor, :, :, variable, pp_index].flatten(), **scatter_kwargs)
                maxi = np.nanmax(past_predictors[predictor, :, :, variable, pp_index])
                mini = np.nanmin(past_predictors[predictor, :, :, variable, pp_index])
            else:
                maxi = np.nanmax(actual_predictors.data)
                mini = np.nanmin(actual_predictors.data)

            y = np.linspace(mini, maxi)
            axsl[i].plot(parameters_list[0][0, 0, 0, variable, index[i]]
                         + parameters_list[1][predictor, 0, 0, variable, index[i]] * y, y, **line_kwargs)
            if timestamps is not None:
                try:
                    time = str(timestamps[index[i]])
                except:
                    time = str(index[i] + 1)
            elif past_observations.timestamps is not None and not isinstance(past_observations.timestamps, list):
                try:
                    time = str(past_observations.timestamps[index[i]])
                except:
                    time = str(index[i] + 1)
            elif past_predictors.timestamps is not None and not isinstance(past_observations.timestamps, list):
                try:
                    time = str(past_predictors.timestamps[index[i]])
                except:
                    time = str(index[i] + 1)
            else:
                time = str(index[i]+1)

            if i % 2 == 0:
                axsl[i].set_ylabel('Predictors '+str(predictor)+' '+plabel)
            axsl[i].set_title('time:'+time)
            axsl[i].legend()

        axsl[-2].set_xlabel('Observations '+olabel)
        axsl[-1].set_xlabel('Observations '+olabel)

        return fig, axsl

    def get_interpolated_parameters(self, predictors, predictor_offset=0, proc_time_offset=0,
                                    interpolate_offset=None, init_params=None):
        """Method to get the postprocessor parameters interpolated to fix missing timesteps. Assumes that the parameters evolve "smoothly" with
        the timesteps to fill the gaps.

        Parameters
        ---------
        predictors: Data
            Data object with the predictors to postprocess. The method will get their timestamps and interpolate the parameters for the timestamps that
            are missing.
        predictor_offset: int, optional
            Allow to skip the first `predictor_offset` values of the given predictors. Default to zero.
        proc_time_offset: int, optional
            Indicate at which time in hours the processor start with respect to the start of the forecast. Default to zero.
        interpolate_offset: None or int, optional
            Allow to start the interpolation before the first data point, and specify at which time (in hour after the start of the forecast).
            `None` to disable. Default to `None`.
        init_params: None or list(int) or list(Data), optional
            Specify the initial parameters of the interpolation if `interpolate_offset` is set. `None` to disable.
            Must be set if `interpolate_offset` is set.
            Default to `None`.
        """
        if self.parameters_list:
            if predictors.timestamps is None:
                warnings.warn('Predictors data are not timestamped. Impossible to interpolate the parameters.', UserWarning)
                return None, None, None
            elif self.parameters_list[0].timestamps is None:
                warnings.warn('Postprocessor was not trained with timestamped data. ' +
                              'Impossible to interpolate the parameters.', UserWarning)
                return None, None, None
            else:
                proc_time_offset_delta = datetime.timedelta(hours=proc_time_offset)
                if interpolate_offset is not None:
                    interpolate_offset_delta = datetime.timedelta(hours=interpolate_offset)

                td = np.diff(self.parameters_list[0].timestamps[0, 0])
                timedelta = np.array([datetime.timedelta(seconds=3600 * t) for t in td], dtype=object)

                pp_time = np.concatenate((np.array([predictors.timestamps[0, 0][0] + proc_time_offset_delta]),
                                            predictors.timestamps[0, 0][0] + proc_time_offset_delta + timedelta.cumsum()))
                data_time = predictors.timestamps[0, 0][predictor_offset:]

                combined_time = np.append(pp_time, data_time)
                combined_time = np.unique(combined_time)

                parameters_list_array = list()
                for params in self.parameters_list:
                    shape = list(params.index_shape + params.shape)
                    shape[4] = len(combined_time)
                    parameters_list_array.append(np.full(shape, np.nan))

                k = 0
                for i in range(len(combined_time)):
                    if combined_time[i] in pp_time:
                        for j in range(len(parameters_list_array)):
                            parameters_list_array[j][:, :, :, :, i] = self.parameters_list[j][:, :, :, :, k]

                        k += 1

                for j in range(len(parameters_list_array)):
                    parameters_list_array[j] = parameters_list_array[j][:, :, :, :, :len(data_time), ...]

                combined_time = combined_time[:len(data_time)]

                if interpolate_offset is not None:
                    interp_time = np.insert(combined_time, 0, predictors.timestamps[0, 0][0] + interpolate_offset_delta)
                    shift_index = 1
                else:
                    interp_time = combined_time.copy()
                    shift_index = 0

                interp_time_timestamp = np.array(list(map(lambda x: x.timestamp(), interp_time)))
                interp_time_timestamp = interp_time_timestamp - interp_time_timestamp[0]

                for p in range(len(parameters_list_array)):
                    parameters = parameters_list_array[p]
                    shape = parameters.shape
                    for i, j, k, l in product(range(shape[0]), range(shape[1]), range(shape[2]), range(shape[3])):
                        if interpolate_offset is not None:
                            if isinstance(init_params[p], Data):
                                params = np.insert(parameters[i, j, k, l], 0, init_params[p][i, j, k, l, 0], axis=0)
                            else:
                                params = np.insert(parameters[i, j, k, l], 0, init_params[p], axis=0)
                        else:
                            params = parameters[i, j, k, l].copy()
                        if len(params.shape) == 1:
                            mask = ~np.isnan(params)
                            f = interp1d(interp_time_timestamp[mask], params[mask], kind='cubic')
                            params[~mask] = f(interp_time_timestamp[~mask])
                        else:
                            for ni in range(params.shape[1]):
                                for nj in range(params.shape[2]):
                                    par1d = params[:, ni, nj]
                                    mask = ~np.isnan(par1d)
                                    f = interp1d(interp_time_timestamp[mask], par1d[mask], kind='cubic')
                                    par1d[~mask] = f(interp_time_timestamp[~mask])
                                    params[:, ni, nj] = par1d
                        parameters_list_array[p][i, j, k, l] = params[shift_index:]

                parameters_list = list()
                timedelta = np.diff(np.insert(combined_time, 0, predictors.timestamps[0, 0][0]))
                ptime = map_times_to_int_array(timedelta.cumsum())
                beta_parameters_time = np.empty((self.parameters_list[1].index_shape[0], 1), dtype=object)
                parameters_time = np.empty((1, 1), dtype=object)
                parameters_time[0, 0] = ptime
                for p in range(self.parameters_list[1].index_shape[0]):
                    beta_parameters_time[p, 0] = ptime

                for i, parameters in enumerate(parameters_list_array):
                    if i == 1:
                        parameters_list.append(Data(parameters, timestamps=beta_parameters_time))
                    else:
                        parameters_list.append(Data(parameters, timestamps=parameters_time))

                return parameters_list, pp_time, combined_time

        else:
            warnings.warn('Postprocessor is not trained. ' +
                          'Impossible to interpolate the parameters.', UserWarning)
            return None, None, None

    def plot_interpolated_parameters(self, predictors, proc_kwargs, timestamps=False, variable='all', raw_param_kwargs=None, ax=None, grid_point=None, **kwargs):
        """Method to plot the postprocessor parameters interpolated to fix missing timesteps obtained by the method :meth:`get_interpolated_parameters`.

        Parameters
        ---------
        predictors: Data
            Data object with the predictors to postprocess. The method will get their timestamps and interpolate the parameters for the timestamps that
            are missing.
        proc_kwargs: dict
            Dictionary of arguments to pass to the :meth:`get_interpolated_parameters` method.
        timestamps: array_like, optional
            Timestamps to pass to the plotting routine. Supersedes the timestamps of the `predictors`.
            If `False`, use the timestamps of the `predictors`. Default to `False`.
        variable: str, slice, list or int, optional
            Allow to select for which variable to plot the parameters. Default to `'all'`.
        raw_param_kwargs: dict
            Dictionary of arguments to pass to the `~.Data.plot` method for the raw (not interpolated) parameters.
        ax: ~matplotlib.axes.Axes, optional
            An axes on which to plot.
        grid_point: tuple(int, int), optional
            If the data are fields, specifies the parameters of which grid point to plot.
        kwargs: dict
            Dictionary of arguments to pass to the `~.Data.plot` method for the interpolated parameters.

        Returns
        -------
        ax: ~matplotlib.axes.Axes
            An axes where the parameters were plotted.
        """

        if plt is None:
            warnings.warn('Matplotlib not loaded, cannot plot !', UserWarning)
            return None

        if self.parameters_list:

            if ax is None:
                fig = plt.figure()
                ax = fig.gca()

            leg = list()
            if raw_param_kwargs is None:
                raw_param_kwargs = dict()

            parameters_list, ppt, t = self.get_interpolated_parameters(predictors, **proc_kwargs)
            if timestamps:
                timestamps = t
                pp_timestamps = ppt
            else:
                timestamps = None
                pp_timestamps = None

            if variable == 'all':
                selected_var = range(self.parameters_list[0].number_of_variables)
            elif isinstance(variable, int):
                selected_var = [variable]
            elif isinstance(variable, slice):
                selected_var = range(variable.stop)[variable]
            elif isinstance(variable, (list, tuple)):
                selected_var = variable
            else:
                warnings.warn('Wrong variable argument, cannot plot !', UserWarning)
                return None

            for var in selected_var:
                self.parameters_list[0].plot(variable=var, timestamps=timestamps, ax=ax, grid_point=grid_point, **raw_param_kwargs)
                leg.append(r'$\alpha_{'+str(var)+r'}$')

            for var in selected_var:
                for p in range(self.parameters_list[1].index_shape[0]):
                    self.parameters_list[1].plot(predictor=p, variable=var, timestamps=timestamps, ax=ax, grid_point=grid_point, **raw_param_kwargs)
                    leg.append(r'$\beta_{'+str(p)+r','+str(var)+'}$')

            for var in selected_var:
                self.parameters_list[2].plot(variable=var, timestamps=timestamps, ax=ax, grid_point=grid_point, **raw_param_kwargs)
                leg.append(r'$\gamma_{1,' + str(var) + r'}$')

            for var in selected_var:
                self.parameters_list[3].plot(variable=var, timestamps=timestamps, ax=ax, grid_point=grid_point, **raw_param_kwargs)
                leg.append(r'$\gamma_{2,' + str(var) + r'}$')

            colors = list()
            for l in ax.lines:
                colors.append(l.get_color())
            colors = iter(colors)

            for var in selected_var:
                parameters_list[0].plot(variable=var, timestamps=pp_timestamps, ax=ax, grid_point=grid_point, color=colors.__next__(), **kwargs)

            for var in selected_var:
                for p in range(parameters_list[1].index_shape[0]):
                    parameters_list[1].plot(predictor=p, variable=var, timestamps=pp_timestamps, ax=ax, grid_point=grid_point,
                                            color=colors.__next__(), **kwargs)

            for var in selected_var:
                parameters_list[2].plot(variable=var, timestamps=pp_timestamps, ax=ax, grid_point=grid_point, color=colors.__next__(), **kwargs)

            for var in selected_var:
                parameters_list[3].plot(variable=var, timestamps=pp_timestamps, ax=ax, grid_point=grid_point, color=colors.__next__(), **kwargs)

            for i in range(len(leg)):
                leg.append(leg[i] + ' (interp1d)')
            ax.legend(leg)
            return ax
        else:
            return None


class BiasCorrection(EnsembleMeanCorrection):
    """A postprocessor to correct the bias. Correct according to the formula:

    .. math::

        \\mathcal{D}^C_{n,m,v} (t) = \\alpha_{v} (t) + \\sum_{p=1}^P \\beta_{p,v} (t) \\mu^{\\rm ens}_{p,n,v} (t) + \\tau_{v} (t) \\bar{\\mathcal{D}}^{\\rm ens}_{0,n,m,v} (t)

    where :math:`\\alpha_{v} (t) = \\left\\langle \\mathcal{O}_{n,v} (t) - \\mu^{\\rm ens}_{0,n,v} (t) \\right\\rangle_n` with :math:`\\mathcal{O}_{n,v} (t)` is the observation
    training set and :math:`\\mu^{\\rm ens}_{0,n,v} (t)` is the ensemble mean of the first predictor of the training set, i.e. the :attr:`~.Data.ensemble_mean` of the predictor
    to correct. :math:`\\bar{\\mathcal{D}}^{\\rm ens}_{0,n,m,v} (t)` is the :attr:`~.Data.centered_ensemble` of the first predictor.
    We have also

    * :math:`\\beta_{0,v} (t) = 1`
    * :math:`\\beta_{p>0,v} (t) = 0`
    * :math:`\\tau_{v} (t) = 1`

    for all :math:`v` and :math:`t`.

    Attributes
    ----------
    parameters_list: list(Data)
        The list of training parameters :math:`\\alpha_{v} (t)`, :math:`\\beta_{n,v} (t)`, :math:`\\tau_{v} (t)`, as :class:`.Data` objects.
        This list is empty until the first :meth:`~.BiasCorrection.train` method call.

    Examples
    --------

    .. plot::
        :format: doctest
        :include-source: 1

        >>> from pp_test.test_data import test_sin
        >>> import postprocessors.MBM as MBM
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Loading random sinusoidal observations and reforecasts
        >>> past_observations, reforecasts = test_sin(60., 3., number_of_observation=200)
        >>> past_observations.index_shape
        (1, 200, 1, 1, 21)
        >>> reforecasts.index_shape
        (1, 200, 10, 1, 21)
        >>>
        >>> # Training the postprocessor
        >>> postprocessor = MBM.BiasCorrection()
        >>> postprocessor.train(past_observations, reforecasts)
        >>>
        >>> # Loading new random sinusoidal observations and forecasts
        >>> observations, forecasts = test_sin(60., 3.)
        >>> corrected_forecasts = postprocessor(forecasts)  # Correcting the new forecasts
        >>>
        >>> # Plotting the forecasts
        >>> ax = observations.plot(global_label='observations', color='tab:green')
        >>> ax = forecasts.plot(ax=ax, global_label='raw forecasts', mfc=None, mec='tab:blue',
        ... ls='', marker='o', ms=3.0)
        >>> ax = corrected_forecasts.plot(ax=ax, global_label='corrected forecasts', mfc=None,
        ... mec='tab:orange', ls='', marker='o', ms=3.0)
        >>> t = ax.set_ylabel('temperature [C°]')
        >>> t = ax.set_xlabel('date')
        >>> t = ax.legend()
        >>> t = ax.set_title('Example of corrected forecasts')
        >>>
        >>> # Plotting the bias
        >>> ax = postprocessor.plot_parameters()
        >>> t = ax.set_title('Postprocessing parameters \\n'+r'($\\gamma_{1,0}$ is $\\tau_{0}$)')
        >>>
        >>> # Computing the CRPS score
        >>> raw_crps = forecasts.CRPS(observations)
        >>> corrected_crps = corrected_forecasts.CRPS(observations)
        >>>
        >>> # Plotting the CRPS score
        >>> ax = raw_crps.plot(timestamps=postprocessor.parameters_list[0].timestamps[0, 0],
        ... global_label='Raw forecast CRPS', color='tab:blue')
        >>> ax = corrected_crps.plot(timestamps=postprocessor.parameters_list[0].timestamps[0, 0], ax=ax,
        ... global_label='Corrected forecast CRPS', color='tab:orange')
        >>> t = ax.set_ylabel('CRPS [C°]')
        >>> t = ax.set_xlabel('time [hour]')
        >>> t = ax.set_title('CRPS score')
        >>> t = ax.legend()
        >>> plt.show()

    """

    def __init__(self):

        EnsembleMeanCorrection.__init__(self)

    @staticmethod
    def _compute_bias(observations, predictors, forecast_init_time=None):

        alpha = observations.observational_mean.data
        alpha -= predictors.ensemble_mean.observational_mean.data[0][np.newaxis, ...]

        if predictors.timestamps is not None:
            timestamps = predictors.timestamps[0, 0]
        elif observations.timestamps is not None:
            timestamps = observations.timestamps[0, 0]
        else:
            parameters_time = None
            return Data(alpha, timestamps=parameters_time)

        timedeltas = np.diff(timestamps)
        if isinstance(forecast_init_time, int):
            first_dt = timestamps[0].hour - forecast_init_time
            ptime = np.concatenate(([first_dt], first_dt + map_times_to_int_array(timedeltas.cumsum())))
        elif isinstance(forecast_init_time, datetime.datetime):
            first_dt = timestamps[0] - forecast_init_time
            ptime = np.concatenate(([first_dt], map_times_to_int_array((first_dt + timedeltas).cumsum())))
        else:
            first_dt = timestamps[0].hour
            ptime = np.concatenate(([first_dt], first_dt + map_times_to_int_array(timedeltas.cumsum())))
        parameters_time = np.empty((1, 1), dtype=object)
        parameters_time[0, 0] = ptime

        return Data(alpha, timestamps=parameters_time)

    def train(self, observations, predictors, forecast_init_time=None, **kwargs):
        """Method to train the postprocessor with an observation and predictors training set. Once trained, the postprocessor parameters
        are stored in the :attr:`~.BiasCorrection.parameter_list`.

        Parameters
        ----------
        observations: Data
            A `Data` object containing the observation training set.
        predictors: Data
            A `Data` object containing the predictors training set. Must be broadcastable with the `observations`.
        forecast_init_time: int or ~datetime.datetime, optional
            The time at which the predictors generation begin (the (re)forecasts initial time). If None, assume that this time is zero (00Z).
        """

        alpha = self._compute_bias(observations, predictors, forecast_init_time=None)
        beta = alpha.full_like(1.)
        for _ in range(predictors.number_of_predictors-1):
            beta.append_predictors(alpha.zeros_like())

        self.parameters_list.clear()
        self.parameters_list.append(alpha)
        self.parameters_list.append(beta)
        self.parameters_list.append(alpha.full_like(1.))
        self.parameters_list.append(alpha.zeros_like())


class EnsembleSpreadScalingCorrection(EnsembleMeanCorrection):
    """A postprocessor to correct the ensemble mean and scale the spread of the ensemble. Correct according to the formula:

    .. math::

        \\mathcal{D}^C_{n,m,v} (t) = \\alpha_{v} (t) + \\sum_{p=1}^P \\beta_{p,v} (t) \\, \\mu^{\\rm ens}_{p,n,v} (t) + \\tau_{v} (t) \\bar{\\mathcal{D}}^{\\rm ens}_{0,n,m,v} (t)

    where :math:`\\alpha_{v} (t) = \\left\\langle \\mathcal{O}_{n,v} (t) - \\sum_{p=1}^P \\beta_{p,v} (t) \\, \\mu^{\\rm ens}_{p,n,v} (t) \\right\\rangle_n` with :math:`\\mathcal{O}_{n,v} (t)`
    the observation training set and :math:`\\mu^{\\rm ens}_{p,n,v} (t)` the ensemble mean of the predictors of the training set.
    :math:`\\bar{\\mathcal{D}}^{\\rm ens}_{0,n,m,v} (t)` is the :attr:`~.Data.centered_ensemble` of the first predictor.
    We have also

    .. math::

        \\beta_{p,v} (t) = \\sum_{p_1=1}^P \\, {\\rm Cov}^{\\rm obs}_{p_1, v} [\\bar{\\mathcal{O}}^{\\rm obs}, \\mu^{\\rm ens}] (t) \\, {\\rm Cov}^{\\rm obs}_{p_1, p_2, v} [\\mu^{\\rm ens}, \\mu^{\\rm ens}] (t)

    where :math:`{\\rm Cov}^{\\rm obs}_{p_1, v} [\\bar{\\mathcal{O}}^{\\rm obs}, \\mu^{\\rm ens}]` is the :meth:`~.Data.ensemble_mean_observational_covariance` with
    the observation training set :math:`\\mathcal{O}_{n,v} (t)`, and :math:`{\\rm Cov}^{\\rm obs}_{p_1, p_2, v} [\\mu^{\\rm ens}, \\mu^{\\rm ens}]` is
    the :attr:`~.Data.ensemble_mean_observational_self_covariance`. Finally, we have

    .. math::

        \\tau_{v} (t) =  \\left\\langle \\sigma^{\\rm{ens}}_{n,v} (t)^2 \\right\\rangle_n^{-1} \\, \\left\\{ \\sigma^{\\rm obs}_{v} (t)^2 - \\sum_{p=1}^P \\, \\beta_{p,v} (t) \\, {\\rm Cov}^{\\rm obs}_{p, v} [\\bar{\\mathcal{O}}^{\\rm obs}, \\mu^{\\rm ens}] (t) \\right\}

    for all :math:`n`, which scales the spread of the ensemble over the lead time :math:`t`. In this formula, :math:`\\sigma^{\\rm{ens}}_{n,v} (t)^2` and :math:`\\sigma^{\\rm obs}_{v} (t)^2` stands
    respectively for the ensemble variance (:attr:`~.Data.ensemble_var`) and for the observational variance (:attr:`~.Data.observational_var`) of :math:`\\mathcal{D}`.

    Attributes
    ----------
    parameters_list: list(Data)
        The list of training parameters :math:`\\alpha_{v} (t)`, :math:`\\beta_{n,v} (t)`, :math:`\\tau_{v} (t)`, as :class:`.Data` objects.
        This list is empty until the first :meth:`~.BiasCorrection.train` method call.

    Examples
    --------

    .. plot::
        :format: doctest
        :include-source: 1

        >>> from pp_test.test_data import test_sin
        >>> import postprocessors.MBM as MBM
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Loading random sinusoidal observations and reforecasts
        >>> past_observations, reforecasts = test_sin(60., 3., number_of_observation=200)
        >>> past_observations.index_shape
        (1, 200, 1, 1, 21)
        >>> reforecasts.index_shape
        (1, 200, 10, 1, 21)
        >>>
        >>> # Training the postprocessor
        >>> postprocessor = MBM.EnsembleSpreadScalingCorrection()
        >>> postprocessor.train(past_observations, reforecasts)
        >>>
        >>> # Loading new random sinusoidal observations and forecasts
        >>> observations, forecasts = test_sin(60., 3.)
        >>> corrected_forecasts = postprocessor(forecasts)  # Correcting the new forecasts
        >>>
        >>> # Plotting the forecasts
        >>> ax = observations.plot(global_label='observations', color='tab:green')
        >>> ax = forecasts.plot(ax=ax, global_label='raw forecasts', mfc=None, mec='tab:blue',
        ... ls='', marker='o', ms=3.0)
        >>> ax = corrected_forecasts.plot(ax=ax, global_label='corrected forecasts', mfc=None,
        ... mec='tab:orange', ls='', marker='o', ms=3.0)
        >>> t = ax.set_ylabel('temperature [C°]')
        >>> t = ax.set_xlabel('date')
        >>> t = ax.legend()
        >>> t = ax.set_title('Example of corrected forecasts')
        >>>
        >>> # Plotting the postprocessing parameters
        >>> ax = postprocessor.plot_parameters()
        >>> t = ax.set_title('Postprocessing parameters \\n'+r'($\\gamma_{1,0}$ is $\\tau_{0}$)')
        >>>
        >>> # Computing the CRPS score
        >>> raw_crps = forecasts.CRPS(observations)
        >>> corrected_crps = corrected_forecasts.CRPS(observations)
        >>>
        >>> # Plotting the CRPS score
        >>> ax = raw_crps.plot(timestamps=postprocessor.parameters_list[0].timestamps[0, 0],
        ... global_label='Raw forecast CRPS', color='tab:blue')
        >>> ax = corrected_crps.plot(timestamps=postprocessor.parameters_list[0].timestamps[0, 0], ax=ax,
        ... global_label='Corrected forecast CRPS', color='tab:orange')
        >>> t = ax.set_ylabel('CRPS [C°]')
        >>> t = ax.set_xlabel('time [hour]')
        >>> t = ax.set_title('CRPS score')
        >>> t = ax.legend()
        >>> plt.show()

    """

    def __init__(self):

        EnsembleMeanCorrection.__init__(self)

    @staticmethod
    def _compute_ss_coefficients(observations, predictors, forecast_init_time=None):

        alpha, beta = EnsembleSpreadScalingCorrection._compute_em_coefficients(observations, predictors, forecast_init_time)

        sigma_epsnn = predictors.ensemble_var.observational_mean.data[0]
        sigma_o = observations.observational_var.data
        sigma_ov = predictors.ensemble_mean_observational_covariance(observations)[0]

        gamma_1_sq = (sigma_o - np.sum(beta.data * sigma_ov, axis=0)[np.newaxis, ...])/sigma_epsnn

        return alpha, beta, Data(np.sqrt(gamma_1_sq), metadata=alpha.metadata, timestamps=alpha.timestamps, dtype=alpha.dtype)

    def train(self, observations, predictors, forecast_init_time=None, **kwargs):
        """Method to train the postprocessor with an observation and predictors training set. Once trained, the postprocessor parameters
        are stored in the :attr:`~.BiasCorrection.parameter_list`.

        Parameters
        ----------
        observations: Data
            A `Data` object containing the observation training set.
        predictors: Data
            A `Data` object containing the predictors training set. Must be broadcastable with the `observations`.
        forecast_init_time: int or ~datetime.datetime, optional
            The time at which the predictors generation begin (the (re)forecasts initial time). If None, assume that this time is zero (00Z).
        """

        alpha, beta, gamma_1 = self._compute_ss_coefficients(observations, predictors, forecast_init_time)

        self.parameters_list.clear()
        self.parameters_list.append(alpha)
        self.parameters_list.append(beta)
        self.parameters_list.append(gamma_1)
        self.parameters_list.append(alpha.zeros_like())


class EnsembleSpreadScalingAbsCRPSCorrection(EnsembleSpreadScalingCorrection):
    """A postprocessor to correct the ensemble mean and scale the spread of the ensemble. Correct according to the formula:

    .. math::

        \\mathcal{D}^C_{n,m,v} (t) = \\alpha_{v} (t) + \\sum_{p=1}^P \\beta_{p,v} (t) \\, \\mu^{\\rm ens}_{p,n,v} (t) + \\tau_{v} (t) \\bar{\\mathcal{D}}^{\\rm ens}_{0,n,m,v} (t)

    The parameters :math:`\\alpha_{v} (t)`, :math:`\\beta_{v} (t)` and :math:`\\tau_{v} (t)` are optimized at each lead time to minimize
    the absolute norm CRPS (:meth:`~.Data.Abs_CRPS`) of :math:`\\mathcal{D}^C_{n,m,v} (t)` with the observations :math:`\\mathcal{O}`.
    This postprocessor must thus be trained with a training set composed of past ensembles :math:`\\mathcal{D}_{p,n,m,v}` and observations :math:`\\mathcal{O}`.
    If needed, the parameters initial conditions for the minimization are constructed using the values given by the :class:`EnsembleSpreadScalingCorrection`, dependding on
    the minimizer and options being chosen.

    Attributes
    ----------
    parameters_list: list(Data)
        The list of training parameters :math:`\\alpha_{v} (t)`, :math:`\\beta_{n,v} (t)`, :math:`\\tau_{v} (t)`, as :class:`.Data` objects.
        This list is empty until the first :meth:`~.BiasCorrection.train` method call.

    Examples
    --------

    .. plot::
        :format: doctest
        :include-source: 1

        >>> from pp_test.test_data import test_sin
        >>> import postprocessors.MBM as MBM
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Loading random sinusoidal observations and reforecasts
        >>> past_observations, reforecasts = test_sin(60., 3., number_of_observation=200)
        >>> past_observations.index_shape
        (1, 200, 1, 1, 21)
        >>> reforecasts.index_shape
        (1, 200, 10, 1, 21)
        >>>
        >>> # Training the postprocessor
        >>> postprocessor = MBM.EnsembleSpreadScalingAbsCRPSCorrection()
        >>> postprocessor.train(past_observations, reforecasts, ntrial=10)
        >>>
        >>> # Loading new random sinusoidal observations and forecasts
        >>> observations, forecasts = test_sin(60., 3.)
        >>> corrected_forecasts = postprocessor(forecasts)  # Correcting the new forecasts
        >>>
        >>> # Plotting the forecasts
        >>> ax = observations.plot(global_label='observations', color='tab:green')
        >>> ax = forecasts.plot(ax=ax, global_label='raw forecasts', mfc=None, mec='tab:blue',
        ... ls='', marker='o', ms=3.0)
        >>> ax = corrected_forecasts.plot(ax=ax, global_label='corrected forecasts', mfc=None,
        ... mec='tab:orange', ls='', marker='o', ms=3.0)
        >>> t = ax.set_ylabel('temperature [C°]')
        >>> t = ax.set_xlabel('date')
        >>> t = ax.legend()
        >>> t = ax.set_title('Example of corrected forecasts')
        >>>
        >>> # Plotting the postprocessing parameters
        >>> ax = postprocessor.plot_parameters()
        >>> t = ax.set_title('Postprocessing parameters \\n'+r'($\\gamma_{1,0}$ is $\\tau_{0}$)')
        >>>
        >>> # Computing the CRPS score
        >>> raw_crps = forecasts.CRPS(observations)
        >>> corrected_crps = corrected_forecasts.CRPS(observations)
        >>>
        >>> # Plotting the CRPS score
        >>> ax = raw_crps.plot(timestamps=postprocessor.parameters_list[0].timestamps[0, 0],
        ... global_label='Raw forecast CRPS', color='tab:blue')
        >>> ax = corrected_crps.plot(timestamps=postprocessor.parameters_list[0].timestamps[0, 0], ax=ax,
        ... global_label='Corrected forecast CRPS', color='tab:orange')
        >>> t = ax.set_ylabel('CRPS [C°]')
        >>> t = ax.set_xlabel('time [hour]')
        >>> t = ax.set_title('CRPS score')
        >>> t = ax.legend()
        >>> plt.show()

    """

    def __init__(self):

        EnsembleSpreadScalingCorrection.__init__(self)
        self.min_result = None

    @staticmethod
    def _atomic_corrected_forecast(params, predictors):
        cem = predictors.centered_ensemble.data[0]
        res = params[0]
        res = res + np.sum((params[1:-1] * predictors.ensemble_mean.data.T).T, axis=0)[np.newaxis, ...]
        res = res + (params[-1]) * cem[np.newaxis, ...]
        return Data(res, metadata=False, timestamps=False, dtype=predictors.dtype)

    @staticmethod
    def _atomic_crps(params, observations, predictors):
        corrected = EnsembleSpreadScalingAbsCRPSCorrection._atomic_corrected_forecast(params, predictors)
        crps = corrected.Abs_CRPS(observations)
        return np.squeeze(crps.data)

    @staticmethod
    def _construct_params_ic(parameters_list):
        params = parameters_list[1]
        params = np.insert(params, 0, parameters_list[0])
        params = np.append(params, parameters_list[2])
        return params

    @staticmethod
    def _construct_params_bounds(parameters_list):
        params = list()
        params.append((-2 * np.abs(parameters_list[0]), 2 * np.abs(parameters_list[0])))
        for beta in parameters_list[1]:
            params.append((-2 * np.abs(beta), 2 * np.abs(beta)))
        params.append((0, 2 * np.abs(parameters_list[2])))
        return params

    def _minimize(self, observations, predictors, ntrial=1, num_threads=None, minimize_func=None, **kwargs):

        if 'method' not in kwargs.keys():
            method = 'Nelder-Mead'
            kwargs['method'] = method
        else:
            method = kwargs['method']

        if num_threads is None:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        else:
            pool = multiprocessing.Pool(processes=num_threads)

        if minimize_func is None:
            minimize_func = self._atomic_crps

        if predictors.is_scalar() and observations.is_scalar():
            nstep = predictors.number_of_time_steps
            nvar = predictors.number_of_variables

            alpha = self.parameters_list[0][0, 0, 0]
            beta = self.parameters_list[1][:, 0, 0, :, :]
            gamma_1 = self.parameters_list[2][0, 0, 0]

            min_result = np.empty((nvar, nstep, ntrial), dtype=object)
            crps_result = np.zeros((nvar, nstep, ntrial))
            self.min_result = np.empty((nvar, nstep), dtype=object)
            self.crps_result = np.zeros((nvar, nstep))

            jobs_list = list()

            for t in range(nstep):
                for n in range(nvar):
                    observations_atom = Data(observations[:, :, :, n, t], metadata=False, timestamps=False, dtype=observations.dtype)
                    predictors_atom = Data(predictors[:, :, :, n, t], metadata=False, timestamps=False, dtype=predictors.dtype)
                    if method in ['differential_evolution', 'shgo', 'dual_annealing']:
                        params = self._construct_params_bounds([alpha[n, t], beta[:, n, t], gamma_1[n, t]])
                        jobs_list.append([(n, t), self._minimize_atom, (minimize_func, observations_atom, predictors_atom, params, kwargs)])
                    elif method == 'basinhopping':
                        params = self._construct_params_ic([alpha[n, t], beta[:, n, t], gamma_1[n, t]])
                        jobs_list.append([(n, t), self._minimize_atom, (minimize_func, observations_atom, predictors_atom, params, kwargs)])
                    else:
                        for r in range(ntrial):
                            rand = np.random.random(2)
                            randa = rand[-1] - 0.5
                            rand_beta = np.random.random(len(beta[:, n, t])) - 0.5
                            params = self._construct_params_ic([4 * randa * alpha[n, t],
                                                                4 * rand_beta * beta[:, n, t],
                                                                2 * rand[0] * gamma_1[n, t]])
                            jobs_list.append([(n, t, r), self._minimize_atom, (minimize_func, observations_atom, predictors_atom, params, kwargs)])

            result = pool.map(_compute, jobs_list)

            if method in ['differential_evolution', 'shgo', 'dual_annealing', 'basinhopping']:
                for res in result:
                    n, t = res[0]
                    self.parameters_list[0][0, 0, 0, n, t] = res[1].x[0]
                    self.parameters_list[1][:, 0, 0, n, t] = res[1].x[1:-1]
                    self.parameters_list[2][0, 0, 0, n, t] = np.abs(res[1].x[-1])

                    self.min_result[n, t] = res[1]
                    self.crps_result[n, t] = res[1].fun
            else:
                for res in result:
                    n, t, r = res[0]
                    min_result[n, t, r] = res[1]
                    crps_result[n, t, r] = res[1].fun

                min_index = np.argmin(crps_result, axis=-1)

                for n in range(min_result.shape[0]):
                    for t in range(min_result.shape[1]):
                        res = min_result[n, t, min_index[n, t]]
                        self.parameters_list[0][0, 0, 0, n, t] = res.x[0]
                        self.parameters_list[1][:, 0, 0, n, t] = res.x[1:-1]
                        self.parameters_list[2][0, 0, 0, n, t] = np.abs(res.x[-1])

                        self.min_result[n, t] = res
                        self.crps_result[n, t] = res.fun
        elif predictors.is_field() and observations.is_field():
            nstep = predictors.number_of_time_steps
            nvar = predictors.number_of_variables
            ni, nj = predictors.shape

            alpha = self.parameters_list[0][0, 0, 0]
            beta = self.parameters_list[1][:, 0, 0, :, :]
            gamma_1 = self.parameters_list[2][0, 0, 0]

            min_result = np.empty((nvar, nstep, ni, nj, ntrial), dtype=object)
            crps_result = np.zeros((nvar, nstep, ni, nj, ntrial))
            self.min_result = np.empty((nvar, nstep, ni, nj), dtype=object)
            self.crps_result = np.zeros((nvar, nstep, ni, nj))

            jobs_list = list()

            for t, n, i, j in product(range(nstep), range(nvar), range(ni), range(nj)):
                observations_atom = Data(observations[:, :, :, n, t][..., i, j], metadata=False, timestamps=False, dtype=observations.dtype)
                predictors_atom = Data(predictors[:, :, :, n, t][..., i, j], metadata=False, timestamps=False, dtype=observations.dtype)
                if method in ['differential_evolution', 'shgo', 'dual_annealing']:
                    params = self._construct_params_bounds([alpha[n, t, i, j], beta[:, n, t, i, j], gamma_1[n, t, i, j]])
                    jobs_list.append([(n, t, i, j), self._minimize_atom, (minimize_func, observations_atom, predictors_atom, params, kwargs)])
                elif method == 'basinhopping':
                    params = self._construct_params_ic([alpha[n, t, i, j], beta[:, n, t, i, j], gamma_1[n, t, i, j]])
                    jobs_list.append([(n, t, i, j), self._minimize_atom, (minimize_func, observations_atom, predictors_atom, params, kwargs)])
                else:
                    for r in range(ntrial):
                        rand = np.random.random(2)
                        randa = rand[-1] - 0.5
                        rand_beta = np.random.random(len(beta[:, n, t, i, j])) - 0.5
                        params = self._construct_params_ic([4 * randa * alpha[n, t, i, j],
                                                            4 * rand_beta * beta[:, n, t, i, j],
                                                            2 * rand[0] * gamma_1[n, t, i, j]])
                        jobs_list.append([(n, t, i, j, r), self._minimize_atom, (minimize_func, observations_atom, predictors_atom, params, kwargs)])

            result = pool.map(_compute, jobs_list)

            if method in ['differential_evolution', 'shgo', 'dual_annealing', 'basinhopping']:
                for res in result:
                    n, t, i, j = res[0]
                    self.parameters_list[0].data[0, 0, 0, n, t, i, j] = res[1].x[0]
                    self.parameters_list[1].data[:, 0, 0, n, t, i, j] = res[1].x[1:-1]
                    self.parameters_list[2].data[0, 0, 0, n, t, i, j] = np.abs(res[1].x[-1])

                    self.min_result[n, t, i, j] = res[1]
                    self.crps_result[n, t, i, j] = res[1].fun
            else:
                for res in result:
                    n, t, i, j, r = res[0]
                    min_result[n, t, i, j, r] = res[1]
                    crps_result[n, t, i, j, r] = res[1].fun

                min_index = np.argmin(crps_result, axis=-1)

                for t, n, i, j in product(range(nstep), range(nvar), range(ni), range(nj)):
                    res = min_result[n, t, i, j, min_index[n, t, i, j]]
                    self.parameters_list[0].data[0, 0, 0, n, t, i, j] = res.x[0]
                    self.parameters_list[1].data[:, 0, 0, n, t, i, j] = res.x[1:-1]
                    self.parameters_list[2].data[0, 0, 0, n, t, i, j] = np.abs(res.x[-1])

                    self.min_result[n, t, i, j] = res
                    self.crps_result[n, t, i, j] = res.fun

        pool.terminate()

    @staticmethod
    def _minimize_atom(minimize_func, observations_atom, predictors_atom, params, kwargs):

        if 'method' not in kwargs.keys():
            method = 'Nelder-Mead'
        else:
            method = kwargs['method']

        if method not in ['differential_evolution', 'shgo', 'dual_annealing', 'basinhopping']:
            optimizer = optimize.minimize
            opt_kwargs = kwargs
        else:
            optimizer = getattr(optimize, method)
            opt_kwargs = dict(kwargs)
            opt_kwargs.pop('method')

        res = optimizer(minimize_func, params, args=(observations_atom, predictors_atom), **opt_kwargs)

        return res

    def train(self, observations, predictors, forecast_init_time=None, num_threads=None, ntrial=1, **kwargs):
        """Method to train the postprocessor with an observation and predictors training set. Once trained, the postprocessor parameters
        are stored in the :attr:`~.BiasCorrection.parameter_list`.

        Parameters
        ----------
        observations: Data
            A `Data` object containing the observation training set.
        predictors: Data
            A `Data` object containing the predictors training set. Must be broadcastable with the `observations`.
        forecast_init_time: int or ~datetime.datetime, optional
            The time at which the predictors generation begin (the (re)forecasts initial time). If None, assume that this time is zero (00Z).
        num_threads: None or int, optional
            The number of threads used to compute the parameters. One thread computes a chunk of the given grid points and lead times.
            If `None`, uses the maximum number of cpu cores available.
            Default to `None`.
        ntrial: int
            Number of parameters initial conditions to try to find the best solution. Not used if the minimizer being chosen is a global one.
            Default to 1.
        kwargs: dict
            Dictionary of options to pass to the :func:`scipy.optimize.minimize` function.
        """

        alpha, beta, gamma_1 = EnsembleSpreadScalingCorrection._compute_ss_coefficients(observations, predictors, forecast_init_time)

        self.parameters_list = [alpha, beta, gamma_1, alpha.zeros_like()]

        self._minimize(observations, predictors, num_threads=num_threads, ntrial=ntrial, **kwargs)


class EnsembleSpreadScalingNgrCRPSCorrection(EnsembleSpreadScalingAbsCRPSCorrection):
    """A postprocessor to correct the ensemble mean and scale the spread of the ensemble. Correct according to the formula:

    .. math::

        \\mathcal{D}^C_{n,m,v} (t) = \\alpha_{v} (t) + \\sum_{p=1}^P \\beta_{p,v} (t) \\, \\mu^{\\rm ens}_{p,n,v} (t) + \\tau_{v} (t) \\bar{\\mathcal{D}}^{\\rm ens}_{0,n,m,v} (t)

    The parameters :math:`\\alpha_{v} (t)`, :math:`\\beta_{v} (t)` and :math:`\\tau_{v} (t)` are optimized at each lead time to minimize
    the Non-homogeneous Gaussian Regression (NGR) CRPS (:meth:`~.Data.Abs_Ngr`) of :math:`\\mathcal{D}^C_{n,m,v} (t)` with the observations :math:`\\mathcal{O}`.
    This postprocessor must thus be trained with a training set composed of past ensembles :math:`\\mathcal{D}_{p,n,m,v}` and observations :math:`\\mathcal{O}`.
    If needed, the parameters initial conditions for the minimization are constructed using the values given by the :class:`EnsembleSpreadScalingCorrection`, dependding on
    the minimizer and options being chosen.

    Attributes
    ----------
    parameters_list: list(Data)
        The list of training parameters :math:`\\alpha_{v} (t)`, :math:`\\beta_{n,v} (t)`, :math:`\\tau_{v} (t)`, as :class:`.Data` objects.
        This list is empty until the first :meth:`~.BiasCorrection.train` method call.

    Examples
    --------

    .. plot::
        :format: doctest
        :include-source: 1

        >>> from pp_test.test_data import test_sin
        >>> import postprocessors.MBM as MBM
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Loading random sinusoidal observations and reforecasts
        >>> past_observations, reforecasts = test_sin(60., 3., number_of_observation=200)
        >>> past_observations.index_shape
        (1, 200, 1, 1, 21)
        >>> reforecasts.index_shape
        (1, 200, 10, 1, 21)
        >>>
        >>> # Training the postprocessor
        >>> postprocessor = MBM.EnsembleSpreadScalingNgrCRPSCorrection()
        >>> postprocessor.train(past_observations, reforecasts, ntrial=10)
        >>>
        >>> # Loading new random sinusoidal observations and forecasts
        >>> observations, forecasts = test_sin(60., 3.)
        >>> corrected_forecasts = postprocessor(forecasts)  # Correcting the new forecasts
        >>>
        >>> # Plotting the forecasts
        >>> ax = observations.plot(global_label='observations', color='tab:green')
        >>> ax = forecasts.plot(ax=ax, global_label='raw forecasts', mfc=None, mec='tab:blue',
        ... ls='', marker='o', ms=3.0)
        >>> ax = corrected_forecasts.plot(ax=ax, global_label='corrected forecasts', mfc=None,
        ... mec='tab:orange', ls='', marker='o', ms=3.0)
        >>> t = ax.set_ylabel('temperature [C°]')
        >>> t = ax.set_xlabel('date')
        >>> t = ax.legend()
        >>> t = ax.set_title('Example of corrected forecasts')
        >>>
        >>> # Plotting the postprocessing parameters
        >>> ax = postprocessor.plot_parameters()
        >>> t = ax.set_title('Postprocessing parameters \\n'+r'($\\gamma_{1,0}$ is $\\tau_{0}$)')
        >>>
        >>> # Computing the CRPS score
        >>> raw_crps = forecasts.CRPS(observations)
        >>> corrected_crps = corrected_forecasts.CRPS(observations)
        >>>
        >>> # Plotting the CRPS score
        >>> ax = raw_crps.plot(timestamps=postprocessor.parameters_list[0].timestamps[0, 0],
        ... global_label='Raw forecast CRPS', color='tab:blue')
        >>> ax = corrected_crps.plot(timestamps=postprocessor.parameters_list[0].timestamps[0, 0], ax=ax,
        ... global_label='Corrected forecast CRPS', color='tab:orange')
        >>> t = ax.set_ylabel('CRPS [C°]')
        >>> t = ax.set_xlabel('time [hour]')
        >>> t = ax.set_title('CRPS score')
        >>> t = ax.legend()
        >>> plt.show()

    """

    def __init__(self):

        EnsembleSpreadScalingCorrection.__init__(self)
        self.min_result = None

    @staticmethod
    def _atomic_crps(params, observations, predictors):
        corrected = EnsembleSpreadScalingNgrCRPSCorrection._atomic_corrected_forecast(params, predictors)
        crps = corrected.Ngr_CRPS(observations)
        return np.squeeze(crps.data)


class EnsembleAbsCRPSCorrection(EnsembleSpreadScalingCorrection):
    """A postprocessor to correct the ensemble mean and scale the spread of the ensemble. Correct according to the formula:

    .. math::

        \\mathcal{D}^C_{n,m,v} (t) = \\alpha_{v} (t) + \\sum_{p=1}^P \\beta_{p,v} (t) \\, \\mu^{\\rm ens}_{p,n,v} (t) + \\left(\\gamma_{1,v} (t) + \\gamma_{2,v} (t) / \\delta_{0, n, v} (t) \\right) \\, \\bar{\\mathcal{D}}^{\\rm ens}_{0,n,m,v} (t)

    where :math:`\\delta_{0, n, v} (t)` is the average over the ensemble members of the  ensemble members distance (:attr:`~.Data.delta`).
    The parameters :math:`\\alpha_{v} (t)`, :math:`\\beta_{v} (t)`, :math:`\\gamma_{1,v} (t)` and :math:`\\gamma_{2, v} (t)` are optimized at each lead time to minimize
    the absolute norm CRPS (:meth:`~.Data.Abs_CRPS`) of :math:`\\mathcal{D}^C_{n,m,v} (t)` with the observations :math:`\\mathcal{O}`.
    This postprocessor must thus be trained with a training set composed of past ensembles :math:`\\mathcal{D}_{p,n,m,v}` and observations :math:`\\mathcal{O}`.
    If needed, the parameters initial conditions for the minimization are constructed using the values given by the :class:`EnsembleSpreadScalingCorrection`, dependding on
    the minimizer and options being chosen.

    The parameters :math:`\\gamma_{1,v} (t)` and :math:`\\gamma_{2,v} (t)` control respectively the scaling of the ensemble spread and its nudging toward the
    climatological variance for the long lead times.

    Attributes
    ----------
    parameters_list: list(Data)
        The list of training parameters :math:`\\alpha_{v} (t)`, :math:`\\beta_{n,v} (t)`, :math:`\\tau_{v} (t)`, as :class:`.Data` objects.
        This list is empty until the first :meth:`~.BiasCorrection.train` method call.

    Examples
    --------

    .. plot::
        :format: doctest
        :include-source: 1

        >>> from pp_test.test_data import test_sin
        >>> import postprocessors.MBM as MBM
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Loading random sinusoidal observations and reforecasts
        >>> past_observations, reforecasts = test_sin(60., 3., number_of_observation=200)
        >>> past_observations.index_shape
        (1, 200, 1, 1, 21)
        >>> reforecasts.index_shape
        (1, 200, 10, 1, 21)
        >>>
        >>> # Training the postprocessor
        >>> postprocessor = MBM.EnsembleAbsCRPSCorrection()
        >>> postprocessor.train(past_observations, reforecasts, ntrial=10)
        >>>
        >>> # Loading new random sinusoidal observations and forecasts
        >>> observations, forecasts = test_sin(60., 3.)
        >>> corrected_forecasts = postprocessor(forecasts)  # Correcting the new forecasts
        >>>
        >>> # Plotting the forecasts
        >>> ax = observations.plot(global_label='observations', color='tab:green')
        >>> ax = forecasts.plot(ax=ax, global_label='raw forecasts', mfc=None, mec='tab:blue',
        ... ls='', marker='o', ms=3.0)
        >>> ax = corrected_forecasts.plot(ax=ax, global_label='corrected forecasts', mfc=None,
        ... mec='tab:orange', ls='', marker='o', ms=3.0)
        >>> t = ax.set_ylabel('temperature [C°]')
        >>> t = ax.set_xlabel('date')
        >>> t = ax.legend()
        >>> t = ax.set_title('Example of corrected forecasts')
        >>>
        >>> # Plotting the postprocessing parameters
        >>> ax = postprocessor.plot_parameters()
        >>> t = ax.set_title('Postprocessing parameters')
        >>>
        >>> # Computing the CRPS score
        >>> raw_crps = forecasts.CRPS(observations)
        >>> corrected_crps = corrected_forecasts.CRPS(observations)
        >>>
        >>> # Plotting the CRPS score
        >>> ax = raw_crps.plot(timestamps=postprocessor.parameters_list[0].timestamps[0, 0],
        ... global_label='Raw forecast CRPS', color='tab:blue')
        >>> ax = corrected_crps.plot(timestamps=postprocessor.parameters_list[0].timestamps[0, 0], ax=ax,
        ... global_label='Corrected forecast CRPS', color='tab:orange')
        >>> t = ax.set_ylabel('CRPS [C°]')
        >>> t = ax.set_xlabel('time [hour]')
        >>> t = ax.set_title('CRPS score')
        >>> t = ax.legend()
        >>> plt.show()

    """

    def __init__(self):

        EnsembleSpreadScalingCorrection.__init__(self)
        self.min_result = None
        self.crps_result = None

    @staticmethod
    def _corrected_forecast_mod(predictors, parameters_list):
        cem = predictors.centered_ensemble.data[0]
        res = parameters_list[0].data
        res = res + np.sum(parameters_list[1].data * predictors.ensemble_mean.data, axis=0)[np.newaxis, ...]
        res = res + (parameters_list[2].data + parameters_list[3].data / predictors.delta.data[0]) * cem[np.newaxis, ...]
        if predictors.timestamps is not None:
            timestamps = predictors.timestamps[0][np.newaxis, ...]
        else:
            timestamps = None
        return Data(res, metadata=predictors.metadata, timestamps=timestamps, dtype=predictors.dtype)

    @staticmethod
    def _atomic_corrected_forecast(params, predictors):
        cem = predictors.centered_ensemble.data[0]
        res = params[0]
        res = res + np.sum((params[1:-2] * predictors.ensemble_mean.data.T).T, axis=0)[np.newaxis, ...]
        res = res + (np.abs(params[-2]) + np.abs(params[-1]) / predictors.delta.data[0]) * cem[np.newaxis, ...]
        return Data(res, metadata=False, timestamps=False, dtype=predictors.dtype)

    @staticmethod
    def _atomic_crps(params, observations, predictors):
        corrected = EnsembleAbsCRPSCorrection._atomic_corrected_forecast(params, predictors)
        crps = corrected.Abs_CRPS(observations)
        return np.squeeze(crps.data)

    @staticmethod
    def _construct_params_ic(parameters_list):
        params = parameters_list[1]
        params = np.insert(params, 0, parameters_list[0])
        params = np.append(params, parameters_list[2:])
        return params

    @staticmethod
    def _construct_params_bounds(parameters_list):
        params = list()
        params.append((-2 * np.abs(parameters_list[0]), 2 * np.abs(parameters_list[0])))
        for beta in parameters_list[1]:
            params.append((-2 * np.abs(beta), 2 * np.abs(beta)))
        params.append((0, 2 * np.abs(parameters_list[2])))
        params.append((0, 2 * np.abs(parameters_list[3])))
        return params

    def _minimize(self, observations, predictors, ntrial=1, num_threads=None, minimize_func=None, **kwargs):

        if 'method' not in kwargs.keys():
            method = 'Nelder-Mead'
            kwargs['method'] = method
        else:
            method = kwargs['method']

        if num_threads is None:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        else:
            pool = multiprocessing.Pool(processes=num_threads)

        if minimize_func is None:
            minimize_func = self._atomic_crps

        if predictors.is_scalar() and observations.is_scalar():
            nstep = predictors.number_of_time_steps
            nvar = predictors.number_of_variables

            alpha = self.parameters_list[0][0, 0, 0]
            beta = self.parameters_list[1][:, 0, 0, :, :]
            gamma_1 = self.parameters_list[2][0, 0, 0]
            gamma_2 = self.parameters_list[3][0, 0, 0]

            min_result = np.empty((nvar, nstep, ntrial), dtype=object)
            crps_result = np.zeros((nvar, nstep, ntrial))
            self.min_result = np.empty((nvar, nstep), dtype=object)
            self.crps_result = np.zeros((nvar, nstep))

            jobs_list = list()

            for t in range(nstep):
                for n in range(nvar):
                    observations_atom = Data(observations[:, :, :, n, t], metadata=False, timestamps=False, dtype=observations.dtype)
                    predictors_atom = Data(predictors[:, :, :, n, t], metadata=False, timestamps=False, dtype=predictors.dtype)
                    if method in ['differential_evolution', 'shgo', 'dual_annealing']:
                        params = self._construct_params_bounds([alpha[n, t], beta[:, n, t], gamma_1[n, t], gamma_2[n, t]])
                        jobs_list.append([(n, t), self._minimize_atom, (minimize_func, observations_atom, predictors_atom, params, kwargs)])
                    elif method == 'basinhopping':
                        params = self._construct_params_ic([alpha[n, t], beta[:, n, t], gamma_1[n, t], gamma_2[n, t]])
                        jobs_list.append([(n, t), self._minimize_atom, (minimize_func, observations_atom, predictors_atom, params, kwargs)])
                    else:
                        for r in range(ntrial):
                            rand = np.random.random(3)
                            randa = rand[-1] - 0.5
                            rand_beta = np.random.random(len(beta[:, n, t])) - 0.5
                            params = self._construct_params_ic([4 * randa * alpha[n, t],
                                                                4 * rand_beta * beta[:, n, t],
                                                                2 * rand[0] * gamma_1[n, t],
                                                                2 * rand[1] * gamma_2[n, t]])
                            jobs_list.append([(n, t, r), self._minimize_atom, (minimize_func, observations_atom, predictors_atom, params, kwargs)])

            result = pool.map(_compute, jobs_list)

            if method in ['differential_evolution', 'shgo', 'dual_annealing', 'basinhopping']:
                for res in result:
                    n = res[0][0]
                    t = res[0][1]
                    self.parameters_list[0][0, 0, 0, n, t] = res[1].x[0]
                    self.parameters_list[1][:, 0, 0, n, t] = res[1].x[1:-2]
                    self.parameters_list[2][0, 0, 0, n, t] = np.abs(res[1].x[-2])
                    self.parameters_list[3][0, 0, 0, n, t] = np.abs(res[1].x[-1])

                    self.min_result[n, t] = res[1]
                    self.crps_result[n, t] = res[1].fun
            else:
                for res in result:
                    n = res[0][0]
                    t = res[0][1]
                    r = res[0][2]
                    min_result[n, t, r] = res[1]
                    crps_result[n, t, r] = res[1].fun

                min_index = np.argmin(crps_result, axis=-1)

                for n in range(min_result.shape[0]):
                    for t in range(min_result.shape[1]):
                        res = min_result[n, t, min_index[n, t]]
                        self.parameters_list[0][0, 0, 0, n, t] = res.x[0]
                        self.parameters_list[1][:, 0, 0, n, t] = res.x[1:-2]
                        self.parameters_list[2][0, 0, 0, n, t] = np.abs(res.x[-2])
                        self.parameters_list[3][0, 0, 0, n, t] = np.abs(res.x[-1])

                        self.min_result[n, t] = res
                        self.crps_result[n, t] = res.fun

        elif predictors.is_field() and observations.is_field():
            nstep = predictors.number_of_time_steps
            nvar = predictors.number_of_variables
            ni, nj = predictors.shape

            alpha = self.parameters_list[0][0, 0, 0]
            beta = self.parameters_list[1][:, 0, 0, :, :]
            gamma_1 = self.parameters_list[2][0, 0, 0]
            gamma_2 = self.parameters_list[3][0, 0, 0]

            min_result = np.empty((nvar, nstep, ni, nj, ntrial), dtype=object)
            crps_result = np.zeros((nvar, nstep, ni, nj, ntrial))
            self.min_result = np.empty((nvar, nstep, ni, nj), dtype=object)
            self.crps_result = np.zeros((nvar, nstep, ni, nj))

            jobs_list = list()

            for t, n, i, j in product(range(nstep), range(nvar), range(ni), range(nj)):
                observations_atom = Data(observations[:, :, :, n, t][..., i, j], metadata=False, timestamps=False, dtype=observations.dtype)
                predictors_atom = Data(predictors[:, :, :, n, t][..., i, j], metadata=False, timestamps=False, dtype=observations.dtype)
                if method in ['differential_evolution', 'shgo', 'dual_annealing']:
                    params = self._construct_params_bounds([alpha[n, t, i, j], beta[:, n, t, i, j], gamma_1[n, t, i, j], gamma_2[n, t, i, j]])
                    jobs_list.append([(n, t, i, j), self._minimize_atom, (minimize_func, observations_atom, predictors_atom, params, kwargs)])
                elif method == 'basinhopping':
                    params = self._construct_params_ic([alpha[n, t, i, j], beta[:, n, t, i, j], gamma_1[n, t, i, j], gamma_2[n, t, i, j]])
                    jobs_list.append([(n, t, i, j), self._minimize_atom, (minimize_func, observations_atom, predictors_atom, params, kwargs)])
                else:
                    for r in range(ntrial):
                        rand = np.random.random(3)
                        randa = rand[-1] - 0.5
                        rand_beta = np.random.random(len(beta[:, n, t, i, j])) - 0.5
                        params = self._construct_params_ic([4 * randa * alpha[n, t, i, j],
                                                            4 * rand_beta * beta[:, n, t, i, j],
                                                            2 * rand[0] * gamma_1[n, t, i, j],
                                                            2 * rand[1] * gamma_2[n, t, i, j]])
                        jobs_list.append([(n, t, i, j, r), self._minimize_atom, (minimize_func, observations_atom, predictors_atom, params, kwargs)])

            result = pool.map(_compute, jobs_list)

            if method in ['differential_evolution', 'shgo', 'dual_annealing', 'basinhopping']:
                for res in result:
                    n, t, i, j = res[0]
                    self.parameters_list[0].data[0, 0, 0, n, t, i, j] = res[1].x[0]
                    self.parameters_list[1].data[:, 0, 0, n, t, i, j] = res[1].x[1:-2]
                    self.parameters_list[2].data[0, 0, 0, n, t, i, j] = np.abs(res[1].x[-2])
                    self.parameters_list[3].data[0, 0, 0, n, t, i, j] = np.abs(res[1].x[-1])

                    self.min_result[n, t, i, j] = res[1]
                    self.crps_result[n, t, i, j] = res[1].fun
            else:
                for res in result:
                    n, t, i, j, r = res[0]
                    min_result[n, t, i, j, r] = res[1]
                    crps_result[n, t, i, j, r] = res[1].fun

                min_index = np.argmin(crps_result, axis=-1)

                for t, n, i, j in product(range(nstep), range(nvar), range(ni), range(nj)):
                    res = min_result[n, t, i, j, min_index[n, t, i, j]]
                    self.parameters_list[0].data[0, 0, 0, n, t, i, j] = res.x[0]
                    self.parameters_list[1].data[:, 0, 0, n, t, i, j] = res.x[1:-2]
                    self.parameters_list[2].data[0, 0, 0, n, t, i, j] = np.abs(res.x[-2])
                    self.parameters_list[3].data[0, 0, 0, n, t, i, j] = np.abs(res.x[-1])

                    self.min_result[n, t, i, j] = res
                    self.crps_result[n, t, i, j] = res.fun

        pool.terminate()

    @staticmethod
    def _minimize_atom(minimize_func, observations_atom, predictors_atom, params, kwargs):

        if 'method' not in kwargs.keys():
            method = 'Nelder-Mead'
        else:
            method = kwargs['method']

        if method not in ['differential_evolution', 'shgo', 'dual_annealing', 'basinhopping']:
            optimizer = optimize.minimize
            opt_kwargs = kwargs
        else:
            optimizer = getattr(optimize, method)
            opt_kwargs = dict(kwargs)
            opt_kwargs.pop('method')

        res = optimizer(minimize_func, params, args=(observations_atom, predictors_atom), **opt_kwargs)

        return res

    def train(self, observations, predictors, forecast_init_time=None, num_threads=None, ntrial=1, **kwargs):
        """Method to train the postprocessor with an observation and predictors training set. Once trained, the postprocessor parameters
        are stored in the :attr:`~.BiasCorrection.parameter_list`.

        Parameters
        ----------
        observations: Data
            A `Data` object containing the observation training set.
        predictors: Data
            A `Data` object containing the predictors training set. Must be broadcastable with the `observations`.
        forecast_init_time: int or ~datetime.datetime, optional
            The time at which the predictors generation begin (the (re)forecasts initial time). If None, assume that this time is zero (00Z).
        num_threads: None or int, optional
            The number of threads used to compute the parameters. One thread computes a chunk of the given grid points and lead times.
            If `None`, uses the maximum number of cpu cores available.
            Default to `None`.
        ntrial: int
            Number of parameters initial conditions to try to find the best solution. Not used if the minimizer being chosen is a global one.
            Default to 1.
        kwargs: dict
            Dictionary of options to pass to the :func:`scipy.optimize.minimize` function.
        """

        alpha, beta, gamma_1 = EnsembleSpreadScalingCorrection._compute_ss_coefficients(observations, predictors, forecast_init_time)
        gamma_2 = gamma_1.copy()

        self.parameters_list = [alpha, beta, gamma_1, gamma_2]

        self._minimize(observations, predictors, num_threads=num_threads, ntrial=ntrial, **kwargs)


class EnsembleNgrCRPSCorrection(EnsembleAbsCRPSCorrection):
    """A postprocessor to correct the ensemble mean and scale the spread of the ensemble. Correct according to the formula:

    .. math::

        \\mathcal{D}^C_{n,m,v} (t) = \\alpha_{v} (t) + \\sum_{p=1}^P \\beta_{p,v} (t) \\, \\mu^{\\rm ens}_{p,n,v} (t) + \\left(\\gamma_{1,v} (t) + \\gamma_{2,v} (t) / \\delta_{0, n, v} (t) \\right) \\, \\bar{\\mathcal{D}}^{\\rm ens}_{0,n,m,v} (t)

    where :math:`\\delta_{0, n, v} (t)` is the average over the ensemble members of the  ensemble members distance (:attr:`~.Data.delta`).
    The parameters :math:`\\alpha_{v} (t)`, :math:`\\beta_{v} (t)`, :math:`\\gamma_{1,v} (t)` and :math:`\\gamma_{2,v} (t)` are optimized at each lead time to minimize
    Non-homogeneous Gaussian Regression (NGR) CRPS (:meth:`~.Data.Abs_Ngr`) of :math:`\\mathcal{D}^C_{n,m,v} (t)` with the observations :math:`\\mathcal{O}`.
    This postprocessor must thus be trained with a training set composed of past ensembles :math:`\\mathcal{D}_{p,n,m,v}` and observations :math:`\\mathcal{O}`.
    If needed, the parameters initial conditions for the minimization are constructed using the values given by the :class:`EnsembleSpreadScalingCorrection`, dependding on
    the minimizer and options being chosen.

    The parameters :math:`\\gamma_{1,v} (t)` and :math:`\\gamma_{2,v} (t)` control respectively the scaling of the ensemble spread and its nudging toward the
    climatological variance for the long lead times.

    Attributes
    ----------
    parameters_list: list(Data)
        The list of training parameters :math:`\\alpha_{v} (t)`, :math:`\\beta_{n,v} (t)`, :math:`\\tau_{v} (t)`, as :class:`.Data` objects.
        This list is empty until the first :meth:`~.BiasCorrection.train` method call.

    Examples
    --------

    .. plot::
        :format: doctest
        :include-source: 1

        >>> from pp_test.test_data import test_sin
        >>> import postprocessors.MBM as MBM
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Loading random sinusoidal observations and reforecasts
        >>> past_observations, reforecasts = test_sin(60., 3., number_of_observation=200)
        >>> past_observations.index_shape
        (1, 200, 1, 1, 21)
        >>> reforecasts.index_shape
        (1, 200, 10, 1, 21)
        >>>
        >>> # Training the postprocessor
        >>> postprocessor = MBM.EnsembleNgrCRPSCorrection()
        >>> postprocessor.train(past_observations, reforecasts, ntrial=10)
        >>>
        >>> # Loading new random sinusoidal observations and forecasts
        >>> observations, forecasts = test_sin(60., 3.)
        >>> corrected_forecasts = postprocessor(forecasts)  # Correcting the new forecasts
        >>>
        >>> # Plotting the forecasts
        >>> ax = observations.plot(global_label='observations', color='tab:green')
        >>> ax = forecasts.plot(ax=ax, global_label='raw forecasts', mfc=None, mec='tab:blue',
        ... ls='', marker='o', ms=3.0)
        >>> ax = corrected_forecasts.plot(ax=ax, global_label='corrected forecasts', mfc=None,
        ... mec='tab:orange', ls='', marker='o', ms=3.0)
        >>> t = ax.set_ylabel('temperature [C°]')
        >>> t = ax.set_xlabel('date')
        >>> t = ax.legend()
        >>> t = ax.set_title('Example of corrected forecasts')
        >>>
        >>> # Plotting the postprocessing parameters
        >>> ax = postprocessor.plot_parameters()
        >>> t = ax.set_title('Postprocessing parameters')
        >>>
        >>> # Computing the CRPS score
        >>> raw_crps = forecasts.CRPS(observations)
        >>> corrected_crps = corrected_forecasts.CRPS(observations)
        >>>
        >>> # Plotting the CRPS score
        >>> ax = raw_crps.plot(timestamps=postprocessor.parameters_list[0].timestamps[0, 0],
        ... global_label='Raw forecast CRPS', color='tab:blue')
        >>> ax = corrected_crps.plot(timestamps=postprocessor.parameters_list[0].timestamps[0, 0], ax=ax,
        ... global_label='Corrected forecast CRPS', color='tab:orange')
        >>> t = ax.set_ylabel('CRPS [C°]')
        >>> t = ax.set_xlabel('time [hour]')
        >>> t = ax.set_title('CRPS score')
        >>> t = ax.legend()
        >>> plt.show()

    """

    def __init__(self):

        EnsembleAbsCRPSCorrection.__init__(self)

    @staticmethod
    def _corrected_forecast_mod(predictors, parameters_list):
        cem = predictors.centered_ensemble.data[0]
        res = parameters_list[0].data
        res = res + np.sum(parameters_list[1].data * predictors.ensemble_mean.data, axis=0)[np.newaxis, ...]
        res = res + np.sqrt(parameters_list[2].data**2 + parameters_list[3].data**2 / predictors.ensemble_var.data[0]) * cem[np.newaxis, ...]
        if predictors.timestamps is not None:
            timestamps = predictors.timestamps[0][np.newaxis, ...]
        else:
            timestamps = None
        return Data(res, metadata=predictors.metadata, timestamps=timestamps, dtype=predictors.dtype)

    @staticmethod
    def _atomic_corrected_forecast(params, predictors):
        cem = predictors.centered_ensemble.data[0]
        res = params[0]
        res = res + np.sum((params[1:-2] * predictors.ensemble_mean.data.T).T, axis=0)[np.newaxis, ...]
        res = res + np.sqrt(params[-2]**2 + params[-1]**2 / predictors.ensemble_var.data[0]) * cem[np.newaxis, ...]
        return Data(res, metadata=False, timestamps=False, dtype=predictors.dtype)

    @staticmethod
    def _atomic_crps(params, observations, predictors):
        corrected = EnsembleNgrCRPSCorrection._atomic_corrected_forecast(params, predictors)
        crps = corrected.Ngr_CRPS(observations)
        return np.squeeze(crps.data)


def _compute(ls):
    return ls[0], ls[1](*ls[2])
