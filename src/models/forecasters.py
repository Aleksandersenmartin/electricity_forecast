"""
Forecasting toolbox for electricity price time series.

Provides a consistent API for multiple forecasting methods so notebooks
can swap methods with minimal code changes. Each forecaster wraps a
different statistical/ML approach.

Usage:
    from src.models.forecasters import NaiveForecaster, ARIMAForecaster

    model = ARIMAForecaster(name="ARIMA", horizon=168)
    model.fit(y_train)
    preds = model.predict(steps=168)
"""

import logging
import time
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseForecaster(ABC):
    """Abstract base class for all forecasters.

    Provides a consistent interface: fit(), predict(), fit_predict().
    Subclasses must implement _fit() and _predict().

    Args:
        name: Human-readable name for this forecaster.
        horizon: Default forecast horizon in steps.
        frequency: Pandas frequency string ("h", "D", "W").
    """

    def __init__(self, name: str, horizon: int, frequency: str = "h") -> None:
        self.name = name
        self.horizon = horizon
        self.frequency = frequency
        self.model_: object | None = None
        self.fit_time_seconds: float = 0.0
        self._y_train: pd.Series | None = None
        self._X_train: pd.DataFrame | None = None
        self._tz: str | None = None

    def fit(
        self,
        y_train: pd.Series,
        X_train: pd.DataFrame | None = None,
    ) -> "BaseForecaster":
        """Fit the forecaster on training data.

        Args:
            y_train: Target time series with DatetimeIndex.
            X_train: Optional exogenous features (same index as y_train).

        Returns:
            self (for chaining).
        """
        # Store timezone, then strip (statsmodels can choke on tz-aware)
        if hasattr(y_train.index, "tz") and y_train.index.tz is not None:
            self._tz = str(y_train.index.tz)
            y_train = y_train.copy()
            y_train.index = y_train.index.tz_localize(None)
            if X_train is not None:
                X_train = X_train.copy()
                X_train.index = X_train.index.tz_localize(None)

        self._y_train = y_train
        self._X_train = X_train

        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fit(y_train, X_train)
        self.fit_time_seconds = round(time.time() - t0, 2)

        logger.info(
            "%s fit complete: %d samples in %.1f seconds",
            self.name, len(y_train), self.fit_time_seconds,
        )
        return self

    def predict(
        self,
        steps: int | None = None,
        X_future: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Generate forecasts for the given number of steps.

        Args:
            steps: Number of steps to forecast. Defaults to self.horizon.
            X_future: Optional exogenous features for future periods.

        Returns:
            Series with DatetimeIndex of predictions.
        """
        if steps is None:
            steps = self.horizon

        if X_future is not None and hasattr(X_future.index, "tz") and X_future.index.tz is not None:
            X_future = X_future.copy()
            X_future.index = X_future.index.tz_localize(None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = self._predict(steps, X_future)

        # Restore timezone (handle DST gaps and overlaps)
        if self._tz is not None and hasattr(preds.index, "tz") and preds.index.tz is None:
            preds.index = preds.index.tz_localize(
                self._tz,
                nonexistent="shift_forward",
                ambiguous="NaT",
            )
            # Forward-fill any NaT from ambiguous (autumn fall-back) times
            if preds.index.isna().any():
                preds = preds[preds.index.notna()]

        preds.name = self.name
        return preds

    def fit_predict(
        self,
        y_train: pd.Series,
        steps: int | None = None,
        X_train: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Convenience: fit then predict in one call.

        Args:
            y_train: Target time series.
            steps: Forecast steps.
            X_train: Optional exogenous features for training.
            X_future: Optional exogenous features for prediction.

        Returns:
            Series of predictions.
        """
        self.fit(y_train, X_train)
        return self.predict(steps, X_future)

    @abstractmethod
    def _fit(self, y_train: pd.Series, X_train: pd.DataFrame | None) -> None:
        """Subclass-specific fitting logic."""

    @abstractmethod
    def _predict(self, steps: int, X_future: pd.DataFrame | None) -> pd.Series:
        """Subclass-specific prediction logic."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', horizon={self.horizon})"


# ---------------------------------------------------------------------------
# 1. Naive Forecaster
# ---------------------------------------------------------------------------

class NaiveForecaster(BaseForecaster):
    """Naive baseline: repeat values from a fixed lag.

    For hourly data, shift=168 copies the same hour from last week.
    For daily data, shift=7 copies the same day from last week.
    For weekly data, shift=52 copies the same week from last year.

    Args:
        name: Forecaster name.
        horizon: Forecast horizon.
        frequency: Pandas frequency string.
        lag: Number of steps to shift. Auto-detected from frequency
            if not provided (168 for hourly, 7 for daily, 52 for weekly).
    """

    def __init__(
        self,
        name: str = "Naive (same hour last week)",
        horizon: int = 168,
        frequency: str = "h",
        lag: int | None = None,
    ) -> None:
        super().__init__(name, horizon, frequency)
        if lag is not None:
            self.lag = lag
        elif frequency == "h":
            self.lag = 168  # same hour last week
        elif frequency == "D":
            self.lag = 7    # same day last week
        elif frequency == "W":
            self.lag = 52   # same week last year
        else:
            self.lag = 168

    def _fit(self, y_train: pd.Series, X_train: pd.DataFrame | None) -> None:
        """Store training data for shift-based prediction."""
        # Nothing to fit — just store the tail for prediction
        self._tail = y_train.iloc[-self.lag:]

    def _predict(self, steps: int, X_future: pd.DataFrame | None) -> pd.Series:
        """Repeat the lagged values to cover the forecast horizon."""
        tail = self._tail
        last_time = self._y_train.index[-1]

        # Build forecast index
        forecast_index = pd.date_range(
            start=last_time + pd.tseries.frequencies.to_offset(self.frequency),
            periods=steps,
            freq=self.frequency,
        )

        # Tile the tail values to cover all requested steps
        n_repeats = (steps // self.lag) + 2
        tiled = np.tile(tail.values, n_repeats)[:steps]

        return pd.Series(tiled, index=forecast_index, name=self.name)


# ---------------------------------------------------------------------------
# 2. ARIMA Forecaster (auto_arima)
# ---------------------------------------------------------------------------

class ARIMAForecaster(BaseForecaster):
    """ARIMA via pmdarima's auto_arima (automatic order selection).

    Uses AIC-based stepwise search to find the best (p,d,q) order.

    Args:
        name: Forecaster name.
        horizon: Forecast horizon.
        frequency: Pandas frequency string.
        max_p: Maximum AR order.
        max_d: Maximum differencing order.
        max_q: Maximum MA order.
        stepwise: Use stepwise search (faster).
    """

    def __init__(
        self,
        name: str = "ARIMA (auto)",
        horizon: int = 168,
        frequency: str = "h",
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        stepwise: bool = True,
    ) -> None:
        super().__init__(name, horizon, frequency)
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.stepwise = stepwise

    def _fit(self, y_train: pd.Series, X_train: pd.DataFrame | None) -> None:
        """Fit auto_arima to find best (p,d,q) and fit the model."""
        import pmdarima as pm

        self.model_ = pm.auto_arima(
            y_train.values,
            max_p=self.max_p,
            max_d=self.max_d,
            max_q=self.max_q,
            stepwise=self.stepwise,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        logger.info("%s selected order: %s", self.name, self.model_.order)

    def _predict(self, steps: int, X_future: pd.DataFrame | None) -> pd.Series:
        """Forecast using the fitted ARIMA model."""
        last_time = self._y_train.index[-1]
        forecast_index = pd.date_range(
            start=last_time + pd.tseries.frequencies.to_offset(self.frequency),
            periods=steps,
            freq=self.frequency,
        )

        preds = self.model_.predict(n_periods=steps)
        return pd.Series(preds, index=forecast_index, name=self.name)


# ---------------------------------------------------------------------------
# 3. SARIMAX Forecaster
# ---------------------------------------------------------------------------

class SARIMAXForecaster(BaseForecaster):
    """SARIMAX with seasonal order via auto_arima.

    Uses pmdarima's auto_arima with seasonal=True to find the best
    (p,d,q)(P,D,Q,m) order. Supports exogenous variables.

    For hourly data with seasonal_period=24, subsamples training data
    to the last max_train_size observations for tractable fitting.

    Args:
        name: Forecaster name.
        horizon: Forecast horizon.
        frequency: Pandas frequency string.
        seasonal_period: Seasonal period (24 for daily cycle, 7 for weekly).
        max_train_size: Max training samples to use (for speed).
        max_p: Maximum AR order.
        max_q: Maximum MA order.
        max_P: Maximum seasonal AR order.
        max_Q: Maximum seasonal MA order.
    """

    def __init__(
        self,
        name: str = "SARIMAX",
        horizon: int = 168,
        frequency: str = "h",
        seasonal_period: int = 24,
        max_train_size: int = 4000,
        max_p: int = 3,
        max_q: int = 3,
        max_P: int = 2,
        max_Q: int = 2,
    ) -> None:
        super().__init__(name, horizon, frequency)
        self.seasonal_period = seasonal_period
        self.max_train_size = max_train_size
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q

    def _fit(self, y_train: pd.Series, X_train: pd.DataFrame | None) -> None:
        """Fit SARIMAX via auto_arima with seasonal component."""
        import pmdarima as pm

        # Subsample for speed
        y = y_train
        X = X_train
        if len(y) > self.max_train_size:
            y = y.iloc[-self.max_train_size:]
            if X is not None:
                X = X.iloc[-self.max_train_size:]
            logger.info(
                "%s: subsampled training from %d to %d observations",
                self.name, len(y_train), len(y),
            )

        exog = X.values if X is not None else None

        self.model_ = pm.auto_arima(
            y.values,
            exogenous=exog,
            seasonal=True,
            m=self.seasonal_period,
            max_p=self.max_p,
            max_q=self.max_q,
            max_P=self.max_P,
            max_Q=self.max_Q,
            max_d=2,
            max_D=1,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        logger.info(
            "%s selected order: %s seasonal_order: %s",
            self.name, self.model_.order, self.model_.seasonal_order,
        )

    def _predict(self, steps: int, X_future: pd.DataFrame | None) -> pd.Series:
        """Forecast using the fitted SARIMAX model."""
        last_time = self._y_train.index[-1]
        forecast_index = pd.date_range(
            start=last_time + pd.tseries.frequencies.to_offset(self.frequency),
            periods=steps,
            freq=self.frequency,
        )

        exog_future = None
        if X_future is not None:
            exog_future = X_future.values[:steps]

        preds = self.model_.predict(n_periods=steps, exogenous=exog_future)
        return pd.Series(preds, index=forecast_index, name=self.name)


# ---------------------------------------------------------------------------
# 4. STL + ARIMA Forecaster
# ---------------------------------------------------------------------------

class STLARIMAForecaster(BaseForecaster):
    """STL decomposition followed by ARIMA on the residual component.

    Decomposes the series into trend + seasonal + residual using STL.
    Fits ARIMA on residuals. Forecasts by extrapolating trend,
    repeating the seasonal cycle, and forecasting residuals.

    Args:
        name: Forecaster name.
        horizon: Forecast horizon.
        frequency: Pandas frequency string.
        seasonal_period: STL seasonal period (24 for daily cycle).
        max_train_size: Max training samples for ARIMA fit.
    """

    def __init__(
        self,
        name: str = "STL + ARIMA",
        horizon: int = 168,
        frequency: str = "h",
        seasonal_period: int = 24,
        max_train_size: int = 4000,
    ) -> None:
        super().__init__(name, horizon, frequency)
        self.seasonal_period = seasonal_period
        self.max_train_size = max_train_size
        self._seasonal_cycle: np.ndarray | None = None
        self._trend_last: float = 0.0
        self._trend_slope: float = 0.0

    def _fit(self, y_train: pd.Series, X_train: pd.DataFrame | None) -> None:
        """Decompose with STL, then fit ARIMA on residuals."""
        import pmdarima as pm
        from statsmodels.tsa.seasonal import STL

        # Subsample for speed
        y = y_train
        if len(y) > self.max_train_size:
            y = y.iloc[-self.max_train_size:]

        # STL decomposition
        stl = STL(y, period=self.seasonal_period, robust=True)
        result = stl.fit()

        self._seasonal_cycle = result.seasonal.values[-self.seasonal_period:]
        self._trend_last = result.trend.values[-1]

        # Estimate trend slope from last 168 hours
        trend_tail = result.trend.dropna().values[-168:]
        if len(trend_tail) > 1:
            self._trend_slope = (trend_tail[-1] - trend_tail[0]) / len(trend_tail)
        else:
            self._trend_slope = 0.0

        # ARIMA on residuals
        residuals = result.resid
        self.model_ = pm.auto_arima(
            residuals.values,
            max_p=3,
            max_d=1,
            max_q=3,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        logger.info("%s: STL period=%d, ARIMA order=%s", self.name, self.seasonal_period, self.model_.order)

    def _predict(self, steps: int, X_future: pd.DataFrame | None) -> pd.Series:
        """Reconstruct forecast = trend + seasonal + residual forecast."""
        last_time = self._y_train.index[-1]
        forecast_index = pd.date_range(
            start=last_time + pd.tseries.frequencies.to_offset(self.frequency),
            periods=steps,
            freq=self.frequency,
        )

        # Trend: linear extrapolation
        trend = self._trend_last + self._trend_slope * np.arange(1, steps + 1)

        # Seasonal: tile the last cycle
        n_repeats = (steps // self.seasonal_period) + 2
        seasonal = np.tile(self._seasonal_cycle, n_repeats)[:steps]

        # Residual: ARIMA forecast
        resid_forecast = self.model_.predict(n_periods=steps)

        preds = trend + seasonal + resid_forecast
        return pd.Series(preds, index=forecast_index, name=self.name)


# ---------------------------------------------------------------------------
# 5. ETS (Exponential Smoothing) Forecaster
# ---------------------------------------------------------------------------

class ETSForecaster(BaseForecaster):
    """Holt-Winters Exponential Smoothing.

    Uses statsmodels ExponentialSmoothing with additive trend and
    additive seasonality.

    Args:
        name: Forecaster name.
        horizon: Forecast horizon.
        frequency: Pandas frequency string.
        seasonal_period: Seasonal period (24 for daily cycle).
        trend: Trend type ("add", "mul", or None).
        seasonal: Seasonal type ("add", "mul", or None).
        max_train_size: Max training samples.
    """

    def __init__(
        self,
        name: str = "ETS (Holt-Winters)",
        horizon: int = 168,
        frequency: str = "h",
        seasonal_period: int = 24,
        trend: str | None = "add",
        seasonal: str | None = "add",
        max_train_size: int = 4000,
    ) -> None:
        super().__init__(name, horizon, frequency)
        self.seasonal_period = seasonal_period
        self.trend = trend
        self.seasonal = seasonal
        self.max_train_size = max_train_size

    def _fit(self, y_train: pd.Series, X_train: pd.DataFrame | None) -> None:
        """Fit Exponential Smoothing model."""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        y = y_train
        if len(y) > self.max_train_size:
            y = y.iloc[-self.max_train_size:]

        self.model_ = ExponentialSmoothing(
            y,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_period,
            initialization_method="estimated",
        ).fit(optimized=True)

        logger.info("%s: AIC=%.1f", self.name, self.model_.aic)

    def _predict(self, steps: int, X_future: pd.DataFrame | None) -> pd.Series:
        """Forecast using the fitted ETS model."""
        last_time = self._y_train.index[-1]
        forecast_index = pd.date_range(
            start=last_time + pd.tseries.frequencies.to_offset(self.frequency),
            periods=steps,
            freq=self.frequency,
        )

        preds = self.model_.forecast(steps)
        return pd.Series(preds.values, index=forecast_index, name=self.name)


# ---------------------------------------------------------------------------
# 6. Prophet Forecaster
# ---------------------------------------------------------------------------

class ProphetForecaster(BaseForecaster):
    """Facebook Prophet with daily/weekly seasonality and optional regressors.

    Prophet expects a DataFrame with columns 'ds' and 'y'. Additional
    exogenous columns are added as regressors.

    Args:
        name: Forecaster name.
        horizon: Forecast horizon.
        frequency: Pandas frequency string.
        daily_seasonality: Enable daily seasonality (period=24h).
        weekly_seasonality: Enable weekly seasonality (period=7d).
        yearly_seasonality: Enable yearly seasonality.
        regressor_columns: List of exogenous column names to include.
    """

    def __init__(
        self,
        name: str = "Prophet",
        horizon: int = 168,
        frequency: str = "h",
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = False,
        regressor_columns: list[str] | None = None,
    ) -> None:
        super().__init__(name, horizon, frequency)
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.regressor_columns = regressor_columns or []

    def _fit(self, y_train: pd.Series, X_train: pd.DataFrame | None) -> None:
        """Fit Prophet model."""
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("prophet not installed. Run: pip install prophet")

        # Suppress Prophet's verbose logging
        prophet_logger = logging.getLogger("prophet")
        prophet_level = prophet_logger.level
        prophet_logger.setLevel(logging.WARNING)
        cmdstanpy_logger = logging.getLogger("cmdstanpy")
        cmdstanpy_level = cmdstanpy_logger.level
        cmdstanpy_logger.setLevel(logging.WARNING)

        try:
            prophet_df = pd.DataFrame({
                "ds": y_train.index,
                "y": y_train.values,
            })

            m = Prophet(
                daily_seasonality=self.daily_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                yearly_seasonality=self.yearly_seasonality,
            )

            # Add regressors
            if X_train is not None and self.regressor_columns:
                for col in self.regressor_columns:
                    if col in X_train.columns:
                        m.add_regressor(col)
                        prophet_df[col] = X_train[col].values

            m.fit(prophet_df)
            self.model_ = m
        finally:
            prophet_logger.setLevel(prophet_level)
            cmdstanpy_logger.setLevel(cmdstanpy_level)

    def _predict(self, steps: int, X_future: pd.DataFrame | None) -> pd.Series:
        """Forecast using the fitted Prophet model."""
        last_time = self._y_train.index[-1]
        forecast_index = pd.date_range(
            start=last_time + pd.tseries.frequencies.to_offset(self.frequency),
            periods=steps,
            freq=self.frequency,
        )

        future_df = pd.DataFrame({"ds": forecast_index})

        # Add regressor values for future
        if X_future is not None and self.regressor_columns:
            for col in self.regressor_columns:
                if col in X_future.columns:
                    future_df[col] = X_future[col].values[:steps]

        result = self.model_.predict(future_df)
        preds = result["yhat"].values

        return pd.Series(preds, index=forecast_index, name=self.name)


# ---------------------------------------------------------------------------
# 7. VAR Forecaster (multivariate)
# ---------------------------------------------------------------------------

class VARForecaster(BaseForecaster):
    """Vector Autoregression for multivariate time series.

    Unlike other forecasters, VAR takes multiple series simultaneously
    and captures cross-series dependencies.

    Args:
        name: Forecaster name.
        horizon: Forecast horizon.
        frequency: Pandas frequency string.
        maxlags: Maximum number of lags for lag selection. None = auto.
        max_train_size: Max training samples.
    """

    def __init__(
        self,
        name: str = "VAR",
        horizon: int = 168,
        frequency: str = "h",
        maxlags: int | None = None,
        max_train_size: int = 4000,
    ) -> None:
        super().__init__(name, horizon, frequency)
        self.maxlags = maxlags
        self.max_train_size = max_train_size
        self._columns: list[str] = []

    def fit_multi(
        self,
        y_train: pd.DataFrame,
    ) -> "VARForecaster":
        """Fit on a multivariate DataFrame (one column per target).

        Args:
            y_train: DataFrame with DatetimeIndex and one column per variable.

        Returns:
            self.
        """
        # Store timezone
        if hasattr(y_train.index, "tz") and y_train.index.tz is not None:
            self._tz = str(y_train.index.tz)
            y_train = y_train.copy()
            y_train.index = y_train.index.tz_localize(None)

        self._y_train = y_train.iloc[:, 0]  # first column for index generation
        self._columns = list(y_train.columns)

        y = y_train
        if len(y) > self.max_train_size:
            y = y.iloc[-self.max_train_size:]

        t0 = time.time()
        self._fit_var(y)
        self.fit_time_seconds = round(time.time() - t0, 2)

        logger.info(
            "%s fit: %d variables, %d samples in %.1f seconds",
            self.name, len(self._columns), len(y), self.fit_time_seconds,
        )
        return self

    def _fit_var(self, y: pd.DataFrame) -> None:
        """Internal VAR fitting."""
        from statsmodels.tsa.api import VAR

        model = VAR(y.dropna())
        self.model_ = model.fit(maxlags=self.maxlags, ic="aic")
        logger.info("%s selected lag order: %d", self.name, self.model_.k_ar)

    def _fit(self, y_train: pd.Series, X_train: pd.DataFrame | None) -> None:
        """Single-variable fit fallback — wraps in DataFrame."""
        df = y_train.to_frame()
        if X_train is not None:
            df = pd.concat([df, X_train], axis=1)
        self._columns = list(df.columns)
        self._fit_var(df)

    def _predict(self, steps: int, X_future: pd.DataFrame | None) -> pd.Series:
        """Forecast all variables and return the first column."""
        preds_df = self.predict_multi(steps)
        return preds_df.iloc[:, 0]

    def predict_multi(self, steps: int | None = None) -> pd.DataFrame:
        """Forecast all variables.

        Args:
            steps: Number of steps to forecast.

        Returns:
            DataFrame with one column per variable.
        """
        if steps is None:
            steps = self.horizon

        last_time = self._y_train.index[-1]
        forecast_index = pd.date_range(
            start=last_time + pd.tseries.frequencies.to_offset(self.frequency),
            periods=steps,
            freq=self.frequency,
        )

        # VAR requires lagged values as input
        lag_order = self.model_.k_ar
        y_input = self.model_.endog[-lag_order:]

        preds = self.model_.forecast(y_input, steps=steps)
        result = pd.DataFrame(preds, index=forecast_index, columns=self._columns)

        if self._tz is not None:
            result.index = result.index.tz_localize(
                self._tz,
                nonexistent="shift_forward",
                ambiguous="NaT",
            )
            if result.index.isna().any():
                result = result[result.index.notna()]

        return result


# ---------------------------------------------------------------------------
# 8. Markov Regime Switching Forecaster
# ---------------------------------------------------------------------------

class MarkovSwitchingForecaster(BaseForecaster):
    """Markov Regime Switching model for detecting price regimes.

    Fits a 2-regime model where each regime has its own mean and
    variance. Useful for distinguishing normal vs spike price behavior.

    Args:
        name: Forecaster name.
        horizon: Forecast horizon.
        frequency: Pandas frequency string.
        k_regimes: Number of regimes (default 2).
        max_train_size: Max training samples.
    """

    def __init__(
        self,
        name: str = "Markov Switching",
        horizon: int = 168,
        frequency: str = "h",
        k_regimes: int = 2,
        max_train_size: int = 4000,
    ) -> None:
        super().__init__(name, horizon, frequency)
        self.k_regimes = k_regimes
        self.max_train_size = max_train_size
        self.regime_means_: np.ndarray | None = None
        self.transition_matrix_: np.ndarray | None = None

    def _fit(self, y_train: pd.Series, X_train: pd.DataFrame | None) -> None:
        """Fit Markov Regime Switching model.

        Normalizes the data before fitting to improve numerical stability
        (electricity prices have extreme spikes that can cause SVD failures).
        Regime means are stored in the original scale.
        """
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        y = y_train
        if len(y) > self.max_train_size:
            y = y.iloc[-self.max_train_size:]

        # Normalize to improve numerical stability (price spikes cause SVD issues)
        self._scale_mean = y.mean()
        self._scale_std = y.std()
        y_scaled = (y - self._scale_mean) / self._scale_std

        # Ensure a proper frequency on the index (statsmodels needs this)
        y_scaled = y_scaled.copy()
        y_scaled.index = pd.date_range(
            start=y_scaled.index[0], periods=len(y_scaled), freq=self.frequency,
        )

        self.model_ = MarkovRegression(
            y_scaled,
            k_regimes=self.k_regimes,
            switching_variance=True,
        ).fit(maxiter=500, disp=False)

        # Extract regime means — params may be pd.Series (string index) or ndarray
        params = self.model_.params
        if isinstance(params, pd.Series):
            scaled_means = np.array([
                params[f"const[{i}]"] for i in range(self.k_regimes)
            ])
        else:
            param_names = self.model_.model.param_names
            const_indices = [i for i, n in enumerate(param_names) if n.startswith("const")]
            scaled_means = np.array([params[i] for i in const_indices])

        # Unscale regime means back to original EUR/MWh
        self.regime_means_ = scaled_means * self._scale_std + self._scale_mean

        # Transition matrix is (k, k, 1) — squeeze to 2D for matrix multiplication
        self.transition_matrix_ = self.model_.regime_transition.squeeze()

        logger.info(
            "%s: regime means=%s",
            self.name,
            [round(m, 2) for m in self.regime_means_],
        )

    def _predict(self, steps: int, X_future: pd.DataFrame | None) -> pd.Series:
        """Forecast by propagating regime probabilities forward.

        Uses the transition matrix to evolve regime probabilities,
        then computes the expected price as a probability-weighted
        average of regime means.
        """
        last_time = self._y_train.index[-1]
        forecast_index = pd.date_range(
            start=last_time + pd.tseries.frequencies.to_offset(self.frequency),
            periods=steps,
            freq=self.frequency,
        )

        # Start from filtered regime probabilities at end of training
        smoothed = self.model_.smoothed_marginal_probabilities
        current_probs = smoothed.iloc[-1].values

        preds = np.zeros(steps)
        P = self.transition_matrix_

        for t in range(steps):
            # Expected value = sum of regime means weighted by probabilities
            preds[t] = np.dot(current_probs, self.regime_means_)
            # Propagate regime probabilities
            current_probs = P @ current_probs

        return pd.Series(preds, index=forecast_index, name=self.name)

    def get_regime_probabilities(self) -> pd.DataFrame:
        """Return smoothed regime probabilities for the training period.

        Returns:
            DataFrame with one column per regime, indexed by training timestamps.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self.model_.smoothed_marginal_probabilities
