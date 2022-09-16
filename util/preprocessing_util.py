import numpy as np

epsilon = 1.0e-08


class DataNormalizer:
    def __init__(self):
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None
        self.lambda_j = None

    def normalize_all(self, x, y, dydx=None):
        if dydx is None:
            return self.scale_x(x), self.scale_y(y)
        else:
            return self.scale_x(x), self.scale_y(y), self.scale_dy_dx(dydx)[0]

    def scale_x(self, x):
        return (x - self._x_mean) / self._x_std

    def unscale_x(self, x_scaled):
        return self._x_mean + self._x_std * x_scaled

    def scale_y(self, y):
        return (y - self._y_mean) / self._y_std

    def unscale_y(self, y_scaled):
        return self._y_mean + self._y_std * y_scaled

    def scale_dy_dx(self, dy_dx):
        dy_dx_scaled = dy_dx / self._y_std.reshape(-1, 1).repeat(axis=1, repeats=self.input_dimension) * self._x_std
        # weights of derivatives in cost function = (quad) mean size
        lambda_j = 1.0 / (np.sqrt((dy_dx_scaled**2).mean(axis=0)) + epsilon)
        return dy_dx_scaled, lambda_j

    def unscale_dy_dx(self, dy_dx_scaled):
        return self._y_std / self._x_std * dy_dx_scaled

    @property
    def input_dimension(self):
        return self._x_mean.shape[0]

    @property
    def output_dimension(self):
        return self._y_mean.shape[0]

    def initialize_with_data(self, x_raw, y_raw, dydx_raw=None, crop=None):
        # crop dataset
        m = crop if crop is not None else x_raw.shape[0]
        x_cropped = x_raw[:m]
        y_cropped = y_raw[:m]
        dycropped_dxcropped = dydx_raw[:m] if dydx_raw is not None else None

        self._x_mean = x_cropped.mean(axis=0)
        self._x_std = x_cropped.std(axis=0) + epsilon

        self._y_mean = y_cropped.mean(axis=0)
        self._y_std = y_cropped.std(axis=0) + epsilon

        # normalize derivatives too
        if dycropped_dxcropped is not None:
            _, lambda_j = self.scale_dy_dx(dycropped_dxcropped)
        else:
            lambda_j = None
        self.lambda_j = lambda_j
