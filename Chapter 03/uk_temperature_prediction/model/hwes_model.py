from statsmodels.tsa.holtwinters import ExponentialSmoothing
import torch

class HwesPredictor(torch.nn.Module):

    def forward(self, x):
        last_values = []
        for r in x.tolist():
            model = ExponentialSmoothing(r,
                                         trend = None,
                                         seasonal = "add",
                                         seasonal_periods = 12
                                         )
            results = model.fit()
            forecast = results.forecast()
            last_values.append([forecast[0]])
        return torch.tensor(data = last_values)
