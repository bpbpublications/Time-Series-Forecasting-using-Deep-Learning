from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch


class SarimaxPredictor(torch.nn.Module):

    def forward(self, x):
        last_values = []
        l = x.tolist()
        counter = 0
        for r in l:
            model = SARIMAX(r,
                            order = (1, 1, 1),
                            seasonal_order = (1, 1, 1, 12))
            results = model.fit(disp = 0)
            forecast = results.forecast()
            last_values.append([forecast[0]])
            counter = counter + 1
            print(f'debug: SARIMA calculation {counter} / {len(l)}')

        return torch.tensor(data = last_values)
