import random
from typing import Dict
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import BaseModel
from ch8.dataset import get_weather_dataset
from ch8.custom_model import CustomModel


class ForecastingModel(BaseModel):

    def __init__(self, n_inp, l_1, l_2, conv1_out,
                 conv1_kernel, conv2_kernel, drop1 = 0, **kwargs):
        # saves arguments in signature to `.hparams` attribute,
        # mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__,
        #  mandatory call - do not skip this
        super().__init__(**kwargs)
        self.network = CustomModel(
            n_inp = self.hparams.n_inp,
            l_1 = self.hparams.l_1,
            l_2 = self.hparams.l_2,
            conv1_out = self.hparams.conv1_out,
            conv1_kernel = self.hparams.conv1_kernel,
            conv2_kernel = self.hparams.conv2_kernel,
            drop1 = self.hparams.drop1
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x is a batch generated based on the TimeSeriesDataset
        network_input = x["encoder_cont"].squeeze(-1)
        prediction = self.network(network_input)

        # We need to return a dictionary
        # that at least contains the prediction and the target_scale.
        # The parameter can be directly forwarded from the input.
        return dict(prediction = prediction, target_scale = x["target_scale"])

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        new_kwargs = {
            "n_inp": dataset.max_encoder_length
        }
        # use to pass real hyperparameters and override defaults set by dataset
        new_kwargs.update(kwargs)
        # example for dataset validation
        assert dataset.max_prediction_length == dataset.min_prediction_length,\
            "Decoder only supports a fixed length"
        assert dataset.min_encoder_length == dataset.max_encoder_length,\
            "Encoder only supports a fixed length"

        return super().from_dataset(dataset, **new_kwargs)


if __name__ == '__main__':

    random.seed(1)
    torch.manual_seed(1)

    dataset = get_weather_dataset()
    model = net = ForecastingModel.from_dataset(
        dataset = dataset,
        l_1 = 400,
        l_2 = 48,
        conv1_out = 6,
        conv1_kernel = 36,
        conv2_kernel = 12,
        drop1 = .1
    )

    print('Custom model summary:')
    print(model.summarize("full"))  # print model summary
    print('==============')

    print('Custom model hyper-parameters:')
    print(model.hparams)
    print('==============')

    print('Custom model forward method execution:')
    dataloader = dataset.to_dataloader(batch_size = 8)
    x, y = next(iter(dataloader))
    p = model(x)
    print(p)
    print('==============')
