import datetime
import os
import datetime as dt
import torch
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ch7.covid.dataset import get_df_complete
from ch7.covid.model.model import EncoderDecoder

dir_path = os.path.dirname(os.path.realpath(__file__))

model_name = 'best_model'
model_params = {
    'hidden_size':    32,
    'hidden_dl_size': 12,
    'input_size':     3,
    'output_size':    1
}

model = EncoderDecoder(**model_params)
model.load_state_dict(torch.load(f'{dir_path}/data/{model_name}.pth'))
model.eval()
from_date = '2020-10-04'
to_date = '2021-04-01'
date_fmt = '%Y-%m-%d'
country = 'Austria'
df = get_df_complete()
au_ts_df = df[df['Country'] == country]['Confirmed'].diff().dropna()
ts = au_ts_df[from_date:to_date].values
test = ts[:120]
max_test = max(test)
test = test / max_test
test_hp_cycle, test_hp_trend = sm.tsa.filters.hpfilter(test)
test_cf_cycle, test_cf_trend = sm.tsa.filters.cffilter(test)
X = []
for i in range(len(test)):
    X.append([test[i], test_hp_trend[i], test_cf_cycle[i]])

x = torch.tensor([X]).float().transpose(0, 1)
model.eval()
predicted = model.predict(x, 60)

in_seq = [e * max_test for e in x[:, -1, 0].view(-1).tolist()]
target_seq = list(ts[120:])
pred_seq = [e * max_test for e in predicted[:, -1, 0].view(-1).tolist()]
x_axis = range(len(in_seq) + len(pred_seq))
start_date = datetime.datetime.strptime(from_date, date_fmt)
end_date = start_date + datetime.timedelta(days = len(in_seq))
prediction_date = start_date + datetime.timedelta(days = len(in_seq) + len(pred_seq))
date_list = mdates.drange(start_date, prediction_date, dt.timedelta(days = 1))

plt.title(f'Prediction for next 60 days')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 35))
plt.plot(date_list[:], in_seq + target_seq, color = 'blue')
plt.plot(date_list[len(in_seq):],
         pred_seq,
         label = 'Model prediction',
         color = 'orange',
         linewidth = 3)
plt.vlines(end_date, 0, max_test, color = 'grey')
plt.legend(loc = "upper right")
plt.gcf().autofmt_xdate()
plt.show()
