import matplotlib.pyplot as plt
from ch4.training_datasets import get_pjme_timeseries

plt.title('PJME Hourly')
plt.plot(get_pjme_timeseries()[:500])
plt.show()
