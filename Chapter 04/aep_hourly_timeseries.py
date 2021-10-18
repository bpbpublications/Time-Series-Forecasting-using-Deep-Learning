import matplotlib.pyplot as plt
from ch4.training_datasets import get_aep_timeseries

plt.title('AEP Hourly')
plt.plot(get_aep_timeseries()[:500])
plt.show()
