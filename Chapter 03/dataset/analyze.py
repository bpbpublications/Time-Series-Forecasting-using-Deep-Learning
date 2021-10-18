import numpy as np
from ch3.uk_temperature_prediction.raw_time_series import raw_time_series

ts = raw_time_series()

print(f'Count: {len(ts)}')
print(f'Max: {np.nanmax(ts)}')
print(f'Min: {np.nanmin(ts)}')
print(f'Avg: {round(np.nanmean(ts), 2)}')
print(f'Median: {round(np.nanmedian(ts), 2)}')
print(f'Std: {round(np.nanstd(ts), 2)}')
print(f'NA values: {np.count_nonzero(np.isnan(ts))}')
