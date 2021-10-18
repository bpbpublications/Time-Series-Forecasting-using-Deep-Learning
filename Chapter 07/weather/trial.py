import nni

from ch7.weather.train import prepare_model

val = prepare_model(nni.get_next_parameter())
nni.report_final_result(val)
