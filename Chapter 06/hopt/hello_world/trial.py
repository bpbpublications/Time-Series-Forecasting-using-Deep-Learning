import nni

params = nni.get_next_parameter()

x = params['x']
y = params['y']
z = params['z']

metric = x + y + z

nni.report_final_result(metric)
