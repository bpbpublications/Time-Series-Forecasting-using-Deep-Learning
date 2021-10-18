from ch2.dfdx.function import get_function

f, params = get_function(2, 4, 3, 5)

f.backward()

print(f'df_dx1: {params["x1"].grad}')
print(f'df_dx2: {params["x2"].grad}')
print(f'df_dx3: {params["x3"].grad}')
print(f'df_dx4: {params["x4"].grad}')
