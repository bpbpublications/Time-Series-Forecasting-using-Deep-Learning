from ch2.dfdx.function import get_function
from torch.autograd import grad

f, params = get_function(2, 4, 3, 5)

df_dx1 = grad(outputs = f, inputs = [params['x1']])

print(df_dx1)
