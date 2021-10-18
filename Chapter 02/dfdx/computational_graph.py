from ch2.dfdx.function import get_function
from torchviz import make_dot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

f, params = get_function(2, 4, 3, 5)

make_dot(f, params).render("f_torchviz")

img = mpimg.imread('f_torchviz.png')
plt.xticks([])
plt.yticks([])
plt.imshow(img)
plt.show()
