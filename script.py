
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from symmetry_no.gaussian_random_field import GaussianRF

a_GRF = GaussianRF(dim=2, size=512, alpha=1, sigma=1, exp=True)
a = a_GRF.sample(1).squeeze().real

# a[a>0] = 12
# a[a<=0] = 2

a = a.numpy()
plt.imshow(a)
plt.colorbar()
plt.show()
