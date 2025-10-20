import numpy as np
import matplotlib.pyplot  as plt

x,y,z = np.loadtxt('circle.dat', unpack=True, usecols=(0,1,2))

plt.minorticks_on()
plt.suptitle("Moto circolare uniforme")
plt.subplot(2,1,1)
plt.plot(y,z, 'k-', label="traiettoria")
plt.axis('equal')
plt.grid()
plt.legend()
plt.subplot(2,1,2)
plt.plot(x,y, 'b-.', label="legge oraria x")
plt.plot(x,z, 'r--', label="legge oraria y")
plt.axis('equal')
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()
