import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('log_85.csv', delimiter=';', skiprows=1)
input1 = data[:,0]
output = data[:,1]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

x = np.linspace(-6,6,1000)

ax.plot(input1, data[:,1], color='red', linestyle='solid')
ax.plot(input1, data[:,2], color='blue', linestyle='solid')
ax.plot(input1, data[:,3], color='yellow', linestyle='solid')

ax.set_title('First line plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
#plt.show()
fig.savefig('log_85.png')

