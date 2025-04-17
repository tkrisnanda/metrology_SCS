import numpy as np
import matplotlib.pyplot as plt
path = '/Users/panxiaozhou/Documents/GitHub/Metrology_project/protocol_simulation/panda4/0_plus_2.csv'
file = np.loadtxt(path, delimiter=' ', skiprows=1)

# data shape: (149, 1001), the first column is phase, 1000 repetitions.
data_phase = file[:, 0]
print(data_phase.size)
data_pg0 = file[:, 1:]
data_pg0_avg = np.nanmean(data_pg0, axis=1)  # Since the data_pg0 has nan value, we should use np.nonmean instead of np.mean
plt.figure(figsize = (5,3))
plt.plot(data_phase, data_pg0_avg, '.')
plt.xlim(0, 1.8)
plt.ylim(0, 1)
plt.show()
