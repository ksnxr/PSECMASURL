import matplotlib
import matplotlib.pyplot as plt
import pickle
from settings import number_iterations, log_interval

matplotlib.use('TkAgg')

with open('cache/ac.txt', 'rb') as f:
    AC = pickle.load(f)
with open('cache/ad.txt', 'rb') as f:
    AD = pickle.load(f)
with open('cache/tft.txt', 'rb') as f:
    TFT = pickle.load(f)
with open('cache/rev_tft.txt', 'rb') as f:
    RevTFT = pickle.load(f)
with open('cache/mc.txt', 'rb') as f:
    mc = pickle.load(f)
with open('cache/md.txt', 'rb') as f:
    md = pickle.load(f)
with open('cache/de.txt', 'rb') as f:
    de = pickle.load(f)
with open('cache/ex.txt', 'rb') as f:
    ex = pickle.load(f)
with open('cache/sr.txt', 'rb') as f:
    sr = pickle.load(f)

plt.figure(figsize=(20, 8))
plt.title('Agents')
x = [i for i in range(number_iterations) if i % log_interval == 0]
plt.plot(x, AC, 'g', label="All C")
plt.plot(x, AD, 'c', label="All D")
plt.plot(x, TFT, 'b', label="TFT")
plt.plot(x, RevTFT, 'm', label="Rev TFT")
plt.legend(loc='upper right')
plt.plot()

fig = plt.figure(figsize=(20, 8))
ax1 = fig.add_subplot(111)
x = [_ for _ in range(number_iterations)]
ax1.plot(x, mc, 'g', label="Mutual Cooperation")
ax1.plot(x, md, 'c', label="Mutual Defection")
ax1.plot(x, ex, 'b', label="Exploitation")
ax1.plot(x, de, 'm', label="Deception")
plt.legend(loc='upper right')
ax2 = ax1.twinx()
ax2.plot(x, sr, 'r', label="Societal Reward")
plt.title("Games")
plt.legend(loc='lower right')
plt.show()
