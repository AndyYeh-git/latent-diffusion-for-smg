import matplotlib.pyplot as plt

steps = list(range(10, 41, 5))
times = [25.51, 25.82, 26.41, 26.98, 29.60, 32.58, 34.13]

plt.figure(figsize=(6,4))
plt.plot(steps, times, marker='o', linestyle='-')
plt.xlabel("generation steps")
plt.ylabel("s/sample")
plt.title("DPM++ Sampling Steps Inference Time")
plt.grid(True)
plt.show()