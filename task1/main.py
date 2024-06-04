import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def generate_samples(mu1, sigma1, mu2, sigma2, num_samples):
    X = np.random.normal(mu1, np.sqrt(sigma1), num_samples)
    Y = np.random.normal(mu2, np.sqrt(sigma2), num_samples)
    return X, Y


def estimate_tau(X, Y, mu1, mu2, sigma1, sigma2, num_samples1, num_samples2):
    sigma_squared = sigma1 / num_samples1 + sigma2 / num_samples2
    tau_hat = (np.mean(X) - np.mean(Y) - (mu1 - mu2)) / np.sqrt(sigma_squared)
    return tau_hat


def find_quantiles(alpha):
    z_alpha = norm.ppf(1 - alpha)
    return z_alpha


sample_size = 1000

num_samples1 = 25
num_samples2 = 10000

mu1 = 2
mu2 = 1
sigma1 = 1
sigma2 = 0.5

alpha = 0.05

variance_diff = sigma1 / num_samples1 + sigma2 / num_samples2

z_alpha = find_quantiles(alpha)

tau_true = mu1 - mu2
tau_hat1_list = []
tau_hat2_list = []
interval1_list = []
interval2_list = []
count_interval1 = 0
count_interval2 = 0

for i in range(sample_size):
    X1, Y1 = generate_samples(mu1, sigma1, mu2, sigma2, num_samples1)
    X2, Y2 = generate_samples(mu1, sigma1, mu2, sigma2, num_samples2)

    tau_hat1 = estimate_tau(X1, Y1, mu1, mu2, sigma1, sigma2, num_samples1, num_samples2)
    tau_hat2 = estimate_tau(X2, Y2, mu1, mu2, sigma1, sigma2, num_samples1, num_samples2)

    interval1 = (tau_hat1 - z_alpha / 2 * np.sqrt(variance_diff), tau_hat1 + z_alpha / 2 * np.sqrt(variance_diff))
    interval2 = (tau_hat2 - z_alpha / 2 * np.sqrt(variance_diff), tau_hat2 + z_alpha / 2 * np.sqrt(variance_diff))

    if interval1[0] <= (mu1 - mu2) <= interval1[1]:
        count_interval1 += 1
    if interval2[0] <= (mu1 - mu2) <= interval2[1]:
        count_interval2 += 1

    tau_hat1_list.append(tau_hat1)
    tau_hat2_list.append(tau_hat2)
    interval1_list.append(interval1)
    interval2_list.append(interval2)

print(f"Для выборки объема {num_samples1}:")
print(f"Доля не попаданий в доверительный интервал: {count_interval1 / sample_size}")

print(f"Для выборки объема {num_samples2}:")
print(f"Доля не попаданий в доверительный интервал: {count_interval2 / sample_size}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(sample_size), tau_hat1_list, label='Оценка параметра', color='b')
plt.hlines(tau_true, 0, sample_size, label='Истинное значение', color='r', linestyle='--')
plt.fill_between(range(sample_size), [i[0] for i in interval1_list], [i[1] for i in interval1_list], color='b',
                 alpha=0.2)
plt.xlabel('Повторения')
plt.ylabel('Оценка параметра')
plt.title(f'Выборка объема {num_samples1}')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(sample_size), tau_hat2_list, label='Оценка параметра', color='g')
plt.hlines(tau_true, 0, sample_size, label='Истинное значение', color='r', linestyle='--')
plt.fill_between(range(sample_size), [i[0] for i in interval2_list], [i[1] for i in interval2_list], color='g',
                 alpha=0.2)
plt.xlabel('Повторения')
plt.ylabel('Оценка параметра')
plt.title(f'Выборка объема {num_samples2}')
plt.legend()

plt.tight_layout()
plt.show()
