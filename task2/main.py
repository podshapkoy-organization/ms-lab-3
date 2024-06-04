import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt


def generate_sample(lam, sample_size):
    return expon.rvs(scale=1 / lam, size=sample_size)


def estimate_median(sample):
    return np.median(sample)


def asymptotic_confidence_interval(sample, alpha):
    n = len(sample)
    median = estimate_median(sample)
    z_alpha = expon.ppf(alpha / 2 + 0.5, scale=1) / np.sqrt(n)
    lower_bound = median - z_alpha
    upper_bound = median + z_alpha
    return (lower_bound, upper_bound)


def experiment(lam, sample_size, sample_count, alpha):
    count_covering = 0
    for _ in range(sample_count):
        sample = generate_sample(lam, sample_size)
        interval = asymptotic_confidence_interval(sample, alpha)
        if interval[0] <= lam <= interval[1]:
            count_covering += 1
    return count_covering / sample_count


lam = 1
sample_size1 = 25
sample_size2 = 10000
sample_count = 1000
alpha = 0.05

coverage_probability1 = experiment(lam, sample_size1, sample_count, alpha)
coverage_probability2 = experiment(lam, sample_size2, sample_count, alpha)

print(f"Доля попаданий в асимптотический доверительный интервал(25): {coverage_probability1}")
print(f"Доля попаданий в асимптотический доверительный интервал(10000): {coverage_probability2}")

x = np.linspace(0.5, 1.5, 1000)
pdf = expon.pdf(x, scale=1 / lam)
plt.plot(x, pdf, 'r-', lw=2, label='PDF of Exp(1)')

plt.xlabel('Значение')
plt.ylabel('Плотность вероятности')
plt.title('PDF экспоненциального распределения')
plt.legend(loc='best')
plt.grid()

plt.show()
