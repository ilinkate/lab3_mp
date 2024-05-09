import random

import numpy
import numpy as np
import time
import math
from scipy.stats import chi2
import matplotlib.pyplot as plt
from nistrng import *


# @brief LCG
def lcg(seed, n):
    k = 1103515
    b = 12345
    M = 2**31-1
    data_lcg = []

    for _ in range(n):
        seed = (k * seed + b) % M
        data_lcg.append(seed % M)
    return data_lcg

# @brief XORShift
def xorshift(seed,n):
    data_xor = []
    for _ in range(n):
        seed ^= (seed << 13) % (2**32)
        seed ^= (seed >> 17) % (2**32)
        seed ^= (seed << 5) % (2**32)
        data_xor.append(seed)
    return data_xor


# @brief Генерация выборок
num_samples = 20
min_range = 5000
min_elements = 100

samples_lcg = []
samples_xor = []

for _ in range(num_samples):
    sample_size = np.random.randint(min_elements, min_elements + 200)
    sample_range = np.random.randint(min_range, min_range + 1000)
    lcg_samples = lcg(sample_range, sample_size)
    samples_lcg.append(lcg_samples)

for _ in range(num_samples):
    sample_size = np.random.randint(min_elements, min_elements + 200)
    sample_range = np.random.randint(min_range, min_range + 1000)
    xor_samples = xorshift(sample_range, sample_size)
    samples_xor.append(xor_samples)

print("Выборки LCG:",lcg_samples)
print("Выборки XOR:",xor_samples)
print()


# @brief Расчет статистических показателей
def calculate_statistics(samples):
    means = []
    deviations = []
    variations = []

    for sample in samples:
        means.append(int(np.mean(sample)))
        deviations.append(int(np.std(sample)))
        variations.append(int((np.std(sample) / np.mean(sample))*100))

    return means, deviations, variations

mean_lcg, dev_lcg, var_lcg = calculate_statistics(samples_lcg)
mean_xor, dev_xor, var_xor = calculate_statistics(samples_xor)

print("Статистика LCG:")
print("Среднее значение:", mean_lcg)
print("Стандартное отклонение:", dev_lcg)
print("Коэффициент вариации (%):", var_lcg)
print()
print("Статистика XOR:")
print("Среднее значение:", mean_xor)
print("Стандартное отклонение:", dev_xor)
print("Коэффициент вариации (%):", var_xor)
print()


# @brief Проверка равномерности и случайности распределения
def chi_squared_test(samples):
    results = []

    for sample in samples:
        num_intervals = 1 + int(np.log2(len(sample)))

        observed = np.histogram(sample, bins=num_intervals)[0]
        expected = len(sample) / num_intervals

        chi_squared = sum((observed - expected)**2 / expected)
        critical_value = chi2.ppf(0.95, int( np.log2(len(sample)))-1)  # Уровень значимости 0.05

    # Сравниваем статистику с критическим значением
        if chi_squared <= critical_value:
          results.append(True)
        else:
          results.append(False)

    return results

chi_lcg = chi_squared_test(samples_lcg)
chi_xor = chi_squared_test(samples_xor)

print("Проверка равномерности и случайности LCG:", chi_lcg)
print("Проверка равномерности и случайности XOR:", chi_xor)
print()

# @brief Применение тестов NIST

print("Применение тестов NIST для ГПСЧ LCG:")
sequence = np.array(lcg_samples)
binary_sequence: numpy.ndarray = pack_sequence(sequence)

eligible_battery: dict = check_eligibility_all_battery(binary_sequence, SP800_22R1A_BATTERY)
    # Print the eligible tests
for name in eligible_battery.keys():
    results = run_all_battery(binary_sequence, eligible_battery, False)[:5]
    # Print results one by one
print("Test results:")
for result, elapsed_time in results:
    if result.passed:
        print("- PASSED - score: " + str(numpy.round(result.score, 3)) + " - " + result.name )
    else:
        print("- FAILED - score: " + str(numpy.round(result.score, 3)) + " - " + result.name )
print()

print("Применение тестов NIST для ГПСЧ XOR:")
sequence = np.array(xor_samples)
binary_sequence: numpy.ndarray = pack_sequence(sequence)

eligible_battery: dict = check_eligibility_all_battery(binary_sequence, SP800_22R1A_BATTERY)
    # Print the eligible tests
for name in eligible_battery.keys():
    results = run_all_battery(binary_sequence, eligible_battery, False)[:5]
    # Print results one by one
print("Test results:")
for result, elapsed_time in results:
    if result.passed:
        print("- PASSED - score: " + str(numpy.round(result.score, 3)) + " - " + result.name )
    else:
        print("- FAILED - score: " + str(numpy.round(result.score, 3)) + " - " + result.name )
print()



# Объемы выборок
sizes = [1000, 10000,100000,1000000]

# @brief Замер времени для разных размеров выборки
time_lcg = []
time_xorshift = []
time_random = []

for size in sizes:
    start = time.time()
    lcg(12345, size)
    end = time.time()
    time_lcg.append(end - start)

    start = time.time()
    xorshift(12345, size)
    end = time.time()
    time_xorshift.append(end - start)

    start = time.time()
    [random.random() for _ in range(size)]
    end = time.time()
    time_random.append(end - start)

# @brief Построение графиков
plt.plot(sizes, time_lcg, label='LCG')
plt.plot(sizes, time_xorshift, label='Xorshift')
plt.plot(sizes, time_random, label='Python Random')
plt.xlabel('Объем выборки')
plt.ylabel('Время, сек')
plt.title('Зависимость объема выборки от времени')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()