from collections import Counter
from math import sqrt, log10
from pathlib import Path
import copy

import matplotlib.pyplot as plt

scratch_root_path = Path('')

nb_exp = 10

def arome():
    counter = Counter()
    counter_squared = Counter()
    for exp in range(nb_exp):
        folder = scratch_root_path / f'samples_AROME_for_AE_{exp + 1}'
        with open(folder / 'log.txt', 'r') as logfile:
            file_content = logfile.read()
        lines = file_content.split('\n')
        counter_temp = Counter()
        counter_temp_squared = Counter()
        for line in lines:
            parts = line.split(': ')
            if len(parts) == 2:
                key, value = parts
                counter_temp[key] = int(value)
                counter_temp_squared[key] = int(value) ** 2
        counter += counter_temp
        counter_squared += counter_temp_squared

    with open(scratch_root_path / 'log_AROME.txt', 'w') as logfile:
        for key in counter.keys():
            counter[key] /= nb_exp
            counter_squared[key] = round(sqrt((counter_squared[key] / nb_exp) - counter[key] ** 2), 2)
            logfile.write(f'{key}:\n')
            logfile.write(f'\tmean: {counter[key]}\tstd: {counter_squared[key]}\n')

    original_counter = copy.deepcopy(counter)
    original_counter_squared = copy.deepcopy(counter_squared)

    max_freq_feature = max(counter, key=counter.get)
    del counter[max_freq_feature]
    del counter_squared[max_freq_feature]

    keys = list(counter.keys())
    keys.remove(max(keys))
    means = [counter[key] for key in keys]
    stds = [counter_squared[key] for key in keys]

    # Plotting the histogram
    fig, ax = plt.subplots()
    bars = ax.bar(keys, means, yerr=stds, capsize=5)

    # Adding labels and title
    ax.set_ylabel('Mean Values')
    ax.set_title('Histogram with Standard Deviation')

    # Adding std annotations above each bar
    for bar, std in zip(bars, stds):
        height = bar.get_height()
        ax.annotate(f'{std:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom')
    plt.xticks(rotation='vertical')
    plt.savefig(scratch_root_path / 'AROME.png', bbox_inches='tight')
    plt.close()
    return original_counter, original_counter_squared

def gan():
    counter = Counter()
    counter_squared = Counter()
    for exp in range(nb_exp):
        folder = scratch_root_path / f'samples_detransformed_for_AE_{exp + 1}'
        with open(folder / 'log.txt', 'r') as logfile:
            file_content = logfile.read()
        lines = file_content.split('\n')
        counter_temp = Counter()
        counter_temp_squared = Counter()
        for line in lines:
            parts = line.split(': ')
            if len(parts) == 2:
                key, value = parts
                counter_temp[key] = int(value)
                counter_temp_squared[key] = int(value) ** 2
        counter += counter_temp
        counter_squared += counter_temp_squared

    with open(scratch_root_path / 'log_GAN.txt', 'w') as logfile:
        for key in counter.keys():
            counter[key] /= nb_exp
            counter_squared[key] = round(sqrt((counter_squared[key] / nb_exp) - counter[key] ** 2), 2)
            logfile.write(f'{key}:\n')
            logfile.write(f'\tmean: {counter[key]}\tstd: {counter_squared[key]}\n')

    original_counter = copy.deepcopy(counter)
    original_counter_squared = copy.deepcopy(counter_squared)

    max_freq_feature = max(counter, key=counter.get)
    del counter[max_freq_feature]
    del counter_squared[max_freq_feature]

    keys = list(counter.keys())
    keys.remove(max(keys))
    means = [counter[key] for key in keys]
    stds = [counter_squared[key] for key in keys]

    # Plotting the histogram
    fig, ax = plt.subplots()
    bars = ax.bar(keys, means, yerr=stds, capsize=5)

    # Adding labels and title
    ax.set_ylabel('Mean Values')
    ax.set_title('Histogram with Standard Deviation')

    # Adding std annotations above each bar
    for bar, std in zip(bars, stds):
        height = bar.get_height()
        ax.annotate(f'{std:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom')
    plt.xticks(rotation='vertical')
    plt.savefig(scratch_root_path / 'GAN.png', bbox_inches='tight')
    plt.close()
    return original_counter, original_counter_squared

def plot_histogram(keys, means, stds, label, ax):
    bars = ax.bar(keys, means, yerr=stds, capsize=5, label=label, alpha=0.5)

    # # Adding std annotations above each bar
    # for bar, std in zip(bars, stds):
    #     height = bar.get_height()
    #     ax.annotate(f'{std:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
    #                 xytext=(0, 3), textcoords='offset points',
    #                 ha='center', va='bottom')

if __name__ == '__main__':
    counter_arome, counter_squared_arome = arome()
    counter_gan, counter_squared_gan = gan()

    with open(scratch_root_path / 'difference.txt', 'w') as logfile:
        for key in counter_arome.keys():
            logfile.write(f'{key}:\t{round(counter_gan[key] - counter_arome[key], 2)}\n')
    
    fig, ax = plt.subplots()

    max_freq_feature = max(counter_arome, key=counter_arome.get)
    del counter_arome[max_freq_feature]
    del counter_squared_arome[max_freq_feature]

    keys_arome = list(counter_arome.keys())
    keys_arome.remove(max(keys_arome))
    means_arome = [counter_arome[key] for key in keys_arome]
    stds_arome = [counter_squared_arome[key] for key in keys_arome]

    max_freq_feature = max(counter_gan, key=counter_gan.get)
    del counter_gan[max_freq_feature]
    del counter_squared_gan[max_freq_feature]

    keys_gan = list(counter_gan.keys())
    keys_gan.remove(max(keys_gan))
    means_gan = [counter_gan[key] for key in keys_gan]
    stds_gan = [counter_squared_gan[key] for key in keys_gan]


    plot_histogram(keys_arome, means_arome, stds_arome, 'AROME', ax)
    plot_histogram(keys_gan, means_gan, stds_gan, 'GAN', ax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.xticks(rotation='vertical')
    plt.savefig(scratch_root_path / 'GANvsAROME.png', bbox_inches='tight')