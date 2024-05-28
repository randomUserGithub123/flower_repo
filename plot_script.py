import pickle
import matplotlib.pyplot as plt

import os
from itertools import zip_longest

from dataset import prepare_dataset
from server import get_evaluate_fn

def count_files_in_directory(directory):
    total_files = 0
    for root, dirs, files in os.walk(directory):
        total_files += len(files)
    return total_files

def plot():
    _, _, testloader = prepare_dataset(10, 64)
    color_map = {
        0: "yellow",
        1: "blue",
    }
    directories = [
        "fedasync_noniid_homogeneous",
        "fedasync_iid_homogeneous",
        "fedasync_noniid_heterogeneous",
        "fedasync_iid_heterogeneous"
    ]
    data_list = list()
    for dir in directories:
        ages_list = list()
        accuracies_list = list()
        num_files = count_files_in_directory('saved_logs/' + dir)
        for i in range(1, num_files + 1):
            print("Opening file " + dir + str(i))
            with open('./saved_logs/' + dir + '/run_' + str(i) + '.bin', 'rb') as file:
                counter = 1
                ages = []
                accuracies = []
                while True:
                    try:
                        data = pickle.load(file)
                        loss, metrics = get_evaluate_fn(10, testloader)(0, data[1], {})
                        ages.append(counter)
                        accuracies.append(metrics['accuracy'])
                        counter += 1
                    except EOFError:
                        break
                    except pickle.UnpicklingError as e:
                        ages.append(counter)
                        accuracies.append(accuracies[len(accuracies) - 1])
                        break
                ages_list.append(ages)
                accuracies_list.append(accuracies)

        averaged_accuracies = []
        for acc_group in zip_longest(*accuracies_list, fillvalue=None):
            filtered_values = [x for x in acc_group if x is not None]
            if filtered_values:
                averaged_accuracies.append(sum(filtered_values) / len(filtered_values))
        
        data_list.append({'server_ages': ages_list[0], 'accuracies': averaged_accuracies})

    plt.figure(figsize=(10, 6))

    for i, dataset in enumerate(data_list):
        ages = dataset['server_ages']
        accuracies = dataset['accuracies']
        while len(ages) < len(accuracies):
            ages.append(None)
        plt.plot(ages, accuracies, marker='o', linestyle='-', label=f'Line {i+1}', color=color_map[i])

    plt.xlabel('Server Age')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Time')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    plot()