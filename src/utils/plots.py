from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl

def load_pickle_file(path):
    with open(path, 'rb') as f:
            data = pkl.load(f)
    return data


def min_regret_calc(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)

    max_length = max(len(data[key]['regret']) for key in data.keys())

    min_regret = np.inf * np.ones(max_length)

    for key in data.keys():
        regret = data[key]['regret']
        min_regret[:len(regret)] = np.minimum(min_regret[:len(regret)], regret)

    return min_regret

def linear_data_load(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return np.array([[key, data[key]['regret'][-1]] for key in data.keys()])


# Set up the tab10 colormap
tab10 = plt.cm.tab10
cmap = tab10

line_width = 1

def pickle_plot_data(axs, data, label, offset,color,keys=None):    
    
    median = np.percentile(data, 50,axis=0) 
    upper_quantile = np.percentile(data, 85,axis=0)
    
    lower_quantile = np.percentile(data,0,axis=0)

    if keys is None:
        x = np.arange(offset,data.shape[1]+offset)
    else:
        x = [key + offset for key in keys]

    # if offset >0:
    #     # extend the list at the starting with the first value of regret 
    #     median = list(median[0]*np.ones(offset)) + list(median)
    #     upper_quantile = list(upper_quantile[0]*np.ones(offset)) + list(upper_quantile)
    #     lower_quantile = list(lower_quantile[0]*np.ones(offset)) + list(lower_quantile)
    #axs.plot(x, median, label = label)
    #axs.fill_between(x, lower_quantile, upper_quantile, alpha=alpha)   
    
    axs.plot(x, median, label=label, linestyle='-', linewidth=line_width, color=color, zorder=2)
    axs.fill_between(x, lower_quantile, upper_quantile, color=color, alpha=0.1, zorder=1)
   
#set3_5 = colormaps["Set1"].resampled(10)
# colors = [
#     '#e6194B',  # Red
#     '#3cb44b',  # Green
#     '#4363d8',  # Blue
#     '#f58231',  # Orange
#     '#911eb4',  # Purple
#     '#42d4f4',  # Cyan
#     '#f032e6',  # Magenta
#     '#bfef45',  # Lime
#     '#fabed4',  # Pink
#     '#469990'   # Teal
# ]
# cmap = ListedColormap(colors)




def plot_optimization_results(algorithm_files,offset=650):
    fig, ax = plt.subplots(1, 1, dpi=300)
    cmap = plt.get_cmap('tab10')

    for count, (algorithm, file_path) in enumerate(algorithm_files.items()):
            
        if algorithm in ['SequOOL', 'SOO']:
            # there is no randomness in the algorithm
            data = load_pickle_file(file_path)[0]
            array = np.array([data[key]['regret'][-1] for key in data.keys()])
            array = array.reshape(1, -1)
            keys_list = list(data.keys())

            # if you do not run linear steps
            if len(keys_list) == 1:
                pickle_plot_data(ax,np.array(list(data.values())[0]['regret']).reshape(1,-1),algorithm, 0, cmap(count))
            else:
                pickle_plot_data(ax, array, algorithm, 0, cmap(count), keys=keys_list)

        elif algorithm == 'RESOO':
            data = load_pickle_file(file_path)
            two_d_array =([[trial_data[key]['regret'][-1] for key in trial_data.keys() if trial_data[key] is not None] for trial_data in data])
            two_d_array = np.array([data for data in two_d_array if len(data) >0])
            keys_list = list(data[0].keys())
            offset = offset if algorithm == 'SequOOL on $\hat{\mathcal{A}}$' else 0
            if len(keys_list) == 1:
                pickle_plot_data(ax, two_d_array, algorithm, offset, cmap(count))
            pickle_plot_data(ax, two_d_array, algorithm, offset, cmap(count), keys=keys_list)


        elif algorithm =='SequOOL on $\hat{\mathcal{A}}$':
            data = load_pickle_file(file_path)
            two_d_array =([[trial_data[key]['regret'][-1] for key in trial_data.keys() if trial_data[key] is not None] for trial_data in data])
            two_d_array = np.array([data for data in two_d_array if len(data) >0])
            keys_list = list(data[0].keys())
            if len(keys_list) == 1:
                pickle_plot_data(ax, two_d_array, algorithm, 650, cmap(count))
            pickle_plot_data(ax, two_d_array, algorithm, 650, cmap(count), keys=keys_list)
        elif algorithm == 'Direct':
            data = load_pickle_file(file_path)[0]
            print(data['best_values'] - data['fmin'])
            ax.plot((data['best_values'] - data['fmin']), label=algorithm, color=cmap(count), linewidth=1)
        elif algorithm in ['Random Search', 'CMA-ES']:
            data = load_pickle_file(file_path)
            two_d_array = [(data[i]['best_values']-data[i]['fmin'])for i in range(len(data))]
            two_d_array = np.array(two_d_array)
            pickle_plot_data(ax, two_d_array, algorithm, 0, cmap(count))

        elif algorithm == 'Dual Annealing':
            data = load_pickle_file(file_path)
            two_d_array =[data[i]['best_values'] -data[i]['fmin'].item() for i in range(len(data))]
            two_d_array = (np.array(two_d_array))
            pickle_plot_data(ax, two_d_array, algorithm, 0, cmap(count))
        elif algorithm == "all_dimensions_look_ahead":
            data = load_pickle_file(file_path)
            two_d_array = np.array([data[i]['regret'] for i in range(len(data))])
            pickle_plot_data(ax,two_d_array,algorithm,0,cmap(count))
        else:
            data = load_pickle_file(file_path)
            ax.plot(data['regret'], label=algorithm, color=cmap(count), linewidth=1)

    plt.yscale('log')
    plt.legend(fontsize=10, loc=loc)
    plt.title(function_name, fontsize=16)
    ax.set_xlabel('Number of evaluations', fontsize=14)
    ax.set_ylabel('Regret', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{function_name}.png')
    plt.show()

    return fig, ax