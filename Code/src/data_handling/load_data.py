import numpy as np


def normalize(arr):
    """
        make arr 0-mean variance-1
    """
    return (arr-arr.mean())/arr.std()


def load_training_data(data_dir, trainN, lead):
    """
        arguments:
            `trainN`, `lead`, `data_dir`

        go to `data_dir` and choose a random simulation.
        choose `trainN` random indices.
        starting at the random indices, select the next `lead`
        elements of the simulation and save to input, then
        save the `lead`+1 element to label.
    """
    # choose a simulation at random
    simulation = data_dir + str(np.random.randint(0,199)) + '_filtered.npy'
    simulation = np.load(simulation)

    # for input and label data
    input_ = np.ndarray((trainN, lead, 32,32))
    label = np.ndarray((trainN, 32,32))
    sim_length = simulation.shape[0]

    for idx in range(trainN):
        # choose a random position between 0 and len(simulation-lead-1) in simulation array
        start = np.random.randint(0, sim_length-lead-1)
        input_[idx] = simulation[start:start+lead]
        label[idx] = simulation[start+lead+1]

    return normalize(input_), normalize(label)

