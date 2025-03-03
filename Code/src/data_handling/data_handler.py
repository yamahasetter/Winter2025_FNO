def normalize(arr):
    """
        make arr 0-mean variance-1
    """
    return (arr-arr.mean())/arr.std()


def load_training_data(data_dir, trainN, lead, selection=None, exclude=None):
    """
        arguments:
            `trainN`, `lead`, `data_dir`

        go to `data_dir` and choose a random simulation.
        choose `trainN` random indices.
        starting at the random indices, select the next `lead`
        elements of the simulation and save to input, then
        save the `lead`+1 element to label.

        use selection to choose a specific data run

        use exclude to exclude a specific run from the output
    """
    # choose a simulation at random
    if selection==None:
        if exclude==None:
            simulation = data_dir + str(np.random.randint(0,199)) + '_filtered.npy'
            simulation = np.load(simulation)
        else:
            simulation = data_dir + str(np.random.choice([i for i in range(0,199) if i not in exclude])) + '_filtered.npy'
            simulation = np.load(simulation)
    else:
        simulation = data_dir + str(selection) + '_filtered.npy'
        simulation = np.load(simulation)

    # for input and label data
    input_ = np.ndarray((trainN, lead, 32,32))
    label = np.ndarray((trainN, lead, 32,32))
    sim_length = simulation.shape[0]

    for idx in range(trainN):
        # choose a random position between 0 and len(simulation-lead-1) in simulation array
        start = np.random.randint(0, sim_length-lead-1)
        input_[idx] = simulation[start:start+lead]
        # label[idx, 0] = simulation[start+lead+1]
        label[idx] = simulation[start+1:start+lead+1]

    # normalize
    input_, label = normalize(input_), normalize(label)

    # torch tensors
    return torch.from_numpy(input_).float(), torch.from_numpy(label).float()


def load_this_run(data_dir, lead, sim_name):
    """
        go to data_dir, choose sim_name and gather the data
        there, return first lead elements.
    """
    simulation = np.load(data_dir + sim_name)

    return simulation[:lead]