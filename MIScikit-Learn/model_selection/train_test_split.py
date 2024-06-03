import numpy as np

def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    """
    Split arrays or matrices into random train and test subsets.
    
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
        If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. 
        If train_size is also None, it will be set to 0.25.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. 
        If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.

    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as the class labels.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    
    arrays = [np.asarray(a) for a in arrays]
    n_samples = arrays[0].shape[0]
    
    if any(a.shape[0] != n_samples for a in arrays):
        raise ValueError("All input arrays must have the same number of samples")

    if test_size is None and train_size is None:
        test_size = 0.25
    if test_size is None:
        test_size = 1.0 - train_size
    if train_size is None:
        train_size = 1.0 - test_size

    if isinstance(test_size, float):
        test_size = int(test_size * n_samples)
    if isinstance(train_size, float):
        train_size = int(train_size * n_samples)

    if stratify is not None:
        if shuffle is False:
            raise ValueError("If stratify is not None, shuffle must be True")
        unique_classes, y_indices = np.unique(stratify, return_inverse=True)
        train_indices = []
        test_indices = []
        for class_index in range(len(unique_classes)):
            class_member_indices = np.where(y_indices == class_index)[0]
            if shuffle:
                np.random.shuffle(class_member_indices)
            n_class_train = int(train_size * len(class_member_indices) / n_samples)
            train_indices.extend(class_member_indices[:n_class_train])
            test_indices.extend(class_member_indices[n_class_train:])
    else:
        indices = np.arange(n_samples)
        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:train_size + test_size]

    return [a[train_indices] for a in arrays] + [a[test_indices] for a in arrays]
