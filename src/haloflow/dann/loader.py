import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_data(X_train, X_test):
    """This function scales the data using the StandardScaler.
    This is done for making sure that the data is scaled properly.

    Parameters
    ----------
    X_train : np.ndarray
        The training data.
    X_test : np.ndarray
        The test data.
    
    Returns
    -------
    X_train : np.ndarray
        The scaled training data.
    X_test : np.ndarray
        The scaled test data.
    """
    scaler = StandardScaler()
    X_combined = np.vstack([X_train, X_test])
    scaler.fit(X_combined)

    # scale the data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test
