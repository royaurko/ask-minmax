from jenks import jenks
import numpy as np


def gvf(array, classes):
    """ Compute goodness of fit with Jenks natural breaks algorithm with num classes = classes
    :param array: The numpy array on which to apply JNB algorithm
    :param classes: The number of classes in JNB algorithm
    :return: The goodness of fit value
    """
    # Get the break points
    classes = jenks(array, classes)
    # Do the actual classification
    classified = np.array([classify(i, classes) for i in array])
    # Max value of zones
    maxz = max(classified)
    # Nested list of zone indices
    zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]
    # Sum of squared deviations from array mean
    sdam = np.sum((array - array.mean()) ** 2)
    # Sorted polygon stats
    array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]
    # Sum of squared deviations of class means
    sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in array_sort])
    # Goodness of variance fit
    gvf = (sdam - sdcm) / sdam
    return gvf


def classify(value, breaks):
    """ Helper function
    :param value: Classify the numbers to correct clusters according to Jenks clusters
    :param breaks: The number of classes
    :return:
    """
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks) - 1