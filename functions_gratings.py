import numpy as np
from scipy.linalg import norm
from functions_misc import normalize_matrix


def project_angles(angle_vector):
    """Project the angles in the vector into exponentials for orientation and direction"""
    exp_dsi = np.array([np.exp(1j * np.deg2rad(angle)) for angle in angle_vector])
    exp_osi = np.array([np.exp(2j * np.deg2rad(angle)) for angle in angle_vector])
    return {'exp_dsi': exp_dsi, 'exp_osi': exp_osi}


def calculate_dsi_osi(roi_data, angle_exponentials):
    """Calculate the DSI and OSI of the ROI based on the input angles"""

    # get the tuning curve
    tuning_curve = calculate_tuning_curve(normalize_matrix(roi_data))

    # calculate the indexes
    dsi = norm(np.sum(angle_exponentials['exp_dsi'] * np.abs(tuning_curve)) / np.sum(np.abs(tuning_curve)))
    osi = norm(np.sum(angle_exponentials['exp_osi'] * np.abs(tuning_curve)) / np.sum(np.abs(tuning_curve)))
    return dsi, osi, tuning_curve


def calculate_tuning_curve(roi_matrix):
    """Use SVD to extract the angular tuning curve of the target ROI"""
    _, _, v = np.linalg.svd(roi_matrix)
    return v[:, 0]
