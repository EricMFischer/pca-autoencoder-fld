import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from numpy.linalg import inv
from skimage.io import imread_collection
from project_1_pca import *


# -------------------------------- HELPER FUNCTIONS --------------------------------------
def get_landmarks(lm_file):
    mat_files = glob.glob(lm_file)
    return [sio.loadmat(f)['lms'] for f in mat_files]

def save_projections(train, lm_train, lm_male, lm_female, male_v_ch, female_v_ch):
    save_geo_projections(lm_train, lm_male)
    save_geo_projections(lm_train, lm_female)
    save_appear_projections(train, male_v_ch)
    save_appear_projections(train, female_v_ch)

def save_geo_projections(lm_train, gender_lms):
    lm_mean = calc_mean(lm_train)
    eigenwarpings = calc_eigenwarpings(lm_train, lm_mean, 50) # (50, 136, 1)
    reconstruct_landmarks(gender_lms, lm_mean, eigenwarpings[:10])

def save_appear_projections(imgs_train, gender_imgs):
    mean = calc_mean(imgs_train)
    eigenfaces = calc_eigenfaces(imgs_train, mean, 50) # (50, 16384, 1)
    reconstruct(gender_imgs, mean, eigenfaces)

def save_appear_projections_after_alignment(imgs_train, gender_imgs, lm_train):
    lm_mean = calc_mean(lm_train)
    mean = calc_mean(imgs_train)
    imgs_aligned = warp_imgs_to_mean(imgs_train, lm_train, lm_mean)
    eigenfaces = calc_eigenfaces(imgs_aligned, mean, 50) # (50, 16384, 1)
    reconstruct(gender_imgs, mean, eigenfaces)

def get_red_imgs(female=False, feature=None):
    appear_f = 'appear_proj_female.pkl' if female else 'appear_proj_male.pkl'
    geo_f = 'geo_proj_female.pkl' if female else 'geo_proj_male.pkl'
    if feature is None:
        appear_proj, geo_proj = np.array(get_data(appear_f)), np.array(get_data(geo_f))
        return np.concatenate((appear_proj, geo_proj), axis=1)
    elif feature is 'geo':
        return np.array(get_data(geo_f))
    elif feature is 'appear':
        appear_f = 'appear_proj_female_aligned.pkl' if female else 'appear_proj_male_aligned.pkl'
        return np.array(get_data(appear_f))

def get_red_test_imgs(female=False, feature=None):
    appear_f = 'appear_proj_female_test.pkl' if female else 'appear_proj_male_test.pkl'
    geo_f = 'geo_proj_female_test.pkl' if female else 'geo_proj_male_test.pkl'
    if feature is None:
        appear_proj, geo_proj = np.array(get_data(appear_f)), np.array(get_data(geo_f))
        return np.concatenate((appear_proj, geo_proj), axis=1)
    elif feature is 'geo':
        return np.array(get_data(geo_f))
    elif feature is 'appear':
        appear_f = 'appear_proj_female_test_aligned.pkl' if female else 'appear_proj_male_test_aligned.pkl'
        return np.array(get_data(appear_f))


# -------------------------------- FISHER FUNCTIONS --------------------------------------
def calc_scatter_matrix(red_imgs, mean_img, dim):
    summation = np.zeros((dim, dim))
    for img in red_imgs:
        diff = np.subtract(img, mean_img).reshape(-1, 1)
        summation += diff.dot(diff.reshape(1, -1))
    return summation

def calc_within_class_scatter_matrix(scatter_M_male, scatter_M_female):
    return scatter_M_male + scatter_M_female

def calc_fisher_face(within_class_scatter_M, mean_male, mean_female):
    return inv(within_class_scatter_M).dot((mean_male - mean_female).reshape(-1, 1))

def get_omega_transpose_x(fisher_face, red_test_imgs):
    return np.array([np.asscalar(fisher_face.T.dot(img)) for img in red_test_imgs])

def get_threshold(fisher_face, mean_male, mean_female):
    return np.asscalar(fisher_face.T.dot((mean_male + mean_female) * .5))

def disp_scatter_plot(fisher_face, red_test_imgs_male, red_test_imgs_female):
    male_vals = get_omega_transpose_x(fisher_face, red_test_imgs_male) # (82,)
    female_vals = get_omega_transpose_x(fisher_face, red_test_imgs_female) # (118,)

    threshold = -0.001
    male_errors = len(np.where(male_vals + threshold <= 0)[0])
    female_errors = len(np.where(female_vals + threshold >= 0)[0])
    error = (male_errors + female_errors) / (len(male_vals) + len(female_vals))

    plt.plot(male_vals, 'o', color='blue', label='Males')
    plt.plot(female_vals, 'o', color='red', label='Females')
    plt.legend()
    plt.show()

def disp_2d_scatter_plot(fisher_face_geo, fisher_face_appear, red_test_imgs_male_geo, red_test_imgs_male_appear, red_test_imgs_female_geo, red_test_imgs_female_appear):
    male_vals_geo = get_omega_transpose_x(fisher_face_geo, red_test_imgs_male_geo) # (82,)
    male_vals_appear = get_omega_transpose_x(fisher_face_appear, red_test_imgs_male_appear) # (82,)
    female_vals_geo = get_omega_transpose_x(fisher_face_geo, red_test_imgs_female_geo) # (118,)
    female_vals_appear = get_omega_transpose_x(fisher_face_appear, red_test_imgs_female_appear) # (118,)

    plt.plot(male_vals_geo, male_vals_appear, 'o', color='blue', label='Males')
    plt.plot(female_vals_geo, female_vals_appear, 'o', color='red', label='Females')
    plt.legend()
    plt.show()

def run_fisher():
    # ----------------------- Fisher: Question 1 ------------------------
    imgs_male = list(imread_collection('./male_images/*.jpg')) # (412, 128, 128, 3)
    imgs_female = list(imread_collection('./female_images/*.jpg')) # (588, 128, 128, 3)
    train_male, test_male = imgs_male[:330], imgs_male[330:]
    train_female, test_female = imgs_female[:470], imgs_female[470:]

    train_male_v_ch, train_female_v_ch = rgb2hsv_ch(train_male, 2), rgb2hsv_ch(train_female, 2)
    test_male_v_ch, test_female_v_ch = rgb2hsv_ch(test_male, 2), rgb2hsv_ch(test_female, 2)
    train, test = train_male_v_ch + train_female_v_ch, test_male_v_ch + test_female_v_ch


    lm_male = get_landmarks('./male_landmarks/*.mat')
    lm_female = get_landmarks('./female_landmarks/*.mat')
    lm_train_male, lm_test_male = lm_male[:330], lm_male[330:]
    lm_train_female, lm_test_female = lm_female[:470], lm_female[470:]
    lm_train, lm_test = lm_train_male + lm_train_female, lm_test_male + lm_test_female
    '''
    1) Find the Fisher face that distinguishes male from female using the training sets,
    and test it on the 200 test faces and report the error rate. This Fisher face
    mixes geometry and appearance differences between male and female.
    '''
    '''
    # save_projections(train, lm_train, lm_train_male, lm_train_female, train_male_v_ch, train_female_v_ch)
    red_imgs_male = get_red_imgs() # (330, 60)
    red_imgs_female = get_red_imgs(True) # (470, 60)
    mean_male = calc_mean(red_imgs_male) # (60,)
    mean_female = calc_mean(red_imgs_female) # (60,)

    scatter_M_male = calc_scatter_matrix(red_imgs_male, mean_male, 60) # (60, 60)
    scatter_M_female = calc_scatter_matrix(red_imgs_female, mean_female, 60)
    within_class_scatter_M = calc_within_class_scatter_matrix(scatter_M_male, scatter_M_female)
    fisher_face = calc_fisher_face(within_class_scatter_M, mean_male, mean_female) # (60, 1)

    # save_projections(train, lm_train, lm_test_male, lm_test_female, test_male_v_ch, test_female_v_ch)
    red_test_imgs_male = get_red_test_imgs() # (82, 60)
    red_test_imgs_female = get_red_test_imgs(True) # (118, 60)
    # disp_scatter_plot(fisher_face, red_test_imgs_male, red_test_imgs_female)
    '''

    # 2) Compute the Fisher face for the key point (geometric shape) and Fisher face for the
    # appearance (after aligning them to the mean position) respectively. Project all the
    # faces to the 2-D feature space learned by the fisher-faces, and visualize how separable
    # these points are.
    # (The within-class scatter matrix is very high-dimensional, you may compute the Fisher
    # faces over the reduced dimensions in steps 2 and 3 in section 2.1: i.e. each face is now
    # reduced to 10 dimensional geometric vector + 50 dimensional appearance vector. After the
    # Fisher Linear Discriminant analysis, we represent each face by as few as 2 dimensions for
    # discriminative purpose and yet, it can tell apart male from female!)

    # save_appear_projections_after_alignment(train, test_female_v_ch, lm_train)
    red_imgs_male_geo = get_red_imgs(False, 'geo') # (330, 10)
    red_imgs_male_appear = get_red_imgs(False, 'appear') # (330, 50)
    red_imgs_female_geo = get_red_imgs(True, 'geo') # (470, 10)
    red_imgs_female_appear = get_red_imgs(True, 'appear') # (470, 50)
    mean_male_geo = calc_mean(red_imgs_male_geo) # (10,)
    mean_male_appear = calc_mean(red_imgs_male_appear) # (50,)
    mean_female_geo = calc_mean(red_imgs_female_geo) # (10,)
    mean_female_appear = calc_mean(red_imgs_female_appear) # (50,)

    scatter_M_male_geo = calc_scatter_matrix(red_imgs_male_geo, mean_male_geo, 10)
    scatter_M_male_appear = calc_scatter_matrix(red_imgs_male_appear, mean_male_appear, 50)
    scatter_M_female_geo = calc_scatter_matrix(red_imgs_female_geo, mean_female_geo, 10)
    scatter_M_female_appear = calc_scatter_matrix(red_imgs_female_appear, mean_female_appear, 50)
    within_class_scatter_M_geo = calc_within_class_scatter_matrix(scatter_M_male_geo, scatter_M_female_geo)
    within_class_scatter_M_appear = calc_within_class_scatter_matrix(scatter_M_male_appear, scatter_M_female_appear)
    fisher_face_geo = calc_fisher_face(within_class_scatter_M_geo, mean_male_geo, mean_female_geo)
    fisher_face_appear = calc_fisher_face(within_class_scatter_M_appear, mean_male_appear, mean_female_appear)

    red_test_imgs_male_geo = get_red_test_imgs(False, 'geo') # (82, 10)
    red_test_imgs_male_appear = get_red_test_imgs(False, 'appear') # (82, 50)
    red_test_imgs_female_geo = get_red_test_imgs(True, 'geo') # (118, 10)
    red_test_imgs_female_appear = get_red_test_imgs(True, 'appear') # (118, 50)
    disp_2d_scatter_plot(fisher_face_geo, fisher_face_appear, red_test_imgs_male_geo, red_test_imgs_male_appear, red_test_imgs_female_geo, red_test_imgs_female_appear)


def main():
    run_fisher()

if __name__ == "__main__":
    main()
