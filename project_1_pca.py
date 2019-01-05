"""
This project compares 3 types of representations in the context of dimension reduction:
2 generative methods, PCA (linear) and Autoencoder (non-linear),
and 1 discriminative method, Fisher Linear Discriminants (FLD).

1000 images (800 training, 200 test)
128 x 128 pixels
68 landmarks (to be aligned geometrically to compare appearance meaningfully)
Training set: calculate eigenvalues and eigenvectors
Test set: calculate reconstruction errors using eigenvectors
"""
import numpy as np
import scipy.io as sio
import glob
import math
import cv2
import pickle
import matplotlib.pyplot as plt
from skimage import color
from skimage.io import imread_collection
from mywarper import warp
K = [1,5,10,15,20,25,30,35,40,45,50]


# --------------------------------- HELPER FUNCTIONS -----------------------------------------
def save_data(data, filename):
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

def get_data(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def get_landmarks():
    mat_files = glob.glob('./landmarks/*.mat')
    return [sio.loadmat(f)['lms'] for f in mat_files]

def rgb2hsv_ch(imgs, channel):
    return [color.rgb2hsv(img)[:, :, channel] for img in imgs]

def hsv2rgb(imgs):
    return [color.hsv2rgb(img) for img in imgs]

def merge_hsv_ch(orig_imgs, recon_imgs):
    hsv_imgs = []
    for i, img in enumerate(orig_imgs):
        h, s = rgb2hsv_ch([img], 0)[0], rgb2hsv_ch([img], 1)[0]
        v = recon_imgs[i].clip(min=0, max=1)
        hsv_imgs.append(cv2.merge([h, s, v]))
    return hsv_imgs

def calc_mean(imgs):
    imgs_sum = np.zeros(np.shape(imgs[0]))
    for x in imgs:
        imgs_sum += x
    return imgs_sum / len(imgs)

def normalize(imgs, mean):
    return [np.subtract(img, mean) for img in imgs]


# ------------------------------------- ERROR FUNCTIONS ------------------------------------------
def calc_recon_error(recon_imgs_by_k_efaces, X_test):
    recon_error_by_k = []
    for recon_imgs in recon_imgs_by_k_efaces:
        recon_error = 0
        for i, recon_img in enumerate(recon_imgs):
            original = rgb2hsv_ch([X_test[i]], 2)[0]
            recon_error += np.sum(np.square(recon_img - original))
        recon_error_by_k.append(recon_error / 128**2 / 200)
    return recon_error_by_k

def calc_recon_error_landmarks(recon_landmarks_by_k_ewarpings, landmark_test):
    recon_error_by_k = []
    for recon_landmarks in recon_landmarks_by_k_ewarpings:
        recon_error = 0
        for i, recon_landmark in enumerate(recon_landmarks): # 200 landmarks
            original = landmark_test[i]
            recon_error += math.sqrt(np.sum(np.square(np.subtract(original, recon_landmark))))
        recon_error_by_k.append(recon_error / 200)
    return recon_error_by_k


# --------------------------------- DISPLAY FUNCTIONS -----------------------------------------
def disp_recon_error(recon_errors, x_label, y_label):
    plt.plot(K, recon_errors)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def disp_eigenfaces(eigenfaces):
    f, axes = plt.subplots(2, 5)
    for i, eigenface in enumerate(eigenfaces):
        row = 0 if i < 5 else 1
        eigenface = eigenface.reshape(128, 128)
        axes[row, i % 5].imshow(eigenface, cmap='binary')
    plt.show()

def disp_eigenwarpings(eigenwarpings, mean, mean_img):
    for ewarping in eigenwarpings:
        ewarping = np.multiply(ewarping.reshape(68, 2), mean) + mean # add back mean for meaning
        x = [np.asscalar(item) for item in ewarping[:, 0]]
        y = [np.asscalar(item) for item in ewarping[:, 1]]
        plt.plot(x, y, 'o')
        plt.gca().invert_yaxis()
        plt.imshow(mean_img, cmap='binary')
        plt.show()

def disp_recon_images(orig_imgs, recon_imgs):
    recon_hsv_imgs = merge_hsv_ch(orig_imgs, recon_imgs)
    recon_rgb_imgs = hsv2rgb(recon_hsv_imgs)
    f, axes = plt.subplots(2, 5)
    for i, img in enumerate(recon_rgb_imgs):
        row = 0 if i < 5 else 1
        axes[row, i % 5].imshow(img)
    plt.show()
    f, axes = plt.subplots(2, 5)
    for i, img in enumerate(orig_imgs):
        row = 0 if i < 5 else 1
        axes[row, i % 5].imshow(img)
    plt.show()

def disp_warped_images(orig_imgs, recon_imgs):
    recon_hsv_imgs = merge_hsv_ch(orig_imgs, recon_imgs)
    recon_rgb_imgs = hsv2rgb(recon_hsv_imgs)
    f, axes = plt.subplots(4, 5)
    for i, img in enumerate(recon_rgb_imgs):
        row = math.floor(i / 5)
        axes[row, i % 5].imshow(img)
    plt.show()
    f, axes = plt.subplots(4, 5)
    for i, img in enumerate(orig_imgs):
        row = math.floor(i / 5)
        axes[row, i % 5].imshow(img)
    plt.show()

def disp_synthesized_images(orig_imgs, synthesized_imgs):
    f, axes = plt.subplots(5, 10)
    for i, img in enumerate(synthesized_imgs):
        row = math.floor(i / 10)
        # img = img.reshape(128, 128)
        axes[row, i % 10].imshow(img, cmap='binary')
    plt.show()

# Plot reconstruction error (squared intensity difference between reconstructed images
# and originals) per pixel (i.e. normalize error by pixel number, and average over the testing
# images) over number of eigenfaces K = 1,5,10,15,...,50.
def disp_appear_recon_error(X_test, X_test_v_ch, mean, eigenfaces):
    recon_imgs_by_k_efaces = reconstruct_by_k_eigenfaces(X_test_v_ch, mean, eigenfaces)
    recon_errors = calc_recon_error(recon_imgs_by_k_efaces, X_test)
    disp_recon_error(recon_errors, 'k eigenfaces', 'Reconstruction Error of Appearance per Pixel')

# Plot reconstruction error (in terms of distance) over number of
# eigenwarpings K = 1,5,10,15,...,50 (error averaged over test images)
def disp_geo_recon_error(landmark_test, landmark_mean, eigenwarpings):
    recon_lms_by_k_ewarpings = reconstruct_by_k_eigenwarpings(landmark_test, landmark_mean, eigenwarpings)
    recon_errors = calc_recon_error_landmarks(recon_lms_by_k_ewarpings, landmark_test)
    disp_recon_error(recon_errors, 'k eigenwarpings', 'Reconstruction Error of Geometry')

# Plot reconstruction error per pixel over eigenfaces K = 1,5,10,15,...,50.
def disp_appear_geo_recon_error(X_test, X_test_v_ch, mean, eigenfaces):
    warped_imgs_by_k_efaces = reconstruct_by_k_eigenfaces(X_test_v_ch, mean, eigenfaces)
    recon_errors = calc_recon_error(warped_imgs_by_k_efaces, X_test)
    disp_recon_error(recon_errors, 'k eigenfaces', 'Reconstruction Error of Geo., App. per Pixel')

def reconstruct_by_k_eigenfaces(X_test_v_ch, mean, eigenfaces):
    return [reconstruct(X_test_v_ch, mean, eigenfaces[:k]) for k in K]

def reconstruct_by_k_eigenwarpings(landmark_test, landmark_mean, eigenwarpings):
    return [reconstruct_landmarks(landmark_test, landmark_mean, eigenwarpings[:k]) for k in K]


# ------------------------------------- WARPING FUNCTIONS ----------------------------------------
# imgs:  (800, 128, 128), landmarks:  (800, 68, 2), landmark_mean:  (68, 2)
def warp_imgs_to_mean(imgs, landmarks, landmark_mean):
    warped_imgs = []
    for i, img in enumerate(imgs):
        img = np.expand_dims(img, axis=2)
        warped_imgs.append(warp(img, landmarks[i], landmark_mean))
    return np.squeeze(warped_imgs)

def warp_imgs_to_recon_pos(imgs, landmark_mean, recon_test_landmarks):
    warped_imgs = []
    for i, img in enumerate(imgs):
        img = np.expand_dims(img, axis=2)
        warped_imgs.append(warp(img, landmark_mean, recon_test_landmarks[i]))
    return np.squeeze(warped_imgs)


# --------------------------------- EIGENVECTOR FUNCTIONS ----------------------------------------
# X: (800, 16384) for training images, (1000, 136) for landmarks
def get_X(norm_imgs):
    return np.matrix([x.flatten() for x in norm_imgs])

def calc_eigenfaces(train, mean, num):
    X = get_X(normalize(train, mean)) # (800, 16384)
    evalues, evectors = np.linalg.eig(X.dot(X.T))
    evectors = evectors[:, np.argsort(evalues)[::-1]] # (800, 800), same as X.dot(X_tr)
    efaces = [] # (50, 16384, 1)
    for i in range(num):
        eface = X.T * evectors[:, i]
        efaces.append((eface / np.linalg.norm(eface)).real)
    # eigenvalues = (np.sort(np.linalg.eigvals(np.array(efaces).reshape(50,128,128)))[::-1]).real
    # save_data(eigenvalues, 'eigenvalues_appear_v2.pkl')
    return efaces

def calc_eigenwarpings(landmarks, mean, num):
    X = get_X(normalize(landmarks, mean))
    evalues, evectors = np.linalg.eig(X.T.dot(X))
    # save_data(np.sort(evalues)[::-1], 'eigenvalues_geo.pkl')
    evectors = evectors[:, np.argsort(evalues)[::-1]]
    return [evectors[:, i] for i in range(num)] # (50, 136, 1)

# -------------------------------- RECONSTRUCTION FUNCTIONS --------------------------------------
# X_test_v_ch: (200, 128, 128), mean: (128, 128), eigenfaces: (50, 16384, 1)
def reconstruct(X_test_v_ch, mean, eigenfaces, synthesize=False):
    recon_imgs = []
    # appear_proj = []
    # eigenvalues = get_data('eigenvalues_appear.pkl')
    for i, I in enumerate(X_test_v_ch):
        sum_over_efaces = np.zeros((128**2, 1))
        # appear_proj.append([])
        for j, eface in enumerate(eigenfaces):
            proj = np.asscalar((I - mean).flatten().dot(eface))
            # if synthesize:
            #     mu, s_d = 0, math.sqrt(eigenvalues[j]) # mean and standard deviation
            #     proj = np.random.normal(mu, s_d)
            #     print('appear proj: ', proj)
            sum_over_efaces += (proj * eface)
            # appear_proj[i].append(proj)
        recon_imgs.append(np.reshape(sum_over_efaces, (128, 128)) + mean)
    # print('appear_proj: ', np.shape(appear_proj))
    # save_data(appear_proj, 'appear_proj_female_test_aligned.pkl')
    return recon_imgs

# Sample coefficients b_i ~ gaussian(mean=0, variance_i=λ_i), and synthesize an image I by:
# I_appearance = mean_face + Σb_i * e_i, where b_i is coeff. sampled for each eigen-axis e_i
# I_landmark = mean_landmark + Σb_i * e_i
# Finally, reconstruct the image by warp(I_appearance, from: mean_landmark, to: I_landmark)
def reconstruct_landmarks(lm_test, mean, eigenwarpings, synthesize=False):
    recon_landmarks = []
    # geo_proj = []
    # eigenvalues = get_data('eigenvalues_geo.pkl')
    for i, lm in enumerate(lm_test):
        sum_over_ewarpings = np.zeros((136, 1))
        # geo_proj.append([])
        for j, ewarping in enumerate(eigenwarpings):
            proj = np.asscalar((lm - mean).flatten().dot(ewarping))
            # if synthesize:
            #     mu, s_d = 0, math.sqrt(eigenvalues[j]) # mean and standard deviation
            #     proj = np.random.normal(mu, s_d)
            #     print('geo proj: ', proj)
            sum_over_ewarpings += (proj * ewarping)
            # geo_proj[i].append(proj)
        recon_landmarks.append(np.reshape(sum_over_ewarpings, (68, 2)) + mean)
    # print('geo_proj: ', np.shape(geo_proj))
    # save_data(geo_proj, 'geo_proj_female_test.pkl')
    return recon_landmarks


def run_pca():
    # ------------------------- PCA: Reconstruct images by appearance --------------------------
    images = imread_collection('./images/*.jpg')
    X_test = images[800:]
    X_train, X_test_v_ch = rgb2hsv_ch(images[:800], 2), rgb2hsv_ch(X_test, 2)
    landmarks = get_landmarks()
    lm_train, lm_test = landmarks[:800], landmarks[800:]

    '''
    # Compute mean and first 50 eigenfaces for training images (with no landmark aligment)
    mean = calc_mean(X_train)
    eigenfaces = calc_eigenfaces(X_train, mean, 50) # (50, 16384, 1)
    disp_eigenfaces(eigenfaces[:10])

    # Use them to reconstruct the 200 test images
    recon_imgs = reconstruct(X_test_v_ch, mean, eigenfaces)
    disp_recon_images(X_test[:10], recon_imgs[:10])
    disp_appear_recon_error(X_test, X_test_v_ch, mean, eigenfaces)
    '''

    '''
    # ------------------------ PCA: Reconstruct landmarks by geometry --------------------------
    # Compute mean and first 50 eigenwarpings for training images
    lm_mean = calc_mean(lm_train)
    eigenwarpings = calc_eigenwarpings(lm_train, lm_mean, 50) # (50, 136, 1)
    # disp_eigenwarpings(eigenwarpings[:10], lm_mean, mean)
    # disp_geo_recon_error(lm_test, lm_mean, eigenwarpings)
    '''

    '''
    # --------------------- PCA: Reconstruct by appearance and geometry ------------------------
    # First align images by warping them into mean position, then compute the eigenfaces
    # from the aligned images (compared to unaligned in Part 1).
    X_train_aligned = warp_imgs_to_mean(X_train, lm_train, lm_mean)
    eigenfaces = calc_eigenfaces(X_train_aligned, mean, 50) # (50, 16384, 1)

    # 1) Recon landmarks: for each test img, project its landmarks onto top 10 eigenwarpings
    recon_landmarks = reconstruct_landmarks(lm_test, lm_mean, eigenwarpings[:10])

    # 2) Recon imgs: now warp test images to mean position and project onto top 50 eigenfaces
    X_test_aligned = warp_imgs_to_mean(X_test_v_ch, lm_test, lm_mean)
    recon_imgs = reconstruct(X_test_aligned, mean, eigenfaces)

    # 3) Warp reconstructed test images to positions of reconstructed test landmarks
    recon_imgs = warp_imgs_to_recon_pos(recon_imgs, lm_mean, recon_landmarks)
    disp_warped_images(X_test[:20], recon_imgs[:20])
    disp_appear_geo_recon_error(X_test, X_test_v_ch, mean, eigenfaces)
    '''

    '''
    # ---------------------------- PCA: Synthesize random faces --------------------------------
    # Sample coefficients b_i ~ gaussian(mean=0, variance_i=λ_i), and synthesize an image I by:
    # I_appearance = mean_face + Σb_i * e_i, where b_i is coeff. sampled for each eigen-axis e_i
    # I_landmark = mean_landmark + Σb_i * e_i
    # Finally, reconstruct the image by warp(I_appearance, from: mean_landmark, to: I_landmark)
    lm_mean = calc_mean(lm_train)
    X_train_aligned = warp_imgs_to_mean(X_train, lm_train, lm_mean)
    eigenfaces = calc_eigenfaces(X_train_aligned, mean, 50) # (50, 16384, 1)
    eigenwarpings = calc_eigenwarpings(lm_train, lm_mean, 50) # (50, 136, 1)

    recon_landmarks = reconstruct_landmarks(lm_train[50:], lm_mean, eigenwarpings[:10], True)

    aligned_imgs = warp_imgs_to_mean(X_train[50:], lm_train[50:], lm_mean)
    recon_imgs = reconstruct(aligned_imgs, mean, eigenfaces, True)

    recon_imgs = warp_imgs_to_recon_pos(recon_imgs, lm_mean, recon_landmarks)
    disp_synthesized_images(X_train[50:], recon_imgs[50:])
    '''

def main():
    run_pca()

if __name__ == "__main__":
    main()
