lm_mean = calc_mean(lm_train)
eigenwarpings = calc_eigenwarpings(lm_train, lm_mean, 50) # (50, 136, 1)
recon_landmarks = reconstruct_landmarks(lm_train, lm_mean, eigenwarpings[:10])
