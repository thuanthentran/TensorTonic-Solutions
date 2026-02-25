import numpy as np

def rmsprop_step(w, g, s, lr, beta, eps):
    # Ép sang numpy array nếu người dùng truyền list
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)

    # Step 1
    new_s = beta * s + (1 - beta) * (g ** 2)

    # Step 2
    new_w = w - (lr / (np.sqrt(new_s) + eps)) * g

    return new_w, new_s