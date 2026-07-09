"""RELION's exact Euler conversions (ported from RELION 4.0 src/euler.cpp).
ZYZ intrinsic; alpha=rot, beta=tilt, gamma=psi (degrees)."""
import numpy as np

def angles2matrix(rot, tilt, psi):
    a, b, g = np.deg2rad([rot, tilt, psi])
    ca, sa, cb, sb, cg, sg = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(g), np.sin(g)
    cc, cs, sc, ss = cb*ca, cb*sa, sb*ca, sb*sa
    return np.array([
        [ cg*cc - sg*sa,  cg*cs + sg*ca, -cg*sb],
        [-sg*cc - cg*sa, -sg*cs + cg*ca,  sg*sb],
        [ sc,             ss,             cb   ],
    ])

def matrix2angles(A):
    eps = 16*np.finfo(np.float32).eps
    abs_sb = np.hypot(A[0,2], A[1,2])
    if abs_sb > eps:
        gamma = np.arctan2(A[1,2], -A[0,2])
        alpha = np.arctan2(A[2,1], A[2,0])
        if abs(np.sin(gamma)) < np.finfo(np.float32).eps:
            sign_sb = np.sign(-A[0,2]/np.cos(gamma))
        else:
            sign_sb = np.sign(A[1,2]) if np.sin(gamma) > 0 else -np.sign(A[1,2])
        beta = np.arctan2(sign_sb*abs_sb, A[2,2])
    else:
        if np.sign(A[2,2]) > 0:
            alpha, beta, gamma = 0.0, 0.0, np.arctan2(-A[1,0], A[0,0])
        else:
            alpha, beta, gamma = 0.0, np.pi, np.arctan2(A[1,0], -A[0,0])
    return np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma)   # rot, tilt, psi

if __name__ == "__main__":
    from scipy.spatial.transform import Rotation
    err = 0.0
    for k in range(200):
        R = Rotation.random(random_state=k).as_matrix()
        err = max(err, np.abs(angles2matrix(*matrix2angles(R)) - R).max())
    print("max roundtrip error over 200 random rotations:", err)
