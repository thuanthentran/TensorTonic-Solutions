import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    
    A = np.asarray(A)  # đảm bảo là numpy array
    
    rows = A.shape[0]
    cols = A.shape[1]
    
    # Tạo ma trận mới có kích thước đảo ngược
    result = np.zeros((cols, rows))
    
    for i in range(rows):
        for j in range(cols):
            result[j][i] = A[i][j]
    
    return result