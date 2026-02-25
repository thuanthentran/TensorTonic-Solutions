import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    N = len(seqs)

    # Nếu seqs rỗng
    if N == 0:
        return np.zeros((0, 0), dtype=int)

    # Xác định max_len
    if max_len is None:
        max_len = max((len(seq) for seq in seqs), default=0)

    # Khởi tạo mảng kết quả với pad_value
    result = np.full((N, max_len), pad_value, dtype=int)

    # Copy từng sequence vào (truncate nếu dài hơn max_len)
    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        if length > 0:
            result[i, :length] = seq[:length]

    return result