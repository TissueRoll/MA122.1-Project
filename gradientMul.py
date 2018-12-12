def gradientMultiply(a, b):
    a_rows = len(a)
    a_cols = len(a[0])
    b_rows = len(b)
    b_cols = len(b[0])
    if a_rows > 1 or b_rows > 1 or a_cols != b_cols:
        return None
    result = []
    for i in xrange(a_cols):
        result[0][i] = a[0][i]*b[0][i]
    return result
