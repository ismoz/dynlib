import math

def rhs(t, y_vec, dy_out, params, runtime_ws):
    _sum0 = 0.0
    for i in range(1, params[4] + 1):
        _sum0 += int(y_vec[0] + (2 * i - 1) > 0) - int(y_vec[0] + (2 * i - 1) < 0) + (int(y_vec[0] - (2 * i - 1) > 0) - int(y_vec[0] - (2 * i - 1) < 0))
    dy_out[0] = params[2] * (params[6] * math.sin(2 * 3.141592653589793 * params[5] * t)) - params[3] * (y_vec[0] if params[4] == 0 else y_vec[0] - _sum0)