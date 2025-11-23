def rk4_stepper(
    t, dt,
    y_curr, rhs,
    params,
    runtime_ws,
    ws,
    stepper_config,
    y_prop, t_prop, dt_next, err_est
):
    # RK4: classic 4-stage explicit method
    # stepper_config is ignored (RK4 has no runtime config)
    n = y_curr.size

    k1 = ws.k1
    k2 = ws.k2
    k3 = ws.k3
    k4 = ws.k4
    y_stage = ws.y_stage

    # Stage 1: k1 = f(t, y)
    rhs(t, y_curr, k1, params, runtime_ws)

    # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
    for i in range(n):
        y_stage[i] = y_curr[i] + 0.5 * dt * k1[i]
    rhs(t + 0.5 * dt, y_stage, k2, params, runtime_ws)

    # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
    for i in range(n):
        y_stage[i] = y_curr[i] + 0.5 * dt * k2[i]
    rhs(t + 0.5 * dt, y_stage, k3, params, runtime_ws)

    # Stage 4: k4 = f(t + dt, y + dt * k3)
    for i in range(n):
        y_stage[i] = y_curr[i] + dt * k3[i]
    rhs(t + dt, y_stage, k4, params, runtime_ws)

    # Combine: y_prop = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    for i in range(n):
        y_prop[i] = y_curr[i] + (dt / 6.0) * (
            k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]
        )

    # Fixed step: dt_next = dt
    t_prop[0] = t + dt
    dt_next[0] = dt
    err_est[0] = 0.0

    return OK
