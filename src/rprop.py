def rprop(val_delta,
          grad,
          initial_stepsize=0.1,
          eta_plus=1.2,
          eta_minus=0.5,
          max_stepsize=50.0,
          min_stepsize=1e-6):
    if not grad:
        return

    val = val_delta[0]
    delta = val_delta[1]

    if delta:
        if grad < 0.:
            if delta < 0.:
                delta *= eta_plus

                if -delta > max_stepsize:
                    delta = -max_stepsize
            else:
                delta *= -eta_minus

                if -delta < min_stepsize:
                    delta = -min_stepsize
        else:
            if delta < 0.:
                delta *= -eta_minus

                if delta < min_stepsize:
                    delta = min_stepsize
            else:
                delta *= eta_plus

                if delta > max_stepsize:
                    delta = max_stepsize
    else:
        if grad < 0.:
            delta = -initial_stepsize
        else:
            delta = initial_stepsize

    val += delta

    val_delta[0] = val
    val_delta[1] = delta
