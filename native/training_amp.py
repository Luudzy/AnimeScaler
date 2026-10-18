"""Small AMP optimizer boundary shared by native training entry points."""


def amp_optimizer_step(scaler, optimizer):
    previous_scale=scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale()>=previous_scale
