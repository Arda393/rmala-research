"""Fixed partial memory correction, without changing retrieval or gate inputs."""
def blend(base,candidate,alpha):
    if not 0.<=alpha<=1.:raise ValueError('alpha must be in [0, 1]')
    if alpha==0.:return base
    if alpha==1.:return candidate
    return base+alpha*(candidate-base)
