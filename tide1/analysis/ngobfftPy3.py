import numpy as np
import numpy.fft as ft

# %%


def k_of_x(x):
    """k-vector with leading zero from x-vector
    also x-vector with leading zero from k-vector"""
    dx = x[1] - x[0]
    N = x.size
    dk = 2.*np.pi/(N*dx)
    inull = N//2
    k = dk*(np.linspace(1, N, N)-inull)

    return k

# %%


def obfft(x, f, n):
    """ Fast Fourier Transform of f living on x into Ff. The length of x and f
    should be an even number, preferably a power of two. x is always a one-
    dimensional array. The index of the zero mode is inull=N/2. fft is
    performed along the nth dimension of f.
    n = -1 for 1D ffts.
    """
    #HW: works best if x=0 is at the last point of x!
    dx = x[1] - x[0]
    N = x.size

    if N != np.shape(f)[n]:
        raise NameError('Wrong axis for the fft in obfft.')

    inull = N//2
    Ff = dx * np.roll(ft.fft(f, N, n), inull-1, n)

    return Ff

# %%


def obifft(k, Ff, n):
    """ Fast Fourier Transform of f living on x into Ff. The length of x and f
    should be an even number, preferably a power of two. x is always a one-
    dimensional array. The index of the zero mode is inull=N/2. fft is
    performed along the nth dimension of f.
    n = -1 for 1D ffts.
    """
    x = k_of_x(k)
    dx = x[1] - x[0]
    N = k.size

    if N != np.shape(Ff)[n]:
        raise NameError('Wrong axis for the ifft in obifft.')

    inull = N//2
    f = (1./dx) * ft.ifft(np.roll(Ff, 1-inull, n), N, n)

    return f

# %%


def obfft2(x, y, f):
    """ 2D Fast Fourier Transform of f living on (x,y) into Ff. The length of
    x,y  and f must be an even number, preferably a power of two. The index of
    the zero mode is inull=jnull=N/2."""
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Nx = x.size
    Ny = y.size
    inull = Nx//2
    # print 'inull = {}'.format(inull)
    jnull = Ny//2
    Ff = dx * dy * np.roll(np.roll(ft.fft2(f), inull-1, 0), jnull-1, 1)
    # Ff = dx * dy * np.roll(ft.fft2(f),jnull-1,0)
    # Ff = dx * dy * ft.fft2(f)
    # for ii in range(x.size): Ff[:,ii] = Ff[:,ii]*(2**ii)

    return Ff

# %%


def obifft2(k, l, Ff):
    """ 2D inverse FFT of f living on (k,l) into Ff. The length of
    k, l and Ff and f must be an even number, preferably a power of two."""
    x = k_of_x(k)
    y = k_of_x(l)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Nx = x.size
    Ny = y.size
    inull = Nx//2
    jnull = Ny//2
    f = 1. / (dx * dy) * ft.ifft2(np.roll(np.roll(Ff, 1-inull, 0), 1-jnull, 1))

    return f

# %%


def obfft3(x, y, z, f):
    """ 3D Fast Fourier Transform of f into Ff.  The length of
    x,y  and f must be an even number, preferably a power of two. The index of
    the zero mode is inull=jnull=kull=N/2.
    x,y,z better be vectors."""
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    Nx = x.size
    Ny = y.size
    Nz = z.size
    inull = Nx//2
    jnull = Ny//2
    knull = Nz//2
    Ff = dx * dy * dz * np.roll(np.roll(np.roll(np.fftn(f), inull-1, 0),
                                        jnull-1, 1), knull-1, 2)

    return Ff

# %%


def obifft3(k, l, m, Ff):
    """ 3D Fast Fourier Transform of f into Ff.  The length of
    x,y  and f must be an even number, preferably a power of two.  The index of
    the zero mode is inull=jnull=kull=N/2.
    x,y,z better be vectors."""
    x = k_of_x(k)
    y = k_of_x(l)
    z = k_of_x(m)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    Nx = x.size
    Ny = y.size
    Nz = z.size
    inull = Nx//2
    jnull = Ny//2
    knull = Nz//2
    f = 1./(dx * dy * dz) * np.ifftn(
        np.roll(np.roll(np.roll(Ff, 1-inull, 0), 1-jnull, 1), 1-knull, 2))

    return f
