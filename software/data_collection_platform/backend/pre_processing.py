from einops import rearrange,reduce
from scipy.signal import filtfilt, iirnotch, butter
import numpy as np

def bandpass(x,
			  fs,
			  low,
			  high,):
		
		nyquist = fs/2
		b,a = butter(4,[low/nyquist,high/nyquist],"bandpass",analog=False)
		n,d,t = x.shape
		x = rearrange(x,"n t d -> (n d) t")
		x = filtfilt(b,a,x)
		x = rearrange(x,"(n d) t -> n t d",n=n)
		return x

def base_preprocess(x,
                    y,
                    notch_freq=60,
                    low=4,
                    high=50,
                    fs=256):
    n,t,d = x.shape
    x = rearrange(x,"n t d -> (n d) t")
    b,a = iirnotch(notch_freq,30,fs)
    x = filtfilt(b,a,x)
    nyquist = fs/2
    b,a = butter(4,[low/nyquist,high/nyquist],"bandpass",analog=False)
    x = filtfilt(b,a,x)
    x = rearrange(x,"(n d) t -> n t d",d=d)
    return x,y

def csp_preprocess(x,
                   y,
                   notch_freq=60,
                   low=4,
                   high=50,
                   fs=256):
    
    x,y = base_preprocess(x,y,notch_freq,
                          low,high,fs)
    
    ax = []

    for i in range(1,10):
        ax.append(bandpass(x,fs,4*i,4*i+4))
    x = np.concatenate(ax,-1)
    mu = np.mean(x,axis=-1)
    sigma = np.std(x,axis=-1)
    x = (x-rearrange(mu,"n d -> n d 1"))/rearrange(sigma,"n d -> n d 1")
    return x,y
