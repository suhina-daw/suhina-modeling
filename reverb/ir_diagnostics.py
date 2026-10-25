# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Import-safe IR diagnostics. Threshold density is diagnostic, not a quality gate."""
import numpy as np
from scipy.signal import find_peaks
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE

def ir_diagnostics(x, fs=48000, decay_seconds=5.321):
 x=np.asarray(x,dtype=float)
 if x.ndim!=2 or x.shape[1]!=2 or not np.isfinite(x).all():raise ValueError('finite stereo IR required')
 window=round(fs*.01);n=len(x)//window*window;blocks=x[:n].reshape(-1,window,2);z=blocks-blocks.mean(axis=1,keepdims=True);std=z.std(axis=1,keepdims=True)
 density=np.mean(abs(z)>std,axis=(1,2))/.3173105079;energy=np.sum(blocks*blocks,axis=1)
 corr=np.sum(blocks[:,:,0]*blocks[:,:,1],axis=1)/np.maximum(np.sqrt(energy[:,0]*energy[:,1]),1e-30)
 balance=10*np.log10(np.maximum(energy[:,0],1e-30)/np.maximum(energy[:,1],1e-30))
 lo=round(fs*.1);hi=min(round(fs*3),len(x));late=x[lo:hi]*np.exp(np.arange(hi-lo)[:,None]/fs*3*np.log(10)/decay_seconds)
 fftsize=1<<(2*len(late)-1).bit_length();ac=np.fft.irfft(np.sum(abs(np.fft.rfft(late,n=fftsize,axis=0))**2,axis=1),n=fftsize)[:round(fs*.5)];ac/=max(ac[0],1e-30)
 start=round(fs*.005);indices,_=find_peaks(abs(ac[start:]),distance=start);indices+=start;indices=sorted(indices,key=lambda i:abs(ac[i]),reverse=True)[:8]
 return {'times_seconds':(np.arange(len(density))*.01).tolist(),'threshold_density':density.tolist(),'lr_balance_db':balance.tolist(),'correlation':corr.tolist(),'recurrence_lag_ms':(np.arange(len(ac))*1000/fs).tolist(),'recurrence':ac.tolist(),'recurrence_peaks':[{'lag_ms':float(i*1000/fs),'correlation':float(ac[i])} for i in indices]}
