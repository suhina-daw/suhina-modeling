# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Five sequential native DSP experiments. Vocal holdout is never used to select settings."""
from pathlib import Path
import json, hashlib, tempfile, subprocess, copy
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve,resample_poly,butter,sosfilt
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA, WORK, CAPTURES, PROBE
R=DATA
O=WORK/'round 13'
EXE=PROBE
REPO=Path(__file__).resolve().parents[3]
FS=12000

def read(p):
 sr,x=wavfile.read(p)
 assert sr==48000 and x.ndim==2 and x.shape[1]==2 and np.isfinite(x).all(),str(p)
 if np.issubdtype(x.dtype,np.integer):x=x.astype(float)/max(abs(np.iinfo(x.dtype).min),np.iinfo(x.dtype).max)
 return resample_poly(x.astype(float),1,4,axis=0)
def env(x):return np.sqrt(np.mean(x[:len(x)//120*120].reshape(-1,120,2)**2,axis=(1,2)))
def mono(x):return 10*np.log10(max(np.sum(x.sum(axis=1)**2)/(2*np.sum(x*x)),1e-15))
def bandpowers(x):
 z=abs(np.fft.rfft(x,axis=0))**2;f=np.fft.rfftfreq(len(x),1/FS)
 p=np.array([np.sum(z[(f>=lo)&(f<hi)]) for lo,hi in [(30,120),(120,500),(500,2000),(2000,5500)]])
 return p/max(np.sum(p),1e-30)
def measure(y,a):
 y=y[:len(a)];y=y*np.sqrt(np.sum(a*a)/max(np.sum(y*y),1e-30));e=env(y);t=env(a);mask=t>t.max()*.001
 error=float(np.mean(abs(20*np.log10(np.maximum(e[mask],1e-20)/t[mask]))))
 onset=int(np.flatnonzero(np.max(abs(a),axis=1)>np.max(abs(a))*.001)[0]);phase=[]
 for lo,hi in [(0,.08),(.08,.5),(.5,2),(2,6)]:
  sl=slice(onset+int(lo*FS),min(len(a),onset+int(hi*FS)))
  phase.append(float(10*np.log10(max(np.sum(y[sl]**2),1e-30)/max(np.sum(a[sl]**2),1e-30))))
 spectral=float(np.mean(abs(10*np.log10(np.maximum(bandpowers(y),1e-15)/np.maximum(bandpowers(a),1e-15)))))
 return dict(envelope_db=error,phase_db=phase,spectral_db=spectral,mono_db=abs(mono(y)-mono(a)),score=error+.15*np.mean(abs(np.array(phase[:3])))+.10*spectral+.05*abs(mono(y)-mono(a)))
def sources():
 data=[];src=CAPTURES/'samples'
 for sound in ['bass','clap','kick','piano','snare','serum-flute','vocals-female','vocals-male']:
  dp=next(p for p in src.glob(sound+'*.wav') if 'dry' in p.name);x=read(dp).mean(axis=1);a=read(src/f'{sound} 48k-stereo-20s - 6000ms.wav')
  data.append(dict(name=sound,x=x,a=a,holdout=sound.startswith('vocals'),slice=(0,len(a))))
 for sound in ['bass','saw','sine','square','triangle']:
  x=read(next((src/'pitch').glob(sound+' dry-*.wav'))).mean(axis=1);a=read(src/'pitch'/f'{sound} 48k-stereo-20s - 6000ms.wav')
  for i in range(5):data.append(dict(name=f'{sound}-pitch{i+1}',x=x,a=a[i*210000:(i+1)*210000],holdout=False,slice=(i*210000,(i+1)*210000)))
 return data

def render(settings,folder):
 cfg=folder/'settings.json';cfg.write_text(json.dumps({'long':settings,'benchmark':settings},indent=2))
 dest=folder/'audio';subprocess.run([str(EXE),str(dest),'--flat-decay','--diffused','--render-only','--impulse-only','--settings',str(cfg)],cwd=REPO,check=True,capture_output=True)
 return read(dest/'long-impulse-wet.wav')[FS:]/.25

def evaluate(h,data):
 rows=[];cache={}
 for d in data:
  key=(d['name'].split('-pitch')[0],len(d['x']))
  if key not in cache:cache[key]=np.column_stack([fftconvolve(d['x'],h[:,c])[:len(d['x'])] for c in range(2)])
  lo,hi=d['slice'];v=measure(cache[key][lo:hi],d['a']);v.update(name=d['name'],holdout=d['holdout']);rows.append(v)
 # Equal aggregate weight for original sources and pitch bank; vocals strictly excluded.
 orig=[v for v in rows if not v['holdout'] and '-pitch' not in v['name']];pitch=[v for v in rows if '-pitch' in v['name']]
 score=.5*np.mean([v['score'] for v in orig])+.5*np.mean([v['score'] for v in pitch])
 return dict(score=float(score),original_envelope=float(np.mean([v['envelope_db'] for v in orig])),pitch_envelope=float(np.mean([v['envelope_db'] for v in pitch])),vocal_envelope=float(np.mean([v['envelope_db'] for v in rows if v['holdout']])),cases=rows)

def run():
 data=sources();inventory=[]
 for p in (CAPTURES/'samples').glob('*vocal*.wav'):
  sr,x=wavfile.read(p);inventory.append(dict(file=p.name,frames=len(x),seconds=len(x)/sr,sample_rate=sr,channels=x.shape[1],peak=float(np.max(abs(x))),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 (O/'vocal-inventory.json').write_text(json.dumps(inventory,indent=2))
 base=json.loads((R/'round 10/conference-native-settings.json').read_text())['long']
 early=json.loads((R/'round 13/early-fit-diffusion0.3-settings.json').read_text())['long']
 stages=[('14','Early balance','Shape/early-late overlap: manual; independent feedforward patterns: Schlecht et al. 2023.'),('15','Input diffusion','Dattorro 1997 section 1.3: preconditioning percussive signals versus transient smearing.'),('16','Late delay timing','Modal excitation and FDN delay geometry: pitch-bank regression guards.'),('17','Frequency dependent decay','Prawda et al. 2019: attenuation filters govern frequency-dependent decay.'),('18','Early diffusion','Dattorro 1997: diffusion placement matters; distinguish early smoothing from tank density.')]
 selected=copy.deepcopy(base);summary=[]
 with tempfile.TemporaryDirectory(prefix='five-rounds-',dir=O) as temp:
  temp=Path(temp);bdir=temp/'baseline';bdir.mkdir();baseline=evaluate(render(base,bdir),data);(O/'baseline.json').write_text(json.dumps(baseline,indent=2))
  for number,title,source in stages:
   rd=R/f'round {number}';rd.mkdir(parents=True, exist_ok=True)
   if (rd/'results.json').exists():raise RuntimeError('Preserving completed round '+number)
   candidates=[('incumbent',copy.deepcopy(selected))]
   if number=='14':
    for factor in [.65,.8,1.,1.2,1.4]:
     s=copy.deepcopy(early);s['early_level']=factor;candidates.append((f'custom-early-{factor}',s))
    for factor in [.75,1.25]:
     s=copy.deepcopy(base);s['early_level']=factor;candidates.append((f'old-early-{factor}',s))
   elif number=='15':
    for v in [.1,.2,.4,.5,.6,.7]:
     s=copy.deepcopy(selected);s['diffusion']=v;candidates.append((f'input-{v}',s))
   elif number=='16':
    for v in [1.2,1.35,1.45,1.55,1.65,1.8]:
     s=copy.deepcopy(selected);s['late_scale']=v;candidates.append((f'late-{v}',s))
   elif number=='17':
    for decay,ratio in [(4.8,1),(5.,1),(5.6,1),(5.9,1),(5.321,.9),(5.321,.75),(5.321,.6)]:
     s=copy.deepcopy(selected);s.update(decay_seconds=decay,high_decay_ratio=ratio);candidates.append((f'decay-{decay}-high-{ratio}',s))
   else:
    for v in [.1,.2,.3,.4,.5,.6]:
     s=copy.deepcopy(selected);s['early_diffusion']=v;candidates.append((f'early-diffusion-{v}',s))
   records=[]
   for i,(name,s) in enumerate(candidates):
    folder=temp/f'{number}-{i}';folder.mkdir();v=evaluate(render(s,folder),data);v.update(candidate=name,settings=s)
    # No source-family average may regress >0.35 dB against Round10. No original case >0.75 dB.
    v['eligible']=v['original_envelope']<=baseline['original_envelope']+.35 and v['pitch_envelope']<=baseline['pitch_envelope']+.35 and all(r['envelope_db']<=b['envelope_db']+.75 for r,b in zip(v['cases'],baseline['cases']) if not r['holdout'] and '-pitch' not in r['name'])
    records.append(v);print(number,name,'score',round(v['score'],3),'vocal',round(v['vocal_envelope'],3),'eligible',v['eligible'],flush=True)
   eligible=[r for r in records if r['eligible']];winner=min(eligible,key=lambda r:r['score']);selected=winner['settings']
   report=dict(round=int(number),question=title,source=source,selection='Original and pitch composite only; vocals held out. No listening-based pass claim.',selected=winner['candidate'],candidates=records)
   (rd/'results.json').write_text(json.dumps(report,indent=2));(rd/'selected-settings.json').write_text(json.dumps({'long':selected,'benchmark':selected},indent=2));summary.append(dict(round=int(number),question=title,selected=winner['candidate'],score=winner['score'],vocal_envelope=winner['vocal_envelope'],original_envelope=winner['original_envelope'],pitch_envelope=winner['pitch_envelope']))
   print('WINNER',number,summary[-1],flush=True)
 (O/'five-round-summary.json').write_text(json.dumps(summary,indent=2))
 print('FIVE ROUNDS COMPLETE',flush=True)
if __name__=='__main__':run()

