import numpy as np

def vad_cough_simple(x: np.ndarray, sr: int = 16000, win=0.05, hop=0.01,
                     rms_th=0.02, flux_th=0.03):
    n = x.shape[0]
    w = int(win*sr); h = int(hop*sr)
    frames = np.lib.stride_tricks.sliding_window_view(x, w)[::h]
    rms = np.sqrt((frames**2).mean(axis=1))
    spec = np.abs(np.fft.rfft(frames*np.hanning(w), axis=1))
    flux = np.maximum(0, np.diff(spec, axis=0)).mean(axis=1)
    flux = np.pad(flux, (1,0))
    mask = (rms>rms_th) & (flux>flux_th)
    events = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            events.append((i*h/sr, (j*h+w)/sr))
            i = j
        else:
            i += 1
    return events
