import numpy as np
import neurokit2 as nk

raw_signal_path = "yourpath"
merged_beat_save_path = "yourpath"

def read_rpeaks(ecg_signal):
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=500)
    r_peak = rpeaks["ECG_R_Peaks"]
    return r_peak

def read_left_right(ecg_signal,r_peak,point = 230):
    lead_r = np.zeros(point*2)
    #If the R wave is too close to the end and is incomplete, discard it
    if r_peak[0] < (point):
        r_peak = r_peak[1:]
    if (r_peak[-1] + point) > 5000:
        r_peak = r_peak[0:-1]
    if (r_peak[-1] + point) > 5000:
        r_peak = r_peak[0:-1]
    for i in range(0,len(r_peak)):
        lead_middle = ecg_signal[(r_peak[i]-point):(r_peak[i]+point)].reshape(1,point*2)
        lead_r = lead_middle+lead_r
        lead_rr = lead_r/(len(r_peak))

    return lead_rr[0]

def change_ECG_to_one_wave(ecg_signal, wavelen=0):
    all_lead = []

    # For some leads that cannot get the wave at all,
    # use the most effective lead instead to ensure the validity of the data from other leads
    ecg_tmp = nk.ecg_clean(ecg_signal[7], sampling_rate=500)
    r_peak_tmp = read_rpeaks(ecg_tmp)
    for i in range(0, ecg_signal.shape[0]):
        ecg_signal[i] = nk.ecg_clean(ecg_signal[i], sampling_rate=500)
        r_peak = read_rpeaks(ecg_signal[i])
        if (len(r_peak)<=1):
            r_peak = r_peak_tmp
        lead_r = read_left_right(ecg_signal[i], r_peak, point=wavelen)
        all_lead.append(lead_r)

    all_lead = np.array(all_lead)
    return all_lead

def creat_ecg_one_wave_data(data_path, save_path):
    wave_all_data = []
    diease_dt =  np.load(data_path, allow_pickle=True)
    for i in range(0, diease_dt.shape[0]):
        print(" x", i)
        wave_data = change_ECG_to_one_wave(diease_dt[i], 250)
        wave_all_data.append(wave_data)
    wave_all_data = np.array(wave_all_data)
    print(wave_all_data.shape)
    wave_all_data.dump(save_path)

creat_ecg_one_wave_data(raw_signal_path, merged_beat_save_path)
