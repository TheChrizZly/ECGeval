import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Parameters
samplingRatePre = 50  # In Hz
samplingRateDuring = 50  # In Hz
# import CSV as dataframe
rawdata = pd.read_csv('data/christianTW.csv', sep=';', decimal=',')

# Extract the ECG signal
ecg_signal = rawdata["Lauf 1: Potential (mV)"].values
cleanECG = nk.ecg_clean(ecg_signal, sampling_rate=samplingRatePre)

# Peak detection and basic analysis
peaks, info = nk.ecg_peaks(cleanECG, sampling_rate=samplingRatePre)
hrv = nk.hrv(peaks, sampling_rate=samplingRatePre)

# Get heart rate
heart_rate = nk.ecg_rate(peaks, sampling_rate=samplingRatePre)
# Extract the numerical value of the heart rate from the array
average_heart_rate = heart_rate[:].mean()
peak_times = info['ECG_R_Peaks'] / samplingRatePre
# Now you can use the f-string for formatting
print(f"Average Heart Rate: {average_heart_rate:.2f} bpm")


# Get time points of R-peaks
r_peaks_time = peaks['ECG_R_Peaks'] / samplingRatePre

# Plot the cleaned ECG signal and the detected R-peaks
#nk.events_plot(r_peaks_time, cleanECG)

# Plot HRV analysis (you can explore different HRV metrics)
#nk.hrv_plot(hrv)

HeartRateData = pd.data = {'Time (seconds)': peak_times, 'Average Heart Rate (bpm)': heart_rate}
heartrate = pd.DataFrame(HeartRateData)
plt.figure()
sns.lineplot(x='Lauf 1: Zeit (min)', y='Lauf 1: Potential (mV)', data=rawdata, marker='x')

plt.figure()
sns.set_theme(style="whitegrid")
sns.lineplot(x='Time (seconds)', y='Average Heart Rate (bpm)', data=heartrate, marker='o')
plt.title('Average Heart Rate Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Average Heart Rate (bpm)')
plt.show()