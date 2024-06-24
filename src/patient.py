import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import seaborn as sns


class Patient:
    def __init__(self, name, filter_low, filter_high, id):
        self.name = name
        self.t_ecg_pre = None
        self.avg_t_hr_pre = None
        self.t_hr_pre = None
        self.time_t_pre = None
        self.t_ecg_post = None
        self.avg_t_hr_post = None
        self.t_hr_post = None
        self.time_t_post = None
        self.t_ecg_data = None
        self.avg_t_hr = None
        self.std_t_hr = None
        self.t_hr = None
        self.time_t = None
        self.m_ecg_pre = None
        self.avg_m_hr_pre = None
        self.m_hr_pre = None
        self.time_m_pre = None
        self.m_ecg_post = None
        self.avg_m_hr_post = None
        self.m_hr_post = None
        self.time_m_post = None
        self.m_ecg_data = None
        self.avg_m_hr = None
        self.std_m_hr = None
        self.m_hr = None
        self.time_m = None
        self.fft_t = None
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.id = id
        self.sampling_rate = 50  # Hz
        self.sampling_rate_pre = 200  # Hz
        file = ['/home/christian/PycharmProjects/EKGAuswertung/data/' + name + 'TV.csv',
        '/home/christian/PycharmProjects/EKGAuswertung/data/' + name + 'TW.csv',
        '/home/christian/PycharmProjects/EKGAuswertung/data/' + name + 'TN.csv',
        '/home/christian/PycharmProjects/EKGAuswertung/data/' + name + 'MV.csv',
        '/home/christian/PycharmProjects/EKGAuswertung/data/' + name + 'MW.csv',
        '/home/christian/PycharmProjects/EKGAuswertung/data/' + name + 'MN.csv']
        raw_data = []
        for i in file:
            raw_data.append(pd.read_csv(i, sep=';', decimal=','))
        self.t_ecg_pre = raw_data[0]["Lauf 1: Potential (mV)"].values
        self.t_ecg_post = raw_data[2]["Lauf 1: Potential (mV)"].values
        self.t_ecg_data = raw_data[1]["Lauf 1: Potential (mV)"].values
        self.m_ecg_pre = raw_data[3]["Lauf 1: Potential (mV)"].values
        self.m_ecg_post = raw_data[5]["Lauf 1: Potential (mV)"].values
        self.m_ecg_data = raw_data[4]["Lauf 1: Potential (mV)"].values

        self.t_hr_pre, self.avg_t_hr_pre, self.time_t_pre = self.calc_hr(self.t_ecg_pre, self.sampling_rate_pre)
        self.t_hr_post, self.avg_t_hr_post, self.time_t_post = self.calc_hr(self.t_ecg_post, self.sampling_rate_pre)
        self.t_hr, self.avg_t_hr, self.time_t = self.calc_hr(self.t_ecg_data, self.sampling_rate)
        self.m_hr_pre, self.avg_m_hr_pre, self.time_m_pre = self.calc_hr(self.m_ecg_pre, self.sampling_rate_pre)
        self.m_hr_post, self.avg_m_hr_post, self.time_m_post = self.calc_hr(self.m_ecg_post, self.sampling_rate_pre)
        self.m_hr, self.avg_m_hr, self.time_m = self.calc_hr(self.m_ecg_data, self.sampling_rate)

        self.yf = fft(self.t_ecg_data)
        self.xf = fftfreq(len(self.t_ecg_data), 1 / self.sampling_rate)[:len(self.t_ecg_data) // 2]  # Get positive frequencies
        self.std_m_hr = np.std(self.m_hr)
        self.std_t_hr = np.std(self.t_hr)

        '''
        # Heart Rate Plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=self.time_m, y=self.m_hr, label='Progressive Muskelentspannung', color="salmon")
        sns.lineplot(x=self.time_t, y=self.t_hr, label='Traumreise', color="skyblue")
        sns.lineplot(x=np.linspace(-60, 0, 2), y=[self.avg_m_hr_pre]*2, label='Herzfrequenz vor/nach Prog. Muskelenstpannung', color='red')
        sns.lineplot(x=np.linspace(720, 780, 2), y=[self.avg_m_hr_post]*2, color='red')
        sns.lineplot(x=np.linspace(-60, 0, 2), y=[self.avg_t_hr_pre]*2, label='Herzfrequenz vor/nach Traumreise', color='blue')
        sns.lineplot(x=np.linspace(720, 780, 2), y=[self.avg_t_hr_post]*2, color='blue')
        plt.title('Herzfrequenz während der Versuche: Person ID: ' + str(self.id))
        plt.ylim(20, 140)
        plt.xlabel('Zeit in Sekunden')
        plt.ylabel('Herzfrequenz in Schlägen pro Minute')
        plt.savefig(self.name+'HR_filtered.png', dpi=600)
        '''

    def calc_hr(self, ecg, sampling_rate):
        # ecg_c = nk.ecg_clean(ecg, sampling_rate=sampling_rate, method="elgendi2010")
        ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=sampling_rate, method="vg")
        peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        hr_raw = nk.ecg_rate(peaks, sampling_rate=sampling_rate)
        hr_low_filtered = hr_raw[np.where(hr_raw >= self.filter_low)]
        hr = hr_low_filtered[np.where(hr_low_filtered <= self.filter_high)]
        times = info['ECG_R_Peaks'] / sampling_rate
        times_low_filtered = times[np.where(hr_raw >= self.filter_low)]
        times_filtered = times_low_filtered[np.where(hr_low_filtered <= self.filter_high)]

        return hr, hr[:].mean(), times_filtered
