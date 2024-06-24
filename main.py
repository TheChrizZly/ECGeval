import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.fft import fft, fftfreq
from src.patient import Patient

# Parameters

christian = Patient('christian', 40, 90, "HO34N")
alice = Patient('alice', 50, 120, "KNP19")
maxi = Patient('max', 60, 100, "SDF32")
marina = Patient('marina', 40, 120, "KJH23")
sarah = Patient('sarah', 50, 120, "Q27DW")
tim = Patient('tim', 40, 100, "LB411")
patients = [christian, alice, maxi, marina, sarah, tim]
handgemessene_hrT = [68, 84, 76, 76, 88, 60, 56, 88, 80, 68, 76, 56]
hangemesse_hrM = [64, 84, 56, 88, 76, 64, 64, 72, 60, 80, 68, 56]
# Plotting
# compare average heart rate before and after exercise
avgHRPre = [patient.avg_m_hr_pre for patient in patients]
avgHRPost = [patient.avg_m_hr_post for patient in patients]
patient_names = [patient.id for patient in patients]
avgDiff = np.array(avgHRPost) - np.array(avgHRPre)
avgDiff = avgDiff.tolist()
data = pd.DataFrame({
    'Patient': patient_names * 2,  # Duplicate patient IDs for before/after
    'Heart Rate': avgHRPre + avgHRPost,  # Combine heart rates
    'Condition': ['Before Exercise'] * len(patients) + ['After Exercise'] * len(patients)  # Add labels
})
plt.rcParams.update({'font.size': 30})
# Plotting with Seaborn
plt.figure(figsize=(10, 8))
sns.lineplot(data=data, x='Patient', y='Heart Rate', hue='Condition', marker='o', palette=['skyblue', 'salmon'])  # Line plot with hues

# Add labels and title
plt.xlabel('Patient')
plt.ylabel('Heart Rate (bpm)')
plt.title('Average Heart Rate Before and After Exercise')
plt.xticks(rotation=45)

# Barplot
dataBar = pd.DataFrame({
    'Patient': patient_names,
    'Heart Rate Difference': avgDiff,
    'Condition': ['Trained', 'Untrained', 'Untrained', 'Trained', 'Untrained', 'Trained']
})
plt.figure(figsize=(10, 8))
sns.barplot(data=dataBar, x='Patient', y='Heart Rate Difference', hue='Condition', palette=['skyblue', 'salmon'])

plt.xlabel('Versuchsperson')
plt.ylabel('Differenz in der Herzfrequenz (bpm)')
plt.title('Differenz in der Herzfrequenz vor und nach der Progressiven Muskelentspannung')
plt.xticks(rotation=45)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.ylim(-21, 8)
plt.grid(axis='y', linestyle='--')    # Add horizontal grid
for index, row in dataBar.iterrows():
    plt.text(index, row['Heart Rate Difference'], round(row['Heart Rate Difference'],1), ha='center', va='bottom')
plt.savefig('BarPlotMuskelentspannung.png', dpi=600)

dataBarT = pd.DataFrame({
    'Patient': patient_names,
    'Heart Rate Difference': avgDiff,
    'Condition': ['Trained', 'Untrained', 'Untrained', 'Trained', 'Untrained', 'Trained']
})
avgHRPreT = [patient.avg_t_hr_pre for patient in patients]
avgHRPostT = [patient.avg_t_hr_post for patient in patients]
patient_names = [patient.id for patient in patients]
avgDiffT = np.array(avgHRPostT) - np.array(avgHRPreT)
avgDiffT = avgDiffT.tolist()
dataBarT = pd.DataFrame({
    'Patient': patient_names,
    'Heart Rate Difference': avgDiffT,
    'Condition': ['Trained', 'Untrained', 'Untrained', 'Trained', 'Untrained', 'Trained']
})
plt.figure(figsize=(10, 8))

sns.barplot(data=dataBarT, x='Patient', y='Heart Rate Difference', hue='Condition', palette=['skyblue', 'salmon'])

plt.xlabel('Versuchsperson')
plt.ylabel('Differenz in der Herzfrequenz (bpm)')
plt.title('Differenz in der Herzfrequenz vor und nach der Traumreise')
plt.xticks(rotation=45)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.ylim(-21, 8)
plt.grid(axis='y', linestyle='--')    # Add horizontal grid
# Add value labels to the bars
for index, row in dataBarT.iterrows():
    plt.text(index, row['Heart Rate Difference'], round(row['Heart Rate Difference'],1), ha='center', va='bottom')
plt.savefig('BarPlotTraumreise.png', dpi=600)

# BarPlot Handgemessen
avgHRPreHT = handgemessene_hrT[:6]
avgHRPostHT = handgemessene_hrT[-6:]
patient_names = [patient.id for patient in patients]
avgDiffHT = np.array(avgHRPostHT) - np.array(avgHRPreHT)
avgDiffHT = avgDiffHT.tolist()
dataBarHT = pd.DataFrame({
    'Patient': patient_names,
    'Heart Rate Difference': avgDiffHT,
    'Condition': ['Trained', 'Untrained', 'Untrained', 'Trained', 'Untrained', 'Trained']
})
plt.figure(figsize=(10, 8))

sns.barplot(data=dataBarHT, x='Patient', y='Heart Rate Difference', hue='Condition', palette=['skyblue', 'salmon'])

plt.xlabel('Versuchsperson')
plt.ylabel('Differenz in der Herzfrequenz (bpm)')
plt.title('Differenz in der handgemessenen Herzfrequenz vor und nach der Traumreise')
plt.xticks(rotation=45)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.ylim(-12, 5)
plt.grid(axis='y', linestyle='--')    # Add horizontal grid
# Add value labels to the bars
for index, row in dataBarHT.iterrows():
    plt.text(index, row['Heart Rate Difference'], round(row['Heart Rate Difference'],1), ha='center', va='bottom')
plt.savefig('BarPlotTraumreise_Handgemessen.png', dpi=600)

# BarPlot Handgemessen
avgHRPreHM = hangemesse_hrM[:6]
avgHRPostHM = hangemesse_hrM[-6:]
patient_names = [patient.id for patient in patients]
avgDiffHM = np.array(avgHRPostHM) - np.array(avgHRPreHM)
avgDiffHM = avgDiffHM.tolist()
dataBarHM = pd.DataFrame({
    'Patient': patient_names,
    'Heart Rate Difference': avgDiffHM,
    'Condition': ['Trained', 'Untrained', 'Untrained', 'Trained', 'Untrained', 'Trained']
})
plt.figure(figsize=(10, 8))

sns.barplot(data=dataBarHM, x='Patient', y='Heart Rate Difference', hue='Condition', palette=['skyblue', 'salmon'])

plt.xlabel('Versuchsperson')
plt.ylabel('Differenz in der Herzfrequenz (bpm)')
plt.title('Differenz in der handgemessenen Herzfrequenz vor und nach der Progressiven Muskelentspannung')
plt.xticks(rotation=45)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.ylim(-12, 5)
plt.grid(axis='y', linestyle='--')    # Add horizontal grid
# Add value labels to the bars
for index, row in dataBarHM.iterrows():
    plt.text(index, row['Heart Rate Difference'], round(row['Heart Rate Difference'],1), ha='center', va='bottom')
plt.savefig('BarPlotProgMus_Handgemessen.png', dpi=600)

plt.show()

#Print std dev for all patients
print('Standard Deviation of Heart Rate for all patients:')
stdDevMean = 0
for patient in patients:
    stdDevMean = stdDevMean + patient.std_m_hr + patient.std_t_hr
    print(patient.name + ' M: ' + str(patient.std_m_hr))
    print(patient.name + ' T: ' + str(patient.std_t_hr))
print('Mean Std: ' + str(stdDevMean/(len(patients)*2)))

