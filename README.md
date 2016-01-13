# Load-Prediction-Scaife-Hall:
This is aimed at replicating the model developed in a research paper and applying it to Scaife hall power consumption.
The model developed in the paper, consists of two major features,
a) A time of week indicator variable and
b) A piecewise linear and continuous outdoor air temperature dependence.
It consists of two loading conditions, a) Occupied and b) Unoccupied, which are considered at the same time based on occupancy times.

Once the prediction is performed anomalies are detected based on a set threshold. This is calculated using a k-fold (here k=10) cross-validation method.

# NILM:
The repository also consists of the Lab-View code for data acquisition of the Light intensity. 
Realtime.vi, shows the short time Fourier Transformation (STFT) of the Light in Realtime, while instruments are turned on/off.
Plots.ipynb is a snapshot of the realtime STFT results, for the different equipments (Vacuum, Hot iron and Fan) tested.

# Poster:
Both of the above are summarized in the Poster present on this repository.
