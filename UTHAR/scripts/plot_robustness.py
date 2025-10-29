# plot_robustness.py
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('results/tut_results.csv')
plt.plot(results.corruption, results.accuracy, marker='o', label='TUT')
plt.xlabel('Packet Drop Probability')
plt.ylabel('Accuracy')
plt.title('Temporal Robustness Curve: TUT (UT-HAR)')
plt.grid()
plt.savefig('results/robustness_curve.png')
