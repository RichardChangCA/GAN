import matplotlib
import matplotlib.pyplot as plt
# from scipy.interpolate import make_interp_spline, BSpline
import csv
import numpy as np
from scipy.signal import savgol_filter

model_name = "GAN"
dataset_name = "MNIST"
num_hidden = "1"
latent_size = "100"
net_name = "D"
color = 'r'

label_name = model_name  + "_" \
       + num_hidden + "_" + latent_size
plot_title = dataset_name

file_name = "csv/run-" + model_name + "_" + dataset_name + "_" \
       + num_hidden + "_" + latent_size + "_256-tag-" + model_name + "_" + net_name + "_loss.csv"

x_value = []
y_value = []
with open(file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
        #     print(f'Column names are {", ".join(row)}')
            line_count += 1
            continue
        x_value.append(int(row[1]))
        y_value.append(float(row[2]))
        line_count += 1
    # print(f'Processed {line_count} lines.')

# print(x_value)
x_value = np.array(x_value)
y_value = np.array(y_value)
plt.plot(x_value,y_value,label=label_name,alpha=0.3,color=color)
y_new = savgol_filter(y_value,51,1) #window size = 51
x_new = x_value
plt.plot(x_new,y_new,label=label_name,color=color)
# x_new = np.linspace(x_value.min(), x_value.max(),line_count+1)
# spl = make_interp_spline(x_value,y_value,k=5)
# y_new = spl(x_new)
# plt.plot(x_value,y_value,label=label_name + "new")

plt.xlabel("iterations")
plt.ylabel(net_name + "_loss")
plt.title(plot_title)
plt.legend()
# plt.show()
plt.savefig(model_name + "_" + dataset_name + "_" \
       + num_hidden + "_" + latent_size + "_" + net_name+ ".png")
# if plt.show(), then no savefig. if plt.show() disabled, then savefig works