import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time

controls = genfromtxt('controls.txt', delimiter=',')
ml_controls = genfromtxt('ml_controls.txt', delimiter=',')

amount_joints = len(controls[0])

for joint in range(0,amount_joints):
    amount = 99

    plt.plot(controls[-amount:-1,joint])
    plt.plot(ml_controls[-amount:-1,joint])

    #plt.ion()
    fig = plt.show()
    #time.sleep(2.0);


    plt.close('all')
