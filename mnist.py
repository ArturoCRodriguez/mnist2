import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784",version=1)
print(mnist.keys)