def init():
    import requests
    import numpy as np
    import json
    import pandas as pd
    import matplotlib.pyplot as mplt
    import plotly.express as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import seaborn as sns
    import datetime as dt
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.mixture import GaussianMixture
    from sklearn.datasets import make_blobs
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.datasets import load_digits
    from sklearn import metrics
    pd.options.mode.chained_assignment = None  # default='warn'
    
def test():
    print("Hello!")