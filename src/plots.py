import numpy as np
from matplotlib import pyplot as plt
import pylab
from statsmodels.graphics.gofplots import qqplot


def plot_joke_rating(dataset):
    fig = plt.figure()
    data = np.nanmean(dataset, axis=0)
    fig.suptitle('Mean joke rating', fontsize=20)
    plt.xlabel('Joke')
    plt.ylabel('Mean rating')
    plt.bar(np.arange(100), data)
    plt.show()


def plot_individual_joke_rating(joke_rating, joke_nr):
    fig = plt.figure()
    fig.suptitle('Distribution rating for ' + str(joke_nr) + '. joke', fontsize=20)
    plt.xlabel('Rating')
    plt.ylabel('Amount of Votes')
    plt.hist(np.where(joke_rating == 99, np.nan, joke_rating), bins=100)
    plt.show()


def plot_qq_individual_joke(joke_rating, joke_nr):
    measurements = np.where(joke_rating == 99, np.nan, joke_rating).astype(float)
    qqplot(measurements, line='s')
    pylab.suptitle('Quantile-Quantile Plot for ' + str(joke_nr) + '. joke')
    pylab.show()
    pass
