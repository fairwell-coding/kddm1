import numpy as np
import pylab
from matplotlib import pyplot as plt
from numpy import ndarray
from statsmodels.graphics.gofplots import qqplot


def plot_rating_distribution(dataset):
    pass


def plot_rating_per_item(dataset):
    pass


def plot_rating_per_user(dataset):
    pass


def boxplot_rating_distribution(dataset):
    data = dataset.flatten(order='C')
    data = data[np.logical_not(np.isnan(data))]
    fig1, ax1 = plt.subplots()
    ax1.set_title('Distribution of Ratings')
    ax1.boxplot(data)
    plt.show()


def distogram_rating_distribution(dataset):
    data = dataset.flatten(order='C')
    data = data[np.logical_not(np.isnan(data))]
    fig1, ax1 = plt.subplots()
    ax1.set_title('Distribution of Ratings')
    ax1.hist(data, 40)
    plt.show()


def plot_joke_rating(dataset, additional_text=''):
    fig = plt.figure()
    data = np.nanmean(dataset, axis=0)
    fig.suptitle('Mean joke rating' + additional_text, fontsize=20)
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


def plot_local_outlier_factor(data, data_scores):
    plt.title("Local Outlier Factor (LOF)")
    plt.scatter(data[:, 0], data[:, 1], color="k", s=3.0, label="Data points")
    radius = (data_scores.max() - data_scores) / (data_scores.max() - data_scores.min())
    plt.scatter(
        data[:, 0],
        data[:, 1],
        s=1000 * radius,
        edgecolors="r",
        facecolors="none",
        label="Outlier scores",
    )
    plt.axis("tight")
    legend = plt.legend(loc="upper left")
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]
    plt.show()


def plot_local_outlier_factor(data, data_scores):
    plt.title("Local Outlier Factor (LOF)")
    plt.scatter(data[:, 0], data[:, 1], color="k", s=3.0, label="Data points")
    radius = (data_scores.max() - data_scores) / (data_scores.max() - data_scores.min())
    plt.scatter(
        data[:, 0],
        data[:, 1],
        s=1000 * radius,
        edgecolors="r",
        facecolors="none",
        label="Outlier scores",
    )
    plt.axis("tight")
    legend = plt.legend(loc="upper left")
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]
    plt.show()
