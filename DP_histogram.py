import numpy as np


'''
Input:  file name
output: 2D numpy array
'''


def read_data(fileName):
    data_set = np.loadtxt(fileName)

    return data_set


# def create_neighbors(db, index_to_remove):
#     return np.concatenate((db[0:index_to_remove], db[index_to_remove+1:]))


# def get_neighboring_databases(db):
#     neighboring_databases = list()
#     for i in range(len(db)):
#         ndb = create_neighbors(db, i)
#         neighboring_databases.append(ndb)
#     return neighboring_databases
def laplace_mechanism(count, sensitivity, epsilon):
    beta = sensitivity / epsilon
    noise = np.random.laplace(0, beta, 1)
    dp_count = count + np.round(noise).astype(int)
    if dp_count > 0:
        return count + np.round(noise).astype(int)
    else:
        return 0


def mean_squared_error(orig_count, dp_count):
    n = len(orig_count)
    total = 0
    for i in range(n):
        total = total+(orig_count[i]-dp_count[i])**2
    return total/n


def report_noisy_max(data, bins, epsilon):
    # set the add noise function to map it to every query
    addNoise = np.vectorize(laplace_mechanism)
    # create the orginal hisogram
    hist, bins = np.histogram(
        data, bins=bins)
    # add noise to the histogram
    dp_hist = addNoise(hist, 1, epsilon)
    # find the max count
    max_index = np.argmax(dp_hist)
    # return the interval
    return bins[max_index: max_index+2]


if __name__ == "__main__":
    # set the value of epsilon
    epsilon = 0.1
    # set the file
    data = read_data('ipums.txt')
    print("\n")

    addNoise = np.vectorize(laplace_mechanism)

    # the original Histogram and DP Histogram for age
    print("Age original Histogram")
    hist, bins = np.histogram(
        data[:, 0], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    print(hist)
    print(bins)

    print("Age DP Histogram")
    dp_hist = addNoise(hist, 1, epsilon)
    print(dp_hist)
    print(bins)
    print("\n")
    print("Mean Squared Error = ", mean_squared_error(hist, dp_hist))
    print("\n\n")

    # the original Histogram and DP Histogram for Gender
    print("Gender original Histogram")
    hist, bins = np.histogram(
        data[:, 1], bins=[1, 2, 3])
    print(hist)
    print(bins)

    print("Gender DP Histogram")
    dp_hist = addNoise(hist, 1, epsilon)
    print(dp_hist)
    print(bins)
    print("\n")
    print("Mean Squared Error = ", mean_squared_error(hist, dp_hist))
    print("\n\n")

    # the original Histogram and DP Histogram for Income(K)
    print("Income(K) original Histogram")
    hist, bins = np.histogram(
        data[:, 2], bins=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    print(hist)
    print(bins)

    print("Income(K) DP Histogram")
    dp_hist = addNoise(hist, 1, epsilon)
    print(dp_hist)
    print(bins)

    print("\n")
    print("Mean Squared Error = ", mean_squared_error(hist, dp_hist))
    print("\n\n")

    print("Private Max Response for Age")
    print("Age Range With Highest Population = ", report_noisy_max(
        data[:, 0], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], epsilon))


# since the histogram is a parelal com the number of bins will only incress the propalty to gite the maximum noise
