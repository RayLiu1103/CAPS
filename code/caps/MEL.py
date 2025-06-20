import os
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import KFold
from time import time
import csv
import pandas as pd
import sys
sys.path.append('..')
import pickle
from feature_env import FeatureEvaluator, base_path
def jFitnessFunction(feat, label, X, opts, task_type):
    # Default of [alpha; beta]
    ws = [0.9, 0.1]  # [1, 0]

    if 'ws' in opts:
        ws = opts['ws']

    # Check if any feature exist
    if np.sum(X == 1) == 0:
        return 1
    else:
        # Error rate
        error = jwrapper_KNN(feat[:, X == 1], label, opts, task_type)
        # Number of selected features
        num_feat = np.sum(X == 1)
        # Total number of features
        max_feat = len(X)
        # Set alpha & beta
        alpha = ws[0]
        beta = ws[1]
        # Cost function
        cost = alpha * error + beta * (num_feat / max_feat)
        # cost = error
        return cost

def jwrapper_KNN(sFeat, label, opts, task_type):
    if 'k' in opts:
        k = opts['k']
    else:
        k = 5

    kf = KFold(n_splits=5)
    Acc = []
    for train_index, test_index in kf.split(sFeat):
        xtrain, xvalid = sFeat[train_index], sFeat[test_index]
        ytrain, yvalid = label[train_index], label[test_index]
        # Training model
        if task_type == 'cls' or task_type == 'mcls':
            My_Model = KNeighborsClassifier(n_neighbors=k)
            My_Model.fit(xtrain, ytrain)
            # Prediction
            pred = My_Model.predict(xvalid)
            # Accuracy
            Acc.append(np.sum(pred == yvalid) / len(yvalid))
        else:
            My_Model = KNeighborsRegressor(n_neighbors=k)
            My_Model.fit(xtrain, ytrain)
            # Prediction
            score = My_Model.score(xvalid, yvalid)
            # Accuracy
            Acc.append(score)

    # Error rate
    error = 1 - np.mean(Acc)
    return error

def jMultiTaskPSO(feat, label, opts, task_type):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    c1 = 2  # cognitive factor
    c2 = 2  # social factor
    c3 = 2  # group social factor
    w = 0.9  # inertia weight
    Vmax = (ub - lb) / 2  # Maximum velocity

    if 'N' in opts:
        N = opts['N']
    else:
        N = 20
    if 'T' in opts:
        max_Iter = opts['T']
    else:
        max_Iter = 100
    if 'c1' in opts:
        c1 = opts['c1']
    if 'c2' in opts:
        c2 = opts['c2']
    if 'c3' in opts:
        c3 = opts['c3']
    if 'w' in opts:
        w = opts['w']
    if 'Vmax' in opts:
        Vmax = opts['Vmax']
    if 'thres' in opts:
        thres = opts['thres']

    # Objective function
    fun = jFitnessFunction
    # Number of dimensions
    dim = feat.shape[1]
    # Feature weight matrix
    weight = np.zeros(dim)
    # Initial
    X = np.random.uniform(lb, ub, (N, dim))
    V = np.zeros((N, dim))
    # Fitness
    fit = np.zeros(N)
    fitG = np.inf
    # Number of subpopulation
    numSub = 2
    fitSub = np.ones(numSub) * np.inf
    Xsub = np.zeros((numSub, dim))

    subSize = int(N / numSub)
    j = 0
    for i in range(N):
        fit[i] = fun(feat, label, X[i, :] > thres, opts, task_type)
        # SubBest update
        if fit[i] < fitSub[j]:
            Xsub[j, :] = X[i, :]
            fitSub[j] = fit[i]
        # Subpopulation update
        if (i + 1) % subSize == 0:
            j += 1
        # Gbest update
        if fit[i] < fitG:
            Xgb = X[i, :]
            fitG = fit[i]

    Xpb = X.copy()
    fitP = fit.copy()
    curve = np.zeros(max_Iter + 1)
    curve[0] = fitG
    fnum = np.zeros(max_Iter + 1)
    fnum[0] = np.sum(Xpb[0, :] > thres)
    t = 1

    while t <= max_Iter:
        k = 0
        for i in range(N):
            if k == 0:  # Subpopulation 1
                for d in range(dim):
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    r3 = np.random.rand()
                    # Velocity update (2a)
                    VB = w * V[i, d] + c1 * r1 * (Xpb[i, d] - X[i, d]) + \
                         c2 * r2 * (Xgb[d] - X[i, d]) + c3 * r3 * (Xsub[1, d] - X[i, d])
                    # Velocity limit
                    VB = np.clip(VB, -Vmax, Vmax)
                    V[i, d] = VB
                    X[i, d] = X[i, d] + V[i, d]
                # Boundary
                XB = X[i, :]
                XB[XB > ub] = ub
                XB[XB < lb] = lb
                X[i, :] = XB
            else:  # Subpopulation 2
                index = weight < 0
                valued_features = weight.copy()
                valued_features[index] = 0  # Let the features with value less than 0 equal to 0
                sum_values = np.sum(valued_features)
                for d in range(dim):
                    p = np.random.rand()  # Probability
                    if valued_features[d] / sum_values > p:
                        X[i, d] = 1
                    else:
                        X[i, d] = 0
            # Fitness
            fit[i] = fun(feat, label, X[i, :] > thres, opts, task_type)
            # Feature vector (new)
            fv_n = X[i, :] > thres
            # Feature vector (old)
            fv_o = Xpb[i, :] > thres
            # Pbest update
            if fit[i] < fitP[i]:
                increase_acc = fitP[i] - fit[i]
                Xpb[i, :] = X[i, :]
                fitP[i] = fit[i]
                # Changes in features (1:new features emerge, -1:old features disappear)
                change = np.logical_xor(fv_n, fv_o)
                case_emerge = np.where(change & (fv_n > fv_o))[0]
                case_disapper = np.where(change & (fv_n < fv_o))[0]
                # Calculate the weight of those features
                for j in case_emerge:
                    weight[j] += increase_acc
                for j in case_disapper:
                    weight[j] -= increase_acc
            else:
                decrease_acc = fit[i] - fitP[i]
                # Changes in features (1:old features, -1:new features)
                change = np.logical_xor(fv_o, fv_n)
                case_emerge = np.where(change & (fv_o > fv_n))[0]
                case_disapper = np.where(change & (fv_o < fv_n))[0]
                # Calculate the weight of those features
                for j in case_emerge:
                    weight[j] -= decrease_acc
                for j in case_disapper:
                    weight[j] += decrease_acc
            # SubBest update
            if fit[i] < fitSub[k]:
                Xsub[k, :] = X[i, :]
                fitSub[k] = fit[i]
            # Subpopulation update
            if (i + 1) % subSize == 0:
                k += 1
            # Gbest update
            if fitP[i] < fitG:
                Xgb = Xpb[i, :]
                fitG = fitP[i]
        curve[t] = fitG
        fnum[t] = np.sum(Xgb > thres)
        print(f"Iteration {t} Best (MEL)= {curve[t]}")
        t += 1

    # Select features based on selected index
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    choice = (Xgb > thres).astype(int)
    sFeat = feat[:, Sf]
    # Store results
    results = {
        'curve': curve,
        'fnum': fnum,
        'fitG': fitG,
        'nf': len(Sf),
        'sf': Sf,
        'ff': sFeat,
        'choice': choice
    }
    return results

def Training(p_name, i, fe):
    start_time = time()
    np.seterr(all='ignore')
    # traindata = loadmat(os.path.join('data', f'{p_name}.mat'))[p_name]

    feat = fe.train.iloc[:, :-1].to_numpy()
    label = fe.train.iloc[:, -1].to_numpy()

    opts = {
        'k': 3,  # Number of k in K-nearest neighbor
        'N': 20,  # Number of solutions
        'T': 100,  # Number of iterations in each phase
        'thres': 0.6
    }

    PSO = jMultiTaskPSO(feat, label, opts, fe.task_type)
    curve, fnum, fitG, nf, sf, ff, choice = PSO.values()

    # np.save(f'curve{p_name}{i}.npy', curve)
    # np.save(f'fnum{p_name}{i}.npy', fnum)

    t = 1 - fitG
    print(t)
    results = {
        'optimized_Accuracy': 1 - fitG,
        'selected_Features': nf,
        'time': f"{time() - start_time:.2f}",
        'choice': choice
    }

    return results

def saveResults(results):
    file_path = os.path.join(os.getcwd(), 'results.csv')
    fieldnames = ['Data Set', 'Avg Accuracy', 'Selected Features', 'Running Time']

    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'Data Set': results['p_name'],
            'Avg Accuracy': results['optimized_Accuracy'],
            'Selected Features': results['selected_Features'],
            'Running Time': results['time']
        })

def main():
    # os.system('clear')
    # os.system('cls')
    # os.system('close')
    num_run = 10
    TASK_LIST = ['svmguide3','german_credit', 'uci_credit_card','spam_base', 'megawatt1','ionosphere',
                  'mice_protein','coil-20','minist_fashion', 'urbansound8k',
                 'openml_589', 'openml_616','IQdataset']
    problems = ['minist_fashion', 'urbansound8k']

    for p_name in problems:
        save_path = f'{base_path}/history/{p_name}/mel/'
        os.makedirs(save_path, exist_ok=True)
        ps_list = []
        with open(f'{base_path}/history/{p_name}/new_fe.pkl', 'rb') as f:
            fe: FeatureEvaluator = pickle.load(f)
        for i in range(num_run):
            results = Training(p_name, i, fe)
            results['p_name'] = p_name
            # saveResults(results)
            print(results)
            best_train = fe.generate_data(results['choice'],  flag='train')
            best_test = fe.generate_data(results['choice'], flag='test')
            performance = fe.get_performance(best_test) * 100
            print('The best performance on {} is {:.2f} found best on test : {}'.format(p_name, performance, results['choice']))
            ps_list.append(performance)
            file = open(os.path.join(save_path, f"mel_result_{i}.txt"), "w")
            file.write(str(performance) + "\n")
            file.close()
        best_train.to_hdf(f'{base_path}/history/{p_name}.hdf', key=f'mel_train')
        best_test.to_hdf(f'{base_path}/history/{p_name}.hdf', key=f'mel_test')
        average = sum(ps_list) / len(ps_list)
        print(f'Average performance on {p_name} is {average:.2f}%')

        file = open(os.path.join(save_path, "mel_result_average.txt"), "w")
        file.write(str(average) + "\n")
        file.close()
if __name__ == "__main__":
    main()


