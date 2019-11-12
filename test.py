import numpy as np
import em
import common

X = np.loadtxt("input.txt")
# X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

mixture, _ = common.init(X, K, seed)

# print("Input:")
# print('X:\n' + str(X))
# print('K: ' + str(K))
# print('Mu:\n' + str(mixture.mu))
# print('Var: ' + str(mixture.var))
# print('P: ' + str(mixture.p))
# print()

# print("After first E-step:")
post, ll = em.estep(X, mixture)
# print('post:\n' + str(post))
# print('LL:' + str(ll))
# print()

# print("After first M-step:")
mu, var, p = em.mstep(X, post, mixture)
# print('Mu:\n' + str(mu))
# print('Var: ' + str(var))
# print('P: ' + str(p))
# print()

# print("After a run")
(mu, var, p), post, ll = em.run(X, mixture, post)
print('Mu:\n' + str(mu))
print('Var: ' + str(var))
print('P: ' + str(p))
print('post:\n' + str(post))
print('LL: ' + str(ll))
X_pred = em.fill_matrix(X, common.GaussianMixture(mu, var, p))
# error = common.rmse(X_gold, X_pred)
# print("X_gold:\n" + str(X_gold))
# X_pred = np.round(X_pred)
fil = open('u2.base','w')
# fil.write(str(n)+' '+str(d)+'\n')
for i in range(len(X_pred)):
	for j in range(len(X_pred[i])):
		tmp = int(round(X_pred[i][j]))
		if tmp == 0:
			tmp = 1
		fil.write(str(i+1)+"   ")
		fil.write(str(j+1)+"   "+str(tmp)+"   874965758\n")
# print("RMSE: " + str(error))
