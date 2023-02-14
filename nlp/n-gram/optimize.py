'''
Filename: optimize.py
Author: s1866666, s1879523

Description: optimizes hyper-parameters used in our estimation methods
'''

from cw1 import *
import matplotlib.pyplot as plt
n=3

training_en = get_data("./assignment1-data/training.en")
training_es = get_data("./assignment1-data/training.es")
training_de = get_data("./assignment1-data/training.de")
test = get_data("./assignment1-data/test")

train_val_sets = [("training.en", training_en), 
				  ("training.es", training_es),
				  ("training.de", training_de)]

train_val_set = training_en
test_set = test

# set params
t = 1 # how many times we validate with each parameters
train_set_ratio = 0.8
step = 5
est = laplace_estimator(n)
opt = optimisor(train_set_ratio,t)
fig, ax = plt.subplots()


print("optimising laplace ...")
print("t =",t)
# training
for tv_set in train_val_sets:
	print(tv_set[0])
	aa = [ 1/step*(i+1) for i in range(step)]
	best_a, min_val_pp = opt.find_best_h_param(tv_set[1], est, aa)

	# print results
	print("\nMinimum val pp =", min_val_pp, "alpha =", best_a,'\n')

	# plot
	alphas = [log[0] for log in opt.log()]
	pps = [log[1] for log in opt.log()]
	opt.clear_log()
	ax.plot(alphas, pps, label=tv_set[0])

	
ax.grid()
ax.legend()
ax.set(xlabel='alpha', ylabel='average validation perplexity',
	       title='cross-validation results of add-alpha smoothing')
plt.show()
fig.savefig("cross-val.png")






print("\ninterpolation")
t=10
print("t={}".format(t))
est = interpolation_estimator(n)
opt = optimisor(train_set_ratio,t)

step = 10
alpha = 0.1
l = [ 1/step*i for i in range(step+1)]
ll = []
for i in l:
	for j in l:
		for m in l:
			if math.isclose(i+j+m,1):
				ll.append([alpha, i, j, m])

for tv_set in train_val_sets[1:]:
	print(tv_set[0])
	best_h_param,min_val_pp = opt.find_best_h_param(tv_set[1], est, ll)
	opt.clear_log()
	print("\nMinimum val pp =", min_val_pp ,"alpha =",best_h_param[0], "lambdas =", best_h_param[1:],'\n')





