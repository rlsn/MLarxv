'''
Filename: n_gram.py
Author: s1866666, s1879523

Description: uses various n-grams to estimate data
'''

from cw1 import *

training_en = get_data("./assignment1-data/training.en")
training_es = get_data("./assignment1-data/training.es")
training_de = get_data("./assignment1-data/training.de")
test = get_data("./assignment1-data/test")

train_set = training_en
test_set = test

seed = random.randrange(sys.maxsize) # seed to generate random text

n=4

alpha = 1
print("\nsmooth: add-one")
for i in range(1, n+1):
	est = laplace_estimator(i, test=True).set_h_param(alpha)
	print("n = ", i)
	model = est(train_set)
	print("PP on test set= ", model.compute_perplexity(test_set))
	print("random generation:\n")
	print(model.generate_from_LM(seed=seed),'\n')

print("\nseed:",seed)



