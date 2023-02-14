'''
Filename: compare_languages.py
Author: s1866666, s1879523

Description: compares the difference between perplexities of languages using
two models
'''

from cw1 import *
n=3

training_en = get_data("./assignment1-data/training.en")
training_es = get_data("./assignment1-data/training.es")
training_de = get_data("./assignment1-data/training.de")
test = get_data("./assignment1-data/test")

alpha = 0
est_lap = laplace_estimator(n).set_h_param(alpha)

model_en = est_lap(training_en)
model_es = est_lap(training_es)
model_de = est_lap(training_de)

print("laplace alpha=", alpha)
print("PP_en =", model_en.compute_perplexity(test))
print("PP_es =", model_es.compute_perplexity(test))
print("PP_de =", model_de.compute_perplexity(test))

alpha = 0.1
est_lap = laplace_estimator(n).set_h_param(alpha)

model_en = est_lap(training_en)
model_es = est_lap(training_es)
model_de = est_lap(training_de)

print("laplace alpha=", alpha)
print("PP_en =", model_en.compute_perplexity(test))
print("PP_es =", model_es.compute_perplexity(test))
print("PP_de =", model_de.compute_perplexity(test))

alpha = 1
est_lap = laplace_estimator(n).set_h_param(alpha)

model_en = est_lap(training_en)
model_es = est_lap(training_es)
model_de = est_lap(training_de)

print("laplace alpha=", alpha)
print("PP_en =", model_en.compute_perplexity(test))
print("PP_es =", model_es.compute_perplexity(test))
print("PP_de =", model_de.compute_perplexity(test))

lambdas = [0.1,0.1,0.8]
est_int = interpolation_estimator(n, test=True).set_h_param([alpha]+lambdas)

model_en = est_int(training_en)
model_es = est_int(training_es)
model_de = est_int(training_de)
print("interpolation lambdas=", lambdas)
print("PP_en =", model_en.compute_perplexity(test))
print("PP_es =", model_es.compute_perplexity(test))
print("PP_de =", model_de.compute_perplexity(test))
