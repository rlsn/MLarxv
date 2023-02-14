'''
Filename: generate_text.py
Author: s1866666, s1879523

Description: generates random texts using our model
'''

from cw1 import *

training_en = get_data("./assignment1-data/training.en")
training_es = get_data("./assignment1-data/training.es")
training_de = get_data("./assignment1-data/training.de")
test = get_data("./assignment1-data/test")
model_read = read_model("./assignment1-data/model-br.en")

seed = random.randrange(sys.maxsize) # seed to generate random text
n=3

train_sets = [("training.en", training_en), 
				  ("training.es", training_es),
				  ("training.de", training_de)]
alpha = 0.1
lambdas=[0.1,0.1,0.8]
out_file = "./text_out"

est_uni = laplace_estimator(1).set_h_param(0)
est_lap = laplace_estimator(n).set_h_param(alpha)


print("read:")
text = model_read.generate_from_LM(number=10000, seed=seed, remove_hash=True)
with open(out_file, "w") as file:
	file.write(text)
print(text[:500])
print("PP on generated text= ", model_read.compute_perplexity(get_data(out_file)))
#uni=est_uni(get_data(out_file)).probs()
#print("unigram probabilities:{}".format(uni))


for train in train_sets:
	print("\n{}, laplace: alpha={}".format(train[0],alpha))
	model_lap = est_lap(train[1])
	text = model_lap.generate_from_LM(number=10000, seed=seed, remove_hash=True)
	with open(out_file, "w") as file:
		file.write(text)
	print(text[:500])
	print("PP on training set= ", model_lap.compute_perplexity(train[1]))
	print("PP on generated text= ", model_lap.compute_perplexity(get_data(out_file)))
	#uni=est_uni(get_data(out_file)).probs()
	#print("unigram probabilities:{}".format(uni))

#est_int = interpolation_estimator(n).set_h_param([alpha]+lambdas)
#model_int = est_int(train_sets[0][0])
#print("\ninterpolation: alpha={}, lambdas={}".format(alpha,lambdas))
#text = model_int.generate_from_LM(number=3000, seed=seed, remove_hash=True)
#with open(out_file, "w") as file:
#	file.write(text)
#print(text[:300])
#print("PP on training set= ", model_int.compute_perplexity(train_set))
#print("PP on generated text= ", model_int.compute_perplexity(get_data(out_file)))


