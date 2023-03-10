'''
Filename: compare_model.py
Author: s1866666, s1879523

Description: compares the differences between model-be.en and add-alpha model
'''


from cw1 import *

seed = random.randrange(sys.maxsize) # seed to generate random text
n = 3 # we are studying trigram model

training_en = get_data("./assignment1-data/training.en")
training_es = get_data("./assignment1-data/training.es")
training_de = get_data("./assignment1-data/training.de")
test = get_data("./assignment1-data/test")

alpha=0.1
est = laplace_estimator(n).set_h_param(alpha)

model_1 = read_model("./assignment1-data/model-br.en")

model_2 = est(training_en)
model_2.write("./model")


#print("Generating random text using seed: "+ str(seed)+'\n')
#print("Text generated by model-br.en (model_1):\n\n"+model_1.generate_from_LM(seed=seed),'\n')
#print("Text generated by model trained on training.en (alpha={}) (model_2):\n\n{}\n".format(alpha,model_2.generate_from_LM(seed=seed)))



print("Computing perplexity of file: "+"./assignment1-data/test")
print("model-br PP: ", model_1.compute_perplexity(test))
print("model-aa PP: ", model_2.compute_perplexity(test))
print("")
print("Computing perplexity of file: "+"./assignment1-data/training.en")
print("model-br PP: ", model_1.compute_perplexity(training_en))
print("model-aa PP: ", model_2.compute_perplexity(training_en))
print("")

