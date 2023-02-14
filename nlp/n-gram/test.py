from cw1 import *

data = get_data("./assignment1-data/training.en")


c_p = n_gram_processor(3)
w_p = word_processor(3)

assert c_p.preprocess_line("adf\bN. $&*J\nKƒ∂oj\n")=="##adfn. jkoj#"

print(c_p.preprocess_line(""))

#print(w_p.get_counts(data))

print("All tests passed!")

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
print(text)
