from cw1 import *
import matplotlib.pyplot as plt
import sys

test = get_data("./assignment1-data/test")
train_en = get_data("./assignment1-data/training.en")
train_es = get_data("./assignment1-data/training.es")
train_de = get_data("./assignment1-data/training.de")

#length_counts = w_p.get_length_counts(train)
#length_data = []
#for length in length_counts:
#	length_data+=[length]*length_counts[length]
#plt.hist(length_data,len(length_counts))
#plt.show()





word_stats=[(4.736,6465),(5.231,4089),(4.794,7346),(5.760,4195),(3.796,9752),(4.475,5789),(4.472,5552),(4.120,7143),(3.790,10084),(3.855,9411),(5.027,5854),(4.626,6988),(4.708,6822),(4.813,6118),(4.418,7174),(4.264, 8064),(4.831,6403),(5.712,4399),(5.560,4307)
]

length = [p[0] for p in word_stats]
counts = [p[1] for p in word_stats]

fig, ax = plt.subplots()
ax.plot(length, counts, "bo")

ax.set(xlabel='average word length in training set', ylabel='number of real words generated',
       title='average word length - number of real words')
ax.grid()
fig.savefig("word_length.png")
plt.show()








def get_statics(filename, number = 100000, n=3, alpha=0.1):
	w_p = word_processor(1)
	train = get_data(filename)
	word_counts = w_p.get_counts(train)
	print("average word length in train set:",w_p.average_word_length(train))
	print("word type present in train set:",len(word_counts))
	for i in range(n):
		est = laplace_estimator(i+1).set_h_param(alpha)
		print("\nn = ", i+1)
		model = est(train)
		print("PP on train set= ", model.compute_perplexity(train))
		text=model.generate_from_LM(number)
		#print("random generation sample:\n",text[:500],'\n')
	
		real_words=[]
		real_word_counts=0
		gen_data=text.split('\n')
		gen_word_counts=w_p.get_counts(gen_data,remove_marks=True)
		for word in gen_word_counts:
			if word in word_counts:
				real_words+=[word]
				real_word_counts+=gen_word_counts[word]
		print("real word counts = {}".format(real_word_counts))
		#print("real word types:",real_words)


assert len(sys.argv)==2
get_statics(sys.argv[1])









