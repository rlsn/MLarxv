'''
Filename: cw1.py
Author: s1866666, s1879523

Description: library for anlp assignment 1
'''
import sys
import string
import re
import math
import random
import numpy as np

def get_data(filename):
	"""
		get data from a given file where each line is a single sample
		return: a list contianing seperate samples
	"""
	with open(filename) as file:
		return [line for line in file]

def read_model(filename):
	"""
		read a language model from a given file
	"""
	data = get_data(filename)
	probabilities = dict()
	n = len(data[0].split('\t')[0])
	for line in data:
		entry = line.split('\t')
		if entry[0][:-1] not in probabilities:
			probabilities[entry[0][:-1]]=dict()
		probabilities[entry[0][:-1]][entry[0][-1]]=float(entry[1])
	return language_model(n,probabilities)

class n_gram_processor:
	"""docstring for n_gram_processor"""
	def __init__(self, n):
		"""
			param: n 	this is an n-gram language model, n should be an integer greater than 0
		"""
		assert n>0, "n must be greater than 0"
		self.n = n
	
	SIGMA = list(" #.0"+string.ascii_lowercase) # all the legal characters in our model
	

	# the legal pattern that each n-gram must match
	# for trigram, among all combinations including hash
	# only "..." "###"(empty line), "##.", "#..", "#.#"(one character line) and "..#" are legal
	# unlegal: ".#.", ".##"
	LEGAL_N_GRAM = "^#*[^#]*#?$" 

	def preprocess_line(self, line):
		"""
			Preprocess the line so only the characters we are interested in
			are present. extra hashes are added to denote the beginning and the end
			of the line for training purposes.

			param: the line to be processed
			returns: the processed line
		"""
		processed = ''
		for char in line.lower():
			if char in " ."+string.ascii_lowercase:
				processed+=char
			elif char.isdigit():
				processed+='0'
		return '#'*(self.n-1)+processed+'#'

	def generate_n_grams(self):
		""" 
		generates all the legal combinations of characters in our char space
		
		"""
		gram_list = list(self.SIGMA)
		if self.n==1:
			return gram_list
		legal_pattern = re.compile(self.LEGAL_N_GRAM)
		
		for i in range(self.n-1):
			gram_list = [gram+char for gram in gram_list for char in self.SIGMA]
		# filter the list so only the legal ones are present
		gram_list = [gram for gram in gram_list if re.match(legal_pattern, gram)]
		return gram_list
	
	def get_counts(self, data):
		"""
			preprocesses and collects n-gram counts in the data
		"""

		counts = dict([(s, 0) for s in self.generate_n_grams()])
		for line in data:
			line = self.preprocess_line(line)
			for i in range(len(line)-self.n+1):
				counts[line[i:i+self.n]]+=1
		return counts

class char_processor(n_gram_processor):
	"""
	docstring for char_processor
	"""
	def __init__(self, n):
		super().__init__(n)
		


class word_processor():
	"""
	docstring for word_processor
		to be developed
	"""

	WORD_ALPHABET = list("-"+string.ascii_lowercase)
	FUNCTION_MARKS = list(".?")

	def __init__(self, n):
		self.n = n

	def preprocess_line(self, line):
		processed = ''
		for char in line.lower():
			if char in [' ']+self.WORD_ALPHABET:
				processed+=char
			elif char in self.FUNCTION_MARKS:
				processed+=' '+char+' '
			#elif char.isdigit():
			#	processed+='0'
		return '# '*(self.n-1)+processed+' #'

	def get_counts(self, data, remove_marks=False):
		"""
		just count unigram now
		"""
		word_counts = dict()
		for line in data:
			words = self.preprocess_line(line).split(' ')
			for word in words:
				if word and ((not remove_marks) or (word not in self.FUNCTION_MARKS)):
					if word not in word_counts:
						word_counts[word]=1
					else:
						word_counts[word]+=1
		return word_counts

	def get_length_counts(self, data, remove_marks=True):
		word_counts = self.get_counts(data)
		length_counts = dict()
		for word in word_counts:
			if (not remove_marks) or (word not in self.FUNCTION_MARKS):
				if len(word) not in length_counts:
					length_counts[len(word)] = word_counts[word]
				else:
					length_counts[len(word)] += word_counts[word]
		return length_counts

	def average_word_length(self, data):
		length_counts = self.get_length_counts(data)
		total = 0
		for length in length_counts:
			total+=length*length_counts[length]
		return total/sum(length_counts.values())

	def get_type_by_length(self,data,remove_marks=True):
		word_counts = self.get_counts(data)
		type_by_length = dict()
		for word in word_counts:
			if (not remove_marks) or (word not in self.FUNCTION_MARKS):
				if len(word) not in type_by_length:
					type_by_length[len(word)] = [word]
				else:
					type_by_length[len(word)] += [word]
		return type_by_length


class language_model:
	"""docstring for language_model
		an object that describes a language model
	"""
	def __init__(self, n, probabilities):
		"""
			param: n 	this is an n-gram language model, n should be an integer greater than 0
		"""
		assert n>0, "n must be greater than 0"
		self.n = n
		self.probabilities = probabilities
		self.proc = n_gram_processor(self.n)

	def probs(self):
		return self.probabilities

	def __str__(self):
		return str(self.probabilities)

	def write(self, filename):
		"""
			write the language model to a file
		"""
		with open(filename, "w") as file:
			if self.n==1: # unigram
				file.write('\n'.join([char+'\t'+str(self.probabilities[char]) for char in self.probabilities]))
				return
			prob = []
			for hist in self.probabilities:
				for char in self.probabilities[hist]:
					prob.append(hist+char+'\t'+str(self.probabilities[hist][char]))
			file.write('\n'.join(prob))

	def write_excerpt(self, filename):
		"""
			write the model as a readable excerpt
			Note that for the sake of readability spaces are replaced with '-'
		"""
		with open(filename, "w") as file:
			if self.n==1: # unigram
				file.write('\n'.join(["P({0})={1:.3e}".format(char.repace(' ', '-'),self.probabilities[char])\
				 for char in self.probabilities]))
				return
			prob = []
			for hist in self.probabilities:
				for char in self.probabilities[hist]:
					_char = char.replace(' ', '-')
					_hist = hist.replace(' ', '-')
					prob.append("P({0}|{1}) = {2:.3e}".format(_char,_hist,self.probabilities[hist][char]))
			file.write('\n'.join(prob))

	def generate_from_LM(self, number = 300, seed=None, remove_hash=True):
		"""
			generate [number] characters of random sequences using the model, if an end-of-sequence
			if generated, start another sequence.
		"""
		random.seed(seed)
		if self.n==1: # unigram
			output = "".join(random.choices(list(self.probabilities.keys())\
				, weights=list(self.probabilities.values()),k=number)).replace('#', '#\n')
		else:
			begin_of_sentence = '#'*(self.n-1)
			output = begin_of_sentence # to begin a sequence 
			for i in range(number):
				output += random.choices(list(self.probabilities[output[-self.n+1:]].keys())\
					, weights=list(self.probabilities[output[-self.n+1:]].values()))[0]
				if output[-1]=='#': # sentence ends
					output+="\n"+begin_of_sentence
		return output.replace('#', '') if remove_hash else output
	
	def compute_perplexity(self, data, penalty=1e-8):
		"""
			compute the perplexity of given data under this model
		"""
		counts = self.proc.get_counts(data)
		log_likelihood = 0
		if self.n==1: # simple unigram
			for char in counts:
				if not self.probabilities[char]==0:
					log_likelihood+=counts[char]*math.log(self.probabilities[char],2)
				else:
					log_likelihood+=counts[char]*math.log(penalty,2)
		else:
			for n_gram in counts:
				if (n_gram[:-1] in self.probabilities) and (n_gram[-1] in self.probabilities[n_gram[:-1]]):
					if not self.probabilities[n_gram[:-1]][n_gram[-1]] == 0: 
						log_likelihood+=counts[n_gram]*math.log(self.probabilities[n_gram[:-1]][n_gram[-1]],2)
					else: 
						#can't evaluate perplexity when there's 0 probability
						# assign a very small number to prevent division by zero
						log_likelihood+=counts[n_gram]*math.log(penalty,2) 
				else: # model doesn't cover the sequence, a penalty is given
					log_likelihood+=counts[n_gram]*math.log(penalty,2)
		return 2**(-log_likelihood/sum(counts.values()))

class estimator:
	"""docstring for estimator, this is a base class"""

	def __init__(self, n, test):
		"""
			param: n 	this is an n-gram language model, n should be an integer greater than 0
			param: test	(boolean) whether to check if probabilities given the same condition add up to 1
			
		"""
		assert n>0, "n must be greater than 0"
		self.n = n
		self.test = test
		self.proc = n_gram_processor(self.n)
		self.h_param = None # hyper-parameters of this estimator
	
	def set_h_param(self, h_param):
		"""
			set hyper-parameters of this estimator
		"""
		self.h_param = h_param
		return self

	def check_sum(self, probability, accept_zero=False):
		"""
			check if Sum_w_3[P(w_3|w_1 w_2)]==1
		"""
		if self.n==1:
			return math.isclose(sum(probability.values()),1)

		for cond in probability:
			prob = 0
			for cond_prob in probability[cond].values():
				prob+=cond_prob
			if not (math.isclose(prob,1,rel_tol=1e-03) or (prob==0 and accept_zero)):
				print(prob)
				return False
		return True

class laplace_estimator(estimator):
	"""
		docstring for laplace_estimator
		This is a probability estimator using laplace smoothing

			h_param:	the value of alpha in the range of [0,1]

			when alpha=0, the estimation is equvalent to a 
			maximum likihood estimation, whereas
			when alpha=1, it implements add-one smoothing. 
			Hence we generalize these methods into one class

	"""
	def __init__(self, n, test=False):
		super().__init__(n, test)
		self.h_param = 0 # set default alpha

	def validate_h_param(self):
		assert self.h_param>=0 and self.h_param<=1, "alpha must be in range [0,1]"

	def estimate(self, data):
		"""
			param:
				data 	the data we use to estimate 
							
			return: a dictionary contains n-gram estimation of the data given alpha
		"""
		self.validate_h_param()
		n_gram_counts = self.proc.get_counts(data)
		probabilities = dict()

		if self.n==1: # simple unigram
			total = sum(n_gram_counts.values())
			probabilities = dict([(char, count/total) for char, count in n_gram_counts.items()])
		else:
			# get history counts and vocabulary size of c_i for n_grams
			# vocabulary size varies in trigrams due to the introduction of hashes
			hist_gram_counts, v_sizes = dict(), dict()
			for n_gram in n_gram_counts:
				if n_gram[:-1] not in hist_gram_counts:
					hist_gram_counts[n_gram[:-1]]=n_gram_counts[n_gram]
					v_sizes[n_gram[:-1]]=1
				else:
					hist_gram_counts[n_gram[:-1]]+=n_gram_counts[n_gram]
					v_sizes[n_gram[:-1]]+=1

			# estimate probabilities
			for n_gram in n_gram_counts:
				prob=0
				if n_gram_counts[n_gram]!=0 or self.h_param:
					prob = (n_gram_counts[n_gram]+self.h_param)/\
					(hist_gram_counts[n_gram[:-1]]+self.h_param*v_sizes[n_gram[:-1]])
				if n_gram[:-1] not in probabilities:
					probabilities[n_gram[:-1]]=dict([(n_gram[-1], prob)])
				else:
					probabilities[n_gram[:-1]][n_gram[-1]]=prob

		if self.test:
			# test if probabilities given the same history add up to 1
			# accept zero probability only if alpha=0. ie. MLE is used
			assert self.check_sum(probabilities, accept_zero=(self.h_param==0))
		return language_model(self.n,probabilities)
	__call__ = estimate

class interpolation_estimator(estimator):
	"""
		docstring for interpolation_estimator
		This is a probability estimator using interpolation
		we use add-one smoothing for 1 through n-gram probability estimation
		to prevent 0 probability which would cause Sum_w_3[P(w_3|w_1 w_2)]!=1
		
			h_param: 	[alpha lambdas]	a list of parameters of length n+1
						the first item is alpha for laplace smoothing, the rest
						are lambdas whose values should add up to 1
					
	"""

	def __init__(self, n, test=False):
		super().__init__(n, test)
		self.h_param = [ 0.1 ]+[1/n for i in range(n)] # set default alpha and lambdas


	

	def validate_h_param(self):
		assert len(self.h_param)==self.n+1, "there should be {} parameters, {} were given".format(self.n+1, len(self.h_param))
		assert math.isclose(sum(self.h_param[1:]),1), "lambdas should add up to 1"
		assert self.h_param[0]>0 and self.h_param[0]<=1, "alpha should be in range (0,1]"
	def estimate(self, data):
		"""
			param:
				data 	the data we use to estimate 
						
			return: a dictionary contains n-gram estimation of the data given alpha
		"""
		self.validate_h_param()
		alpha, lambdas = self.h_param[0], self.h_param[1:]
		grams_probabilities = [] # add-one models of from 1-gram to n-gram
		probabilities = dict() # our final result using interpolation

		if self.n==1: # simple unigram
			probabilities = laplace_estimator(1).set_h_param(alpha)(data).probs()
		else:
			# compute add-one of 1-n gram
			for i in range(self.n):
				grams_probabilities.append(laplace_estimator(i+1).set_h_param(alpha)(data).probs())
			
			# compute final interpolated probabilities
			for n_gram_hist in grams_probabilities[-1]:
				if n_gram_hist not in probabilities:
					probabilities[n_gram_hist]=dict()
				for char in grams_probabilities[-1][n_gram_hist]:
					# get all corresponding probabilities 1-n gram
					probs = [grams_probabilities[0][char]]
					for i in range(1,self.n):
						probs.append(grams_probabilities[i][n_gram_hist[-i:]][char])

					probabilities[n_gram_hist][char] = np.dot(lambdas, probs)

		if self.test:
			# test if probabilities given the same history add up to 1
			assert self.check_sum(probabilities)
		return language_model(self.n,probabilities)

	__call__ = estimate

class logger:
	"""docstring for logger"""
	def __init__(self, log_format, print_log=True):
		self.log = []
		self.log_format = log_format
		self.print_log = print_log # whether to print the log upon update
	def __call__(self, log):
		print("{}: ".format(len(self.log)+1)+self.log_format%tuple(log))
		self.log+=[log]
	def lg(self):
		return self.log
	def clr(self):
		self.log=[]

class optimisor:
	"""
		docstring for optimisor
		using Monte Carlo cross-validation to optimise the hyper-parameters of an estimation method
	"""
	def __init__(self, train_set_ratio, t):
		"""
			params: 
				train_set_ratio 	the percent of data we use to train, the rest we use to validate
				t 	the number of times we compute pp for each set of parameter with different train & val set 
		"""	
		assert train_set_ratio<1 and train_set_ratio>0, "train_set_ratio must in range (0,1)"
		self.train_set_ratio = train_set_ratio
		self.t = t
		self.logger = logger("param: %s, avg val pp: %.4e")

	def shuffle_and_split_data(self, data):
		random.shuffle(data)
		return data[:round(self.train_set_ratio*len(data))], data[round(self.train_set_ratio*len(data)):]
	
	def log(self):
		return self.logger.lg()

	def clear_log(self):
		self.logger.clr()

	def find_best_h_param(self, data, estimator, h_params):
		"""
			returns the hyper-parameters which gives the 
			least average perplexity on validation set
			and minimal validation perplexity
			param:
				data 		the data on which we train and validate model
				estimator 	the estimator whose hyper-parameter we are optimising
				h_params 	list of hyper-parameters to try

		"""
				
		min_val_pp, best_h_param =sys.maxsize, None
	
		for h_param in h_params:
			estimator.set_h_param(h_param)
			pp_sum = 0
			for i in range(self.t):
				train_set, val_set = self.shuffle_and_split_data(data)
				model = estimator(train_set)
				pp_sum+=model.compute_perplexity(val_set)
			pp_avg = pp_sum/self.t

			if pp_avg < min_val_pp:
				min_val_pp = pp_avg
				best_h_param = h_param

			self.logger([h_param, pp_avg])
				
		return best_h_param, min_val_pp

	