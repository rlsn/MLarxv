def optimise_alpha(data):
	folds = split_k_fold(data)
	for i in range(len(folds)):
		val, train = folds.pop(i), folds