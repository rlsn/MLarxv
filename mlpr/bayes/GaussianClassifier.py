import numpy as np

class GaussianClassifier(object):
    """docstring for GaussianClassifier"""
    def __init__(self, inputs, targets):
        assert inputs.shape[0] == targets.shape[0], "input size does not match target size!"
        self.inputs = inputs
        self.targets = targets
        self.labels = np.unique(targets)
        

class BayesClassifier(GaussianClassifier):
    """docstring for BayesClassifier"""
    def __init__(self, inputs, targets):
        super(BayesClassifier, self).__init__(inputs, targets)

    def train(self):
        self.inputs_by_target = {}
        for sample, target in zip(self.inputs, self.targets):
            if not target in self.inputs_by_target:
                self.inputs_by_target[target] = []
            self.inputs_by_target[target].append(sample)
        
        self.sigma, self.mean, self.sigma_inv, self.sigma_det, self.pi = {}, {}, {}, {}, {}
        for target in self.inputs_by_target:
            self.inputs_by_target[target] = np.array(self.inputs_by_target[target])
            self.sigma[target] = np.cov(self.inputs_by_target[target].T)
            self.mean[target] = np.mean(self.inputs_by_target[target], 0)
            self.sigma_inv[target] = np.linalg.inv(self.sigma[target])
            self.sigma_det[target] = np.linalg.det(self.sigma[target])
            self.pi[target] = self.inputs_by_target[target].shape[0]/self.inputs.shape[0]
        
    def likelihood(self, sample):
        likelihood = {}
        dim = len(sample)
        for target in self.inputs_by_target:
            mean = self.mean[target]
            sigma = self.sigma[target]
            sigma_inv = self.sigma_inv[target]
            sigma_det = self.sigma_det[target]
            likelihood[target]=np.exp(-0.5*(sample-mean).dot(sigma_inv).dot(sample-mean))/(2*np.pi**dim*sigma_det)**0.5
        return likelihood

    def predict(self, sample):
        likelihood = self.likelihood(sample)
        max_joint = 0
        predict = None
        for target in likelihood:
            joint = likelihood[target]*self.pi[target]
            if joint > max_joint:
                max_joint = joint
                predict = target
        return predict
    
    def joint(self, sample):
        likelihood = self.likelihood(sample)
        joint = {}
        for target in likelihood:
            joint[target]=likelihood[target]*self.pi[target]
        return joint

    def joint_all(self, samples):
        joints = {}
        for target in self.labels:
            joints[target] = np.zeros(samples.shape[0])
            for i,s in enumerate(samples):
                joints[target][i]=self.joint(s)[target]
        return joints
    
    def predict_all(self,samples):
        predicts = np.zeros(samples.shape[0])
        for i,s in enumerate(samples):
            predicts[i]=self.predict(s)
        return predicts

class NaiveBayesClassifier(GaussianClassifier):
    """docstring for NaiveBayesClassifier"""
    def __init__(self, inputs, targets):
        super(NaiveBayesClassifier, self).__init__(inputs, targets)
    
    def train(self):
        self.inputs_by_target = {}
        for sample, target in zip(self.inputs, self.targets):
            if not target in self.inputs_by_target:
                self.inputs_by_target[target] = []
            self.inputs_by_target[target].append(sample)
        
        self.sigma, self.mean, self.pi = {}, {}, {}
        for target in self.inputs_by_target:
            self.inputs_by_target[target] = np.array(self.inputs_by_target[target])
            self.sigma[target] = np.var(self.inputs_by_target[target], 0)
            self.mean[target] = np.mean(self.inputs_by_target[target], 0)
            self.pi[target] = self.inputs_by_target[target].shape[0]/self.inputs.shape[0]

    def likelihood(self, sample):
        likelihood = {}
        for target in self.inputs_by_target:
            likelihood[target] = 1
            for d in range(len(sample)):
                mean = self.mean[target][d]
                sigma = self.sigma[target][d]
                likelihood[target]*=np.exp(-0.5*(sample[d]-mean)**2/sigma)/(2*np.pi*sigma)**0.5
        return likelihood

    def joint(self, sample):
        likelihood = self.likelihood(sample)
        joint = {}
        for target in likelihood:
            joint[target]=likelihood[target]*self.pi[target]
        return joint

    def joint_all(self, samples):
        joints = {}
        for target in self.labels:
            joints[target] = np.zeros(samples.shape[0])
            for i,s in enumerate(samples):
                joints[target][i]=self.joint(s)[target]
        return joints
    
    
    def predict(self, sample):
        likelihood = self.likelihood(sample)
        max_joint = 0
        predict = None
        for target in likelihood:
            joint = likelihood[target]*self.pi[target]
            if joint > max_joint:
                max_joint = joint
                predict = target
        return predict
    
    def predict_all(self,samples):
        predicts = np.zeros(samples.shape[0])
        for i,s in enumerate(samples):
            predicts[i]=self.predict(s)
        return predicts