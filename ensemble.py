import pickle
import numpy
#import numpy as np

from sklearn.metrics import classification_report 
class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):        
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
         
    def fit(self,X,y):
         w = numpy.ones((X.shape[0],))/X.shape[0]
         '''Build a boosted classifier from the training set (X, y).

        Returns:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
         self.weakClassifiesArr = []
         self.alphaArr = []
         for i in range(self.n_weakers_limit):
             weak_classifier = self.weak_classifier(max_depth=2)		      		       
             weakClassify = weak_classifier.fit(X, y, sample_weight=w)
             self.weakClassifiesArr.append(weakClassify)
             y_pre = weakClassify.predict(X)
             minErr = 0  
             result = y_pre+y
             for j in range(result.shape[0]):                                 
                 if result[j]==0:
                     minErr+=w[j]
             if minErr>0.5:                  
                  break        
             alpha = float(numpy.log((1.0 - minErr)/minErr)/2.0)        
             self.alphaArr.append(alpha)  
             w = w*(numpy.e**(-1*alpha*y*y_pre))
             w = w/w.sum()
              
    def predict_scores(self, X):
        self.Result = numpy.zeros((X.shape[0],),dtype=numpy.float32)
        
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''		
        for i in range(len(self.alphaArr)):
            self.Result += self.alphaArr[i]*self.weakClassifiesArr[i].predict(X)
        
    def predict(self, X, threshold=0):	    
        result = numpy.zeros((X.shape[0],),dtype=numpy.float32)
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''	    
        for i in range(len(self.alphaArr)):               
            result += self.alphaArr[i]*(self.weakClassifiesArr[i].predict(X))           	
        self.Predict = numpy.sign(result)	
		
    def report(self, y):
        labels =[1,-1] 
        target_names = ['labels_1','labels_-1'] 
        reportString = classification_report(y, self.Predict, labels=labels,target_names= target_names,digits=3)
        file_object = open('report.txt', 'w')
        file_object.write(reportString)
        file_object.close()
        
    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
