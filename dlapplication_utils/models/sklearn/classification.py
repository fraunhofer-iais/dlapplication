    def getSKLearnLogisticRegression(self, regParam, dim=1):
        from DLplatform.learning.batch.sklearnClassifiers import LogisticRegression
        
        learner = LogisticRegression(regParam = regParam, dim = dim)
        return learner