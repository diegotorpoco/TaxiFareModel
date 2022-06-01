# imports
from TaxiFareModel.encoders import TaxiFarePipeline
from TaxiFareModel.data import get_data,clean_data
from TaxiFareModel.utils import compute_rmse


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = self.set_pipeline()
        self.X = X
        self.y = y

    def set_pipeline(self):
        pipeline = TaxiFarePipeline().create_pipeline()
        return pipeline

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X,self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred,y_test)
        return rmse



# if __name__ == "__main__":
#     # clean data
#     # set X and y
#     # hold out
#     # train
#     # evaluate
#     print('')
