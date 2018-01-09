from chainer import links as l
from chainer import functions as f

class CustomClassifier(l.Classifier):

    def get_predictions(self, x):
        raw_predictions = self.predictor(x)
        return f.sigmoid(raw_predictions)

