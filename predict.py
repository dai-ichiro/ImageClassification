from autogluon.vision import ImagePredictor

predictor = ImagePredictor.load('predictor.ag')

proba = predictor.predict_proba('test1.jpg')

print(proba)

