import tweet_feature_generation
import user_feature_generation
import retraining

def predict(tweet):
    retrained_classification = retraining.main(tweet)
    return retrained_classification