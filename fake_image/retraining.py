import user_feature_generation
import tweet_feature_generation
import pickle
import pandas as pd

from distutils.sysconfig import get_python_lib
retrain_model_path = (get_python_lib())+'/fake_image/models/retrain_model.pkl'
retrain_model_path = '/'.join(retrain_model_path.split('\\'))

def main(tweet):
    tweet_val = tweet_feature_generation.main(tweet)
    user_val = user_feature_generation.main(tweet)
    if tweet_val == user_val:
        return tweet_val
    else:
        with open(retrain_model_path, 'rb') as f:
            retrain_model = pickle.load(f)
        #preds = []
        tweet_features = tweet_feature_generation.gen_features(tweet)
        flatten = lambda lst: reduce(lambda l, i: l + flatten(i) if isinstance(i, (list, tuple)) else l + [i], lst, [])
        features = flatten(tweet_features)
        # logging.debug(features)
        df = pd.DataFrame([features],
                          columns=['tweetTextLen', 'numItemWords', 'questionSymbol', 'exclamSymbol',
                                   'externLinkPresent',
                                   'numberNouns', 'happyEmo', 'sadEmo', 'containFirstPron', 'containSecPron',
                                   'containThirdPron',
                                   'numUpperCase', 'positiveWords', 'negativeWords', 'numMentions', 'numHashtags',
                                   'numUrls',
                                   'rtCount', 'slangWords', 'colonSymbol', 'pleasePresent', 'WotValue', 'numQuesSymbol',
                                   'numExclamSymbol', 'readabilityValue', 'Indegree', 'Harmonic',
                                   'AlexaPopularity', 'AlexaReach', 'AlexaCountry', 'AlexaDelta'])
        preds = retrain_model.predict(df)
    return preds[0]