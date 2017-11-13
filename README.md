# fake_image
 
 Detects if a tweet is fake or not.
 
 Installation:</br>
 Clone / Download the latest version from the master branch.</br>
 Extract / Unzip zipped files.</br>
CD into the downloaded directory.</br>
Run: pip install . 

Example Usage:
```python
import fake_image
import json
with open('json_tweets.json') as json_file:
    tweet = json.load(json_file) 
fake_image.predict(tweet)
```
