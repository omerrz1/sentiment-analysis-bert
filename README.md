## My website https://www.imomer.net
## sentiment-analysis
# API coming soon
### 1. Load model and tokenizer
First we need to load the pretrained model and associated tokenizer:

```python

from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer

model = TFAutoModelForSequenceClassification.from_pretrained("sentiment-analysis") 
tokenizer = AutoTokenizer.from_pretrained("sentiment-analysis_tokeniser")
```
### 2. Define input text
Next we need some text input that we want to run through the model to get sentiment predictions:
```python

text = "I really enjoyed that movie!"
```
### 3. Tokenize input
We tokenize the input text using the tokenizer. This converts the text into numeric tokens that the model can understand:
```python

inputs = tokenizer(text, return_tensors="tf")
```
### 4. Get predictions
We pass the tokenized input to the model to generate predictions:
```python

outputs = model(inputs)
```
This outputs the raw prediction logits.
### 5. Decode predictions
We decode the logits into actual classes using argmax. This returns the predicted class:
```python

predictions = tf.argmax(outputs.logits, axis=1)
```
### 6. Print output
Finally we print the predicted class, which will be 1 for positive sentiment:
```python
print(predictions)
```
