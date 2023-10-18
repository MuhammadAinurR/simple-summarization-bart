# Simple Text Summarization with BART
This repository contains a simple text summarization system built with the BART model. The model is pre-trained on the CNN/Daily Mail dataset and can generate a summary for a given piece of text.

## Features
* Uses the facebook/bart-large-cnn model from Hugging Face’s transformers library.
* The system takes a piece of text and returns a summarized version of it.
* The summary is generated directly from the input text.
## Usage
The main function in this repository is summarize(text). Here’s how to use it:

### Python


    from transformers import BartTokenizer, BartForConditionalGeneration

### Load the pre-trained BART model and tokenizer
    model_name = 'facebook/bart-large-cnn'
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

### Text to summarize
    text = """
    ... Your text here ...
    """
Replace "... Your text here ..." with the text you want to summarize. 

### main function
    def summarize(text):
        # Tokenize the input text
        inputs = tokenizer([text], max_length=1024, return_tensors='pt')

        # Generate a summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, length_penalty=2.0, max_length=100, min_length=30, early_stopping=True)

        # Decode the summary
        summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    
        return summary

    summary = summarize(text)
    print("Summary:", summary)
The summarize() function will return a summarized version of the input text.

## Requirements
* Python 3.6 or later.
* PyTorch 1.0.0 or later.
* Transformers library from Hugging Face.
  
## Installation
You can install the required packages with pip:

    pip install torch transformers
