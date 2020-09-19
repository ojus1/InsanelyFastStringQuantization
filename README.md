# InsanelyFastStringQuantization

This repository implements ["Extremely Fast Text Feature Extraction for Classification and Indexing"](https://www.hpl.hp.com/techreports/2008/HPL-2008-91R1.pdf) in Pure-Python for extremely fast string quantization.

There are NO dependencies! ..... If you don't plan on using the progress bar (tqdm).

A Pure-Javascript implementation is in the works for in-browser Deep learning (Tensorflow.js).

### NOTE: Tested only on Python >= 3.7, May not work on other versions of Python!

## About
 Given an input string, a hash of the string is returned that has certain properties:
- No model required to generate features from string of arbitrary length.
- Extremely low memory requirements for the lookup table
- Insanely. Fast. Over 7200000 Characters/sec in Pure-Python!
- The quantized feature vector represents the PRESENCE of words. 
- Rather than frequency in the case of TF-IDF or BOW.
- Since this hashing is very lossy, it's not recommended for applications where inference speed is not a priority.
    
## Getting Started
    from InsanelyFastStringQuantization import Hasher
    
    vectorizer = Hasher(16, random_table=False) # Generate feature vector of size 16, and use a static-hard-coded lookup table
    # random_table is recommended to be set to False for consistency between production environments, 
    # or properly control seed for consistency hashing

    # Quantize a single string
    print(vectorizer.vectorize("Hello World!")) # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

    # Quantize a list of strings
    print(vectorizer.vectorize(["Hello World!", "Buy Now!", "Add to Cart"])) 
    # [
    #    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
    #    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    #    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    # ]

## Contributors

#### Surya Kant Sahu

* Exploring the intersection of Recommender Systems and Reinforcement Learning. Built Data Pipelines for multiple realtime Machine Learning applications. 
* I play the Piano. A huge fan of Frédéric Chopin and Japanese Neo-Classical.
* Contact: 
    + [Github](https://github.com/ojus1)
    + [LinkedIn](https://www.linkedin.com/in/surya-kant-oju/)
