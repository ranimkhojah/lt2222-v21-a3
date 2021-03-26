# LT2222 V21 Assignment 3

Your name: *Ranim Khojah*

## Part 1
train.py has also been modified by adding comments to explain individual LOCs.

#### def a()
Overall, the function converts a file string content to a list of charecters.
In detail, the function takes a filepath in a form of string as an argument. It reads the file, then tokenizes the letters of the words in each sentence. It also adds two start tokens `<S>` and two end tokens `<E>` at the beginning and the end of each sentence respectively. Finally, it returns two lists, the first contains all the charecters in the file and the second is unique list of those charecters.
 
#### def b()
Overall, the function creates vowel instances that consist of i) the vowel itself and ii) the context/ features of the vowel.
In detail, the function expects the output of function a() as an input, it goes through an iteration to i) convert alphabetical data (i.e. vowels) to numeric data (i.e. corresponding index of the vowel in a predifined list of vowels) and store them in list gt. then ii) process the context (two letters before and two after the vowel) of each vowel by vectorizing each of the four letter in the context. By that, it creates instances that consist of the class i.e. numeric version of a vowel and the corresponding vectorized context as features.

#### def g()
function g() is a helper function used by b() to vectorize each letter in the context. It takes a single character as an argument then it creates a vector of size << length of set of unique characters in the file >>, all zeros, beside a '1' value at the position that is equal to the index where that vowel is present in the set of chars. For instance, if the list of all chars is \['a','b','c','d','e','f'] so the vectorized version of the vowel 'a' would be 100000 and 'e' would be 000010 ... etc.
 
 
#### command-line arguments
Optional arguments: 
 - (--k) is the number of features (hiddensize) and has a default value of 200 
 - (--r) is the number of epochs/ iterations that the training process will go through and has a default value of 100
 - (m) is the path to the training data file (raw data)
 - (h) is the path of the output model file that should be written when the model is created (usually ends with .pt)


## Part 2
this part can be tested via: `python eval.py trained_model.pt /home/xsayas@GU.GU.SE/scratch/lt2222-v21-resources/svtest.lower.txt /home/xsayas@GU.GU.SE/scratch/lt2222-v21-resources/svtrain.lower.txt`


## Part 3

```--k 200    --r 100    accuracy: 0.2241483358981322
--k 400    --r 100    accuracy: 0.14973899399535318
--k 600    --r 100    accuracy: 0.20538004284722852
--k 800    --r 100    accuracy: 0.1601943212335174
--k 1000   --r 100    accuracy: 0.12648984641380767

--k 200    --r 20     accuracy: 0.1601943212335174
--k 200    --r 50     accuracy: 0.15931927220059744
--k 200    --r 200    accuracy: 0.12704806734860144
--k 200    --r 300    accuracy: 0.09642135119640324
--k 200    --r 400    accuracy: 0.015992275429226637 
```

The model that had the highest accuracy is the one the used the default parameters i.e. --k 200 and --r 100.
based on the results, it seems like the accuracy decreased by increasing the number of epochs, that may be due to overfitting that occured when more epochs are iterated through even after convergence.

