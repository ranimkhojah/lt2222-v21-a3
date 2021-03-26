import torch
from train import a as tokenize
import argparse
import numpy as np


vowels = sorted(['y', 'é', 'ö', 'a', 'i', 'å', 'u', 'ä', 'e', 'o'])

def vectorize(x, p, len_vocab): 
    z = np.zeros(len_vocab) #to prevent "mat1 and mat2 shapes cannot be multiplied (66282x400 and 452x200)" error when predicting
    z[p.index(x)] = 1
    return z

def create_instances(u, p, len_vocab): 
    gt = [] 
    gr = [] 
    for v in range(len(u) - 4):
        if u[v+2] not in vowels: 
            continue 
            
        h2 = vowels.index(u[v+2])
        gt.append(h2) 
        r = np.concatenate([vectorize(x, p, len_vocab) for x in [u[v], u[v+1], u[v+3], u[v+4]]])
        gr.append(r) 

    return torch.from_numpy(np.array(gr)), torch.from_numpy(np.array(gt)) #convert gr and gt to Tensor
        
def write_preds(predicted_i, real_i, filename):
    TP = 0
    content = []
    header = "Prediction \t Actual"
    content.append(header)
    for i in range(len(predicted_i)):
        if predicted_i[i] == real_i[i]:
            TP += 1
        predicted_val = str(vowels[predicted_i[i]])
        real_val = str(vowels[real_i[i]])
        row = predicted_val+" \t \t \t "+real_val
        content.append(row)
    with open(filename, 'w') as f:
        for row in content:
            f.write('%s\n' % row)
    return TP
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str) #model path
    parser.add_argument("test_data", type=str) #test data file: /home/xsayas@GU.GU.SE/scratch/lt2222-v21-resources/svtest.lower.txt
    parser.add_argument("train_data", type=str) #train data file: /home/xsayas@GU.GU.SE/scratch/lt2222-v21-resources/svtrain.lower.txt
    
    args = parser.parse_args()
    
    # Load a model produced by train.py. (Take a look at model.py.)
    model = torch.load(args.model_path)
    model.eval() #set dropout and batch normalization layers
    
    len_vocab = len(tokenize(args.train_data)[1])
    
    # Load the test data
    tok_test_data = tokenize(args.test_data)
    
    # Create evaluation instances compatible with the training instances.
    feats, classes = create_instances(tok_test_data[0], tok_test_data[1], len_vocab)
    
    # Use the model to predict instances.
    prediction = model(feats.float())
    
    # Write the text with the predicted (as opposed to the real) vowels back into an output file.
    predicted_i = prediction.argmax(dim=1).numpy()
    real_i = classes.numpy()
    TP = write_preds(predicted_i, real_i, "preds_output.csv")
    
    # Print the accuracy of the model to the terminal.
    acc = TP / len(real_i)
    print("Accuracy: ", acc)
    