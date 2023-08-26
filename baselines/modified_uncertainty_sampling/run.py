import pickle
import numpy as np
import OED
from sklearn.linear_model import LogisticRegression

task = "hawk"
strategy = "diverse"


with open("data/embedding_image_ids.pkl","rb") as f:
    image_ids = pickle.load(f)
print(len(image_ids))

imid_to_idx = {}
for i in range(len(image_ids)):
    imid_to_idx[image_ids[i]] = i

with open("data/embeddings.npy","rb") as f:
    embeddings = np.load(f)
print(embeddings.shape)

embeddings = embeddings - np.mean(embeddings, axis=0)

#cov = embeddings.T @ embeddings / embeddings.shape[0]
#print(np.linalg.eigvalsh(cov))

with open("data/"+task+".txt","r") as f:
    positives = []
    for line in f.readlines():
        positives.append(line.strip())
print(len(positives))

embedded_positives = list(set(positives).intersection(set(image_ids)))
print(len(embedded_positives))



n, d = embeddings.shape

labels = np.zeros((n,))
positive_idx = []
for pos in embedded_positives:
    positive_idx.append(imid_to_idx[pos])
labels[positive_idx] = 1

model = LogisticRegression(C = 10**6, max_iter = 1000)

print("starting fit")
if False:
    model.fit(embeddings,labels)
    with open("data/model_"+task+".pkl","wb") as f:
        pickle.dump(model,f)
else:
    with open("data/model_"+task+".pkl","rb") as f:
        model = pickle.load(f)
print("finished fit")
fit_labels = model.predict(embeddings)

TP = np.sum( fit_labels[positive_idx] == 1)
FN = np.sum( fit_labels[positive_idx] != 1)
FP = np.sum( np.logical_and(fit_labels == 1, labels == 0) )

print("True positives: {}\nFalse negatives: {}\nFalse positives: {}".format(TP,FN,FP))


probs = model.predict_proba(embeddings)[:,1]

pp = n//100

if task == "sushi":
    topp = np.argsort(probs)[-5*pp:-2*pp]
if task == "cupcake":
    topp = np.argsort(probs)[-5*pp:-2*pp]
if task == "hawk":
    topp = np.argsort(probs)[-5*pp:-2*pp]

#if task == "sushi":
#    close_idx = np.logical_and(probs >= 10**-6,probs < 10**-4).nonzero()[0]
#elif task == "cupcake":
#    close_idx = np.logical_and(probs >= 10**-6,probs < 10**-5).nonzero()[0]
#elif task == "hawk":
#    close_idx = np.logical_and(probs >= 10**-5,probs < 10**-4).nonzero()[0]
if task == "cupcake":
    true_idx = np.logical_and(probs >= 0.5, labels==1).nonzero()[0]
if task == "sushi":
    true_idx = (labels==1).nonzero()[0]
if task == "hawk":
    true_idx = np.logical_and(probs >= 0.5, labels==1).nonzero()[0]

close_idx = list(set(topp).difference(set(true_idx)))

print(len(close_idx))
print(len(true_idx))

if strategy == "even":
    with open("data/train_sets/"+task+"_even.csv","w") as f:
        f.write("ImageID,Confidence\n")
    
        true_labels = min(500, len(true_idx))

        for idx in np.random.choice(true_idx,size=(true_labels,),replace=False):
            f.write("{},1\n".format(image_ids[idx]))
        
        for idx in np.random.choice(close_idx,size=(1000-true_labels,),replace=False):
            f.write("{},0\n".format(image_ids[idx]))
        
if strategy == "diverse":
    with open("data/train_sets/"+task+"_27.csv","w") as f:
        f.write("ImageID,Confidence\n")
    
        if task == "cupcake":
            true_labels = 500
        if task == "sushi":
            true_labels = 500
        if task == "hawk":
            true_labels = 333


        true_embeddings = embeddings[true_idx,:]
        design_points = OED.random(true_embeddings,true_labels)
        selected_idx = [true_idx[point] for point in design_points]
        
        for idx in selected_idx:
            f.write("{},1\n".format(image_ids[idx]))
       

        close_embeddings = embeddings[close_idx,:]
        design_points = OED.random(close_embeddings,1000-true_labels)
        selected_idx = [close_idx[point] for point in design_points]
        
        for idx in selected_idx:
            f.write("{},0\n".format(image_ids[idx]))
        



print("done!")
