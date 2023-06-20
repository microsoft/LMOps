
# This script prepares the training images and ratings for the training.
# It assumes that all images are stored as files that PIL can read.
# It also assumes that the paths to the images files and the average ratings are in a .parquet files that can be read into a dataframe ( df ).

from datasets import load_dataset
import pandas as pd
import statistics
from torch.utils.data import Dataset, DataLoader
import clip
import torch
from PIL import Image, ImageFile
import numpy as np
import time

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)



device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

 
f = "trainingdata.parquet"
df = pd.read_parquet(f)  #assumes that the df has the columns IMAGEPATH  & AVERAGE_RATING


x = []
y = []
c= 0

for idx, row in df.iterrows():
    start = time.time()

    average_rating = float(row.AVERAGE_RATING)
    print(average_rating)
    if average_rating <1:
       continue

    img= row.IMAGEPATH  #assumes that the df has the column IMAGEPATH
    print(img)

    try:
       image = preprocess(Image.open(img)).unsqueeze(0).to(device)
    except:
   	   continue

    with torch.no_grad():
       image_features = model.encode_image(image)

    im_emb_arr = image_features.cpu().detach().numpy() 
    x.append(normalized ( im_emb_arr) )      # all CLIP embeddings are getting normalized. This also has to be done when inputting an embedding later for inference
    y_ = np.zeros((1, 1))
    y_[0][0] = average_rating
    #y_[0][1] = stdev      # I initially considered also predicting the standard deviation, but then didn't do it

    y.append(y_)


    print(c)
    c+=1




x = np.vstack(x)
y = np.vstack(y)
print(x.shape)
print(y.shape)
np.save('x_OpenAI_CLIP_L14_embeddings.npy', x)
np.save('y_ratings.npy', y)
