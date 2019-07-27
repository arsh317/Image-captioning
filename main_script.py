from keras.preprocessing import sequence
from keras.layers import LSTM, Embedding, Dropout, Dense
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
from PIL import Image
from keras import Input
from keras.layers.merge import add


token_dir = 'Flickr8k_text/Flickr8k.token.txt'

###### code to make caption dictionary whose keys are image file name
###### and values are image caption.


image_captions = open(token_dir).read().split('\n')
caption = {}    
for i in range(len(image_captions)-1):
    id_capt = image_captions[i].split("\t")
    id_capt[0] = id_capt[0][:len(id_capt[0])-2] # to rip off the #0,#1,#2,#3,#4 from the tokens file
    if id_capt[0] in caption:
        caption[id_capt[0]].append(id_capt[1])
    else:
        caption[id_capt[0]] = [id_capt[1]]
        

caption['1000268201_693b08cb0e.jpg'] 

########       

train_imgs_id = open("Flickr8k_text/Flickr_8k.trainImages.txt").read().split('\n')[:-1]


train_imgs_captions = open("Flickr8k_text/trainimgs.txt",'w')
for img_id in train_imgs_id:
    for captions in caption[img_id]:
        desc = "<start> "+captions+" <end>"
        train_imgs_captions.write(img_id+"\t"+desc+"\n")
        train_imgs_captions.flush()
train_imgs_captions.close()

test_imgs_id = open("Flickr8k_text/Flickr_8k.testImages.txt").read().split('\n')[:-1]

test_imgs_captions = open("Flickr8k_text/testimgs.txt",'w')
for img_id in test_imgs_id:
    for captions in caption[img_id]:
        desc = "<start> "+captions+" <end>"
        test_imgs_captions.write(img_id+"\t"+desc+"\n")
        test_imgs_captions.flush()
test_imgs_captions.close()


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x


model = InceptionV3(weights='imagenet')

new_input = model.input
new_output = model.layers[-2].output
model_new = Model(new_input, new_output)

def encode(image):
    image = preprocess(image)
    temp_enc = model_new.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc


images = 'Flickr8k_Dataset/'

train_imgs_id = open("Flickr8k_text/Flickr_8k.trainImages.txt").read().split('\n')[:-1]
test_imgs_id = open("Flickr8k_text/Flickr_8k.testImages.txt").read().split('\n')[:-1]

encoding_train = {}
for img in tqdm(train_imgs_id): #tqdm instantly make your loops show a smart progress meter
    path = images+str(img)
    encoding_train[img] = encode(path)
    
with open("encoded_train_images_inceptionV3.p", "wb") as encoded_pickle: 
    pickle.dump(encoding_train, encoded_pickle) #python object can be pickled so that it can be saved on disk.

encoding_train = pickle.load(open('encoded_train_images_inceptionV3.p', 'rb'))

encoding_train['3556792157_d09d42bef7.jpg'].shape   


encoding_test = {}
for img in tqdm(test_imgs_id):
    path = images+str(img)
    encoding_test[img] = encode(path)

with open("encoded_test_images_inceptionV3.p", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)

encoding_test = pickle.load(open('encoded_test_images_inceptionV3.p', 'rb')) 


dataframe = pd.read_csv('Flickr8k_text/trainimgs.txt', delimiter='\t')
captionz = []
img_id = []
dataframe = dataframe.sample(frac=1)
iter = dataframe.iterrows()

for i in range(len(dataframe)):
    nextiter = next(iter)
    captionz.append(nextiter[1][1])
    img_id.append(nextiter[1][0])

no_samples=0
tokens = []
tokens = [i.split() for i in captionz]
for caption in captionz:
    no_samples+=len(caption.split())-1


vocab= [] 
for token in tokens:
    vocab.extend(token)
vocab = list(set(vocab))
with open("vocab.p", "wb") as pickle_d:
   pickle.dump(vocab, pickle_d)



vocab= pickle.load(open('vocab.p', 'rb'))
print (len(vocab))

vocab_size = len(vocab)

word_idx = {val:index for index, val in enumerate(vocab)}
idx_word = {index:val for index, val in enumerate(vocab)}

word_idx['end']


caption_length = [len(caption.split()) for caption in captionz]
max_length = max(caption_length)
max_length # maximum lenght of a caption.


def data_process(batch_size):
    partial_captions = []
    next_words = []
    images = []
    total_count = 0
    while 1:
    
        for image_counter, caption in enumerate(captionz):
            current_image = encoding_train[img_id[image_counter]]
    
            for i in range(len(caption.split())-1):
                total_count+=1
                partial = [word_idx[txt] for txt in caption.split()[:i+1]]
                partial_captions.append(partial)
                next = np.zeros(vocab_size)
                next[word_idx[caption.split()[i+1]]] = 1
                next_words.append(next)
                images.append(current_image)

                if total_count>=batch_size:
                    next_words = np.asarray(next_words)
                    images = np.asarray(images)
                    partial_captions = sequence.pad_sequences(partial_captions, maxlen=max_length, padding='post')
                    total_count = 0
                
                    yield [[images, partial_captions], next_words]
                    partial_captions = []
                    next_words = []
                    images = []
                    
               
embedding_dim = 300
                

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.summary()

no_epochs = 10
for i in range(no_epochs): 
    epoch=1
    batch_size = 128
    model.fit_generator(data_process(batch_size=batch_size), 
                    steps_per_epoch=no_samples/batch_size,
                    epochs=epoch, verbose=1, callbacks=None)


model.save('my_model.h5')