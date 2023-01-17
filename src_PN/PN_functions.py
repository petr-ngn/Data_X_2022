import os
import pandas as pd
import numpy as np
from time import time
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.data import AUTOTUNE
#from tensorflow.keras.models import Model
#from tensorflow.keras.backend import epsilon
#from tensorflow.keras.optimizers import Adam
#from tensorflow.math import square, maximum, reduce_mean, sqrt, reduce_sum
from sklearn.metrics import roc_curve, f1_score, accuracy_score, recall_score, precision_score
import seaborn as sns

#Function for reading the images' names, labels and bounding boxes
def reading_crop_inputs(identity_file = './data/Anno/identity_CelebA.txt', bbox_file = './data/Anno/list_bbox_celeba.txt'):
    #Loading the image names (image) and labels (image_id)
    identity = pd.read_csv(identity_file, sep = " ", header = None,
                        names = ['image', 'image_id'])
    #Loading the bounding boxes of images
    bbox = pd.read_csv(bbox_file, delim_whitespace = True)

    return identity, bbox



#Function for cropping all the images
def face_crop_export(id_df, bbox, crop_all = False):

    #Function for cropping, resizing and exporting a single image in .jpg format.
    def face_crop(image_name, bbox):

        #Loading Image
        image_path = './data/Img/img_celeba/' + image_name
        img = cv2.imread(image_path)

        #Setting bounding box coordinates
        startX = bbox[bbox['image_id'] == image_name]['x_1'].values[0]
        startY = bbox[bbox['image_id'] == image_name]['y_1'].values[0]
        endX = startX + bbox[bbox['image_id'] == image_name]['width'].values[0]
        endY = startY + bbox[bbox['image_id'] == image_name]['height'].values[0]
    
        #Cropping
        crop_img = img[startY:endY, startX:endX]
        output_img = crop_img
            
        #Resizing
        output_img = cv2.resize(crop_img, (224, 224))

        #Exporting the cropped image
        cv2.imwrite(f'./cropped_images/{image_name}', output_img)


    #Start time initialization (for execution time measurement)
    start_time = time()

    #List of images to crop (whether to crop all the images in the identity file or only those images which haven't been cropped yet)
    imgs_to_crop = id_df['image'] if crop_all == True else list(set(id_df['image'].tolist()).difference(os.listdir("./cropped_images/")))

    #Initialization for counting number of images which cannot be cropped.
    k = 0

    #Crop each image
    for ind, img in enumerate(imgs_to_crop):
        try:
            #Cropping image
            face_crop(img, bbox)

            #Print statement
            no_imgs = f'{ind + 1 - k}/{len(imgs_to_crop)}' #How many images have been cropped so far
            speed = (ind + 1- k)/(time() - start_time) * 60 #How many images have been cropped per minute on average
            eta = (len(imgs_to_crop) - (ind + 1 - k)) / ((ind + 1 - k)/(time() - start_time)) / 60 #Estimated remaining execution time
        
            print(f"{no_imgs} images cropped ... {speed:.2f} images/min | ETA: {eta:.2f} minutes                 ", end = '\r')

        #Skip if the image has no applicable bounding boxes.    
        except cv2.error as e:
            k += 1
            continue



#Function for reading and subsampling the images.
def images_subsampling(identity_file = './data/Anno/identity_CelebA.txt', atts_file =  "./data/Anno/list_attr_celeba.txt", bad_imgs_file = 'final_bad_imgs.csv'):

        #Reading the images' names and labels.
        identity = pd.read_csv(identity_file, sep = " ", header = None,
                            names = ['image', 'image_id'])

        bad_imgs = pd.read_csv(bad_imgs_file)

        #Dropping the images which could not be cropped.
        imgs_to_drop = list(set(identity['image'].tolist()).difference(os.listdir("./cropped_images/")))
        identity = identity[~identity['image'].isin(imgs_to_drop)]
        identity = identity[~identity['image'].isin(bad_imgs['image'].tolist())]

        #Selecting only images, whose classes occur at least 5 times.
        #imgs_names = identity['image_id'].value_counts().reset_index().rename(columns = {'image_id':'count','index':'image_id'}).query('count >= 5')['image_id']
        #identity_filtered = identity[identity['image_id'].isin(imgs_names)]

        #Reading attributes file
        atts = pd.read_csv(atts_file, delim_whitespace = True).reset_index().rename(columns = {'index':'image'})

        #Filtering images based on subsampling on identity file and replace -1 with 0
        atts = atts[atts['image'].isin(identity['image'])].replace(-1, 0)

        #Exclude images which fullill at least on the following images (in order to exclude noisy images)
        exclude_imgs = (atts['w5_o_Clock_Shadow'] == 1) | (atts['Wearing_Hat'] == 1) | (atts['Eyeglasses'] == 1) | (atts['Smiling'] == 1) | (atts['Narrow_Eyes'] == 1) | (atts['Pale_Skin'] == 1) | (atts['Blurry'] == 1)
        
        #final outputs:
        atts_filtered = atts[~exclude_imgs]
        identity_filtered = identity[~exclude_imgs]

        young_old_ids = identity_filtered.merge(atts_filtered[['image','Young']], on = 'image').groupby('image_id')['Young'].mean().reset_index().rename(columns = {'Young':'Young_photo'})
        young_old_ids['Young_photo'] = [0 if i <= 0.5 else 1 for i in young_old_ids['Young_photo']]
        young_old_filter = identity_filtered.merge(young_old_ids[['image_id','Young_photo']], on ='image_id').merge(atts[['image','Young']], on ='image').query('Young_photo == Young')['image'].tolist()

        atts_filtered = atts_filtered[atts_filtered['image'].isin(young_old_filter)]
        identity_filtered = identity[identity['image'].isin(young_old_filter)]

        gray_hair_ids = identity_filtered.merge(atts_filtered[['image','Gray_Hair']], on = 'image').groupby('image_id')['Gray_Hair'].mean().reset_index().rename(columns = {'Gray_Hair':'Gray_Hair_photo'})
        gray_hair_ids['Gray_Hair_photo'] = [0 if i <= 0.5 else 1 for i in gray_hair_ids['Gray_Hair_photo']]
        gray_hair_filter = identity_filtered.merge(gray_hair_ids[['image_id','Gray_Hair_photo']], on ='image_id').merge(atts[['image','Gray_Hair']], on ='image').query('Gray_Hair_photo == Gray_Hair')['image'].tolist()

        atts_filtered = atts_filtered[atts_filtered['image'].isin(gray_hair_filter)]
        identity_filtered = identity[identity['image'].isin(gray_hair_filter)]


        final_imgs = identity_filtered['image_id'].value_counts().reset_index().rename(columns = {'image_id':'count','index':'image_id'}).query('count >= 5')['image_id']
        final_identity = identity_filtered[identity_filtered['image_id'].isin(final_imgs)]
        final_atts = atts_filtered[atts_filtered['image'].isin(final_identity['image'])]

        return final_identity, final_atts



#Function for splitting the images into training set, validation set, and test set.
def images_split(identity_filtered, validation_size, test_size, seed, export = False):

    #Extracting the images' names and their labels.
    imgs = identity_filtered['image']
    labels = identity_filtered['image_id']

    #Stratified split - in order to preserve the same labels' distribution across the samples.
    _, test_imgs, __, test_labels = train_test_split(imgs, labels,
                                               test_size = test_size,
                                               random_state = seed,        
                                               stratify = labels)

    train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(_, __,
                                               test_size = validation_size/(1-test_size),
                                               random_state = seed,        
                                               stratify = __)
    
    #Reseting row indices
    train_imgs, train_labels = train_imgs.reset_index(drop = True), train_labels.reset_index(drop = True)
    valid_imgs, valid_labels = valid_imgs.reset_index(drop = True), valid_labels.reset_index(drop = True)
    test_imgs, test_labels = test_imgs.reset_index(drop = True), test_labels.reset_index(drop = True)

    #Exporting the samples.
    if export:
        pd.concat((train_imgs, train_labels), axis = 1).to_csv('./csv/train_imgs_list.csv', index = False)
        pd.concat((valid_imgs, valid_labels), axis = 1).to_csv('./csv/valid_imgs_list.csv', index = False)
        pd.concat((test_imgs, test_labels), axis = 1).to_csv('./csv/test_imgs_list.csv', index = False)

    return train_imgs, train_labels, valid_imgs, valid_labels, test_imgs, test_labels



#Function for generating balanced pairs of images (50% positive pairs, 50% negative pairs)
def pairs_generator(identity_file, atts_df, seed, target_number, exclude_imgs_list, export_name = None):
    #Start time initialization (for execution time measurement)
    start_time = time()

    #List for storing the pair labels (1 = positive pair | 0 = negative pair)
    pair_labels = []
    #List for storing the pair image names.
    pair_images_names = []

    if len(exclude_imgs_list) !=0:
        id_df = identity_file[~identity_file['image'].isin(exclude_imgs_list)]
        atts = atts_df[~atts_df['image'].isin(exclude_imgs_list)]
        labels = id_df['image_id'].reset_index(drop = True)
        image_names = id_df['image'].reset_index(drop = True)
    else:
        atts = atts_df.copy()
        labels = identity_file['image_id'].reset_index(drop = True)
        image_names = atts_df['image'].reset_index(drop = True)


    #Array with all the labels
    unique_classes = np.unique(labels)

    #Dictionary for storing images' indices for each label.
    dict_idx = {i:np.where(labels == i)[0] for i in unique_classes}
    
    male_indices = pd.DataFrame(image_names).merge(atts[['image', 'Male']], on ='image').query('Male == 1').index.to_list()
    female_indices = pd.DataFrame(image_names).merge(atts[['image', 'Male']], on ='image').query('Male == 0').index.to_list()

    np.random.shuffle(male_indices)
    np.random.shuffle(female_indices)

    indices = [i for pair in [pair for pair in zip(male_indices, female_indices)] for i in pair] + \
                list(set(female_indices + male_indices) - 
                     set([i for pair in [pair for pair in zip(male_indices, female_indices)] for i in pair]))
                       
    np.random.seed(seed)
    np.random.shuffle(indices)

    #Count initialization
    no_pairs_generated = 0

    i_male = 0
    i_female = 0

    delta_seed = 1
    #For each image, find its positive pair and negative pair
    for _ in range(len(image_names)):
        

        if no_pairs_generated < target_number/2:
            current_img_list = male_indices
            idx_a = current_img_list[i_male]

            i_male += 1
        else:
            current_img_list = female_indices
            idx_a = current_img_list[i_female]

            i_female += 1
   

        #Current anchor image - its label (person) and photo name
        label = labels[idx_a]
        current_image_name = image_names[idx_a]

        #Increment for chaning the random seed.
  
        #Positive image - random image of the same person
        np.random.seed(seed + delta_seed)
        idx_b = np.random.choice(dict_idx[label])
        


        #If the pair is existing in the list, select randomly another image again.
        #If the photos are the same (if the indices are the same), then again randomly select another image.

        #Creating a list of all posible positive pairs for given image.
        all_pos_combos = [sorted([current_image_name, image_names[b]]) for b in dict_idx[label]]

        #While loop as a constraint for generating unique positive pairs only (no duplicated such as [a,b] and [b,a])
            #Another constraint - such positive pair has to include the different images..
            #If there is not existing any combination of two positive images which is not included in the list of generated pairs - skip and proceed with the next image.
        while True:
            positive_image_name = image_names[idx_b] #Positive image name
            pair_names = [sorted(pair) for pair in pair_images_names] #List of pairs generated so far
            current_pos_pair = sorted([current_image_name, positive_image_name]) #Current generated positive pair
            

            #If the (1) current generated pair is not included in the list of all generated pairs so far and at the same time the positive pair includes 2 different photos or (2) there is no existing positive pair left - skip.
            if ((current_pos_pair not in pair_names) & (len(set(current_pos_pair)) != 1)) or (len(all_pos_combos) == 0):
                break
            
            #If is the positive pair is duplicated or has 2 same images or there are still some combinations left:
            else:
                #Change the random seed to sample different image name
                np.random.seed(seed + delta_seed) #Change the random seed to sample different image name
                idx_b = np.random.choice(dict_idx[label])

                delta_seed += 1 #Change the increment of the random seed in order to sample a different image name.
                try:
                    all_pos_combos.remove(current_pos_pair) #Remove the genereted pair from the list of all possible pairs.
                except ValueError:
                    continue
        
        #If the previous iteration has been break because there were not any combination of two images left - exit this iteration (do not look for the negative pair) and proceed with the next image.
            #In order to acheive balanced distribution of pairs (positive/negative).
        if (len(all_pos_combos) != 0):
            
            #Randomly sampling a different label (person) and its image name's index.
            np.random.seed(seed + delta_seed)
            negative_index = np.random.choice(dict_idx[np.random.choice([i for i in dict_idx.keys()
                                                                        if i != label])])

           #While loop as a constraint for generating unique positive pairs only (no duplicated such as [a,b] and [b,a]).
            while True:
                negative_image_name = image_names[negative_index] #Negative image name
                pair_names = [sorted(pair) for pair in pair_images_names] #List of pairs generated so far
                current_neg_pair = sorted([current_image_name, negative_image_name]) #Current generated negative pair
                genders = [atts.loc[atts['image'] == current_image_name,'Male'].values[0],
                               atts.loc[atts['image'] == negative_image_name,'Male'].values[0]]
                
                 #If the negative pair is already existing in the list, select randomly another image again.
                if (current_neg_pair not in pair_names) and (genders[0] != genders[1]):
                    break
                
                #If is the negative pair:
                else:
                    #Change the random seed to sample different image name
                    np.random.seed(seed + delta_seed)
                    negative_index = np.random.choice(dict_idx[np.random.choice([i for i in dict_idx.keys()
                                                                                if i != label])])
                    delta_seed += 1

            #Appending the positive pair's images' names and its label.
            pair_images_names.append([current_image_name, positive_image_name])
            pair_labels.append([1])
            no_pairs_generated += 1
            
            #Appending the negative pair's images' names and its label.
            pair_images_names.append([current_image_name, negative_image_name])
            pair_labels.append([0])
            no_pairs_generated += 1

        #Print statement
            no_pairs = f'{no_pairs_generated}/{target_number}' #How many pairs have been created so far
            print(f'{no_pairs} pairs created', end = '\r')
            
            if no_pairs_generated >= target_number:
                break

    #Data frame storing all negative and positive pairs (with the image names) and their pair label (1 = positive | 0 = negative).
    final_df = pd.concat((pd.DataFrame((pair_images_names), columns = ['img_1', 'img_2']),
                            pd.DataFrame(pair_labels, columns = ['label'])),
                            axis = 1)

    final_df = final_df.sample(frac = 1,
                                random_state = seed).sample(frac = 1,
                                random_state = seed)

    #Exporting the generated pairs and their labels.
    if export_name != None:
        final_df.to_csv(f'./csv/{export_name}_pairs.csv', index  = False)
    
    #Print statement
    print('                                                                                                 ', end = '\r') #Removing the previous statements

    #Final statement
    print(f'{no_pairs_generated} unique balanced pairs generated', '\n')
    print(f'Total Run Time: {(time() - start_time)/60:.2f} minutes', '\n')
    
    return final_df



#Function for checking descriptive information about the generated pairs in given sample
def pairs_check(pairs_df, atts):
    
    #Accessing unique labels [0, 1] and their frequencies
    labels = pairs_df['label'].replace(1, 'Positive').replace(0, 'Negative').value_counts().index
    freqs = pairs_df['label'].value_counts().values

    #Print the label distribution
    print(f"Label distribution ... {labels[0]}: {freqs[0]} ({freqs[0]/sum(freqs)*100:.0f}%) | {labels[1]}: {freqs[1]} ({freqs[1]/sum(freqs)*100:.0f}%)")

    #Acessing the unique pairs from given sample
    unique_pairs = set([str(sorted([i,j])) for i,j in zip(pairs_df['img_1'], pairs_df['img_2'])])

    #Check whether the number of unique pairs match the number of generated pairs in withn sample
    if len(unique_pairs):
        print(f'Number of unique pairs ... {len(unique_pairs)}')
    else:
        print(f"Number of unique pars doesn't match the number of pairs in given sample ({len(unique_pairs)} vs {pairs_df.shape[0]})")

    #Print whether the are any pairs which contains a single picture only
    print(f"Number of pairs containing the same image ... {(pairs_df['img_1'] == pairs_df['img_2']).sum()}")

    num_unique_imgs = len(list(set(pairs_df['img_1'].tolist() + pairs_df['img_2'].tolist())))
    print(f'Number of images ... {num_unique_imgs}')

    gender = pairs_df.merge(atts[['image','Male']].replace(1, 'Male').replace(0, 'Female'), left_on ='img_1', right_on = 'image')['Male'].value_counts().index
    gender_freqs = pairs_df.merge(atts[['image','Male']].replace(1, 'Male').replace(0, 'Female'), left_on ='img_1', right_on = 'image')['Male'].value_counts().values

    print(f"Gender distribution ... {gender[0]}: {gender_freqs[0]} ({gender_freqs[0]/sum(freqs)*100:.0f}%) | {gender[1]}: {gender_freqs[1]} ({gender_freqs[1]/sum(gender_freqs)*100:.0f}%)")



#Function for plotting an image and its positive and negative pair
def plot_pairs(pairs_df, base_img = None, resize = True):
        
        if base_img == None:
                base_img = np.random.choice(pairs_df['img_1'])

        #Filter pairs which include the baseline image
        filtered_df = pairs_df[pairs_df['img_1'] == base_img]

        #Filter a positive pair of given baseline image
        positive_img = filtered_df.query('label == 1')['img_2'].values[0]

        #Filter a negative pair of given baseline image
        negative_img = filtered_df.query('label == 0')['img_2'].values[0]

        #Folder path definition
        folder_path = 'cropped_images' if resize == True else 'data/Img/img_celeba'

        #Plot the baseline image
        fig = plt.figure(figsize=(15,15))
        ax = plt.subplot(131)
        ax.set_title(f"Anchor image - {base_img}")
        ax.imshow(cv2.cvtColor(cv2.imread(f'./{folder_path}/{base_img}'),
        cv2.COLOR_BGR2RGB))

        #Plot the positive image
        ax = plt.subplot(132)
        ax.set_title(f"Positive image - {positive_img}")
        ax.imshow(cv2.cvtColor(cv2.imread(f'./{folder_path}/{positive_img}'),
                  cv2.COLOR_BGR2RGB))

        #Plot the negative image
        ax = plt.subplot(133)
        ax.set_title(f"Negative image - {negative_img}")
        ax.imshow(cv2.cvtColor(cv2.imread(f'./{folder_path}/{negative_img}'),
                  cv2.COLOR_BGR2RGB))

        plt.tight_layout()
        plt.show()



#Function for reading pairs and load either as data frame or as arrays.
def read_pairs(sample_name, separate = True):
    
    final_df = pd.read_csv(f'./csv/{sample_name}_pairs.csv')

    if separate:
        for col in ['img_1', 'img_2']:
            final_df[col] =  [f'./cropped_images/{i}'for i in final_df[col]]

        imgs = final_df[['img_1', 'img_2']]
        labels = final_df[['label']]

        return np.array(imgs), np.array(labels)
        
    else:
        return final_df



#Function for processing pairs of images
def tf_img_pipeline(anchor, comparison):
    
    #Function for processing an image (reading, decoding, resizing and tensor conversion)
    def tf_img_processing(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels = 3)
        img = tf.image.resize(img, [224,224], method = 'bilinear')
        img = tf.image.convert_image_dtype(img, tf.float32) /  tf.constant(255, dtype = tf.float32)

        return img

    return tf_img_processing(anchor), tf_img_processing(comparison)



#Function for processing labels
def tf_label_pipeline(label):
    return tf.cast(label, tf.float32)



#Function for tensorflow dataset creation - input for modelling
def tf_data_processing_pipeline(images, labels):

    images_tf = tf.data.Dataset.from_tensor_slices((images[:, 0] , images[:, 1])).map(tf_img_pipeline)
    labels_tf = tf.data.Dataset.from_tensor_slices(labels).map(tf_label_pipeline)

    dataset = tf.data.Dataset.zip((images_tf,
                                    labels_tf)).batch(10,
                                                      num_parallel_calls = AUTOTUNE).cache().prefetch(buffer_size = AUTOTUNE)
    return dataset



#Function for plotting the validation and training loss
def plot_val_train_loss(history_model, export = True):
  
  plt.figure(figsize = (12, 10))

  plt.plot(history_model.history['loss'])
  plt.plot(history_model.history['val_loss'])

  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.legend(['train', 'validation'])

  plt.grid()
  plt.tight_layout()

  if export:
    plt.savefig('validation_train_loss.png')

  plt.show()




#Function for calculating an optimal threshold for classification by minimizing the difference between TPR and FNR.
def opt_threshold(model, tf_dataset, np_labels, plot = True):
  predictions_prob = model.predict(tf_dataset).flatten().tolist()
  df_results = pd.DataFrame({'labels': ['Positive' if i[0] == 1 else 'Negative'
                                        for i in np_labels],
                             'prob': predictions_prob})
  threshold = 1/2 * df_results.groupby('labels')['prob'].mean().sum()
  if plot:
    plt.figure(figsize = (15, 10))
    sns.boxplot(data = df_results, y = 'prob', x = 'labels')
    plt.axhline(y = threshold, color = 'r', linestyle = '--',
                linewidth = 1, label = 'Threshold')
    plt.legend()
    plt.title('Distribution of predicted probabilites per label', size = 13)
    plt.tight_layout()
    plt.show()
  return threshold



#Function for making predictions based on provided threshold.
def make_predictions(model, tf_dataset, threshold):
  predictions_prob = model.predict(tf_dataset).flatten().tolist()
  final_predictions = pd.Series(predictions_prob).apply(lambda x: 1 if x > threshold else 0)

  return final_predictions



#Function for making evaluation based on provided metric, true labels and predicted labels.
def make_evaluation(true_labels, predictions, metric):

  metrics_dict = {'accuracy':accuracy_score,
                  'recall':recall_score,
                  'precision':precision_score,
                  'F1':f1_score}

  score = metrics_dict[metric]([i[0] for i in true_labels], predictions)

  return score



