# ------------------------------------------------------------------------------
# Script Name: prepareDataAndPredict.py
# Description: The script used for data preparation and inference. The input images,
#           mask images and segmentation images are stored.
# Author: Luka Siktar
# Date Created: 2025-01-16
# Last Modified: 2025-01-16
# Version: 1.0
# Contact: lsiktar@fsb.hr, luka.siktar@gmail.com
# ------------------------------------------------------------------------------
import os
import cv2
import torch
import numpy as np
from scipy.interpolate import splprep, splev


def prepare_data_and_predict(main_dir, images_dir, net):
    #Image size
    dir_name=images_dir.split("/")[-1]
    patch_size=[224, 224]
    x, y = patch_size[0], patch_size[1]

    #Read the file names and create complete image paths
    images_paths=[]
    for file in os.listdir(images_dir):
        full_path=os.path.join(images_dir, file)
        images_paths.append(full_path)

    #Create the ouput directories
    #images=[]
    predictions=[]

    output_dir = f"{main_dir}_processed/output_images" # Directory where output images will be saved
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    output_dir1 = f"{main_dir}_processed/output_segmentations" # Directory where segmentation images will be saved
    os.makedirs(output_dir1, exist_ok=True) 

    output_dir2 = f"{main_dir}_processed/input_images"  # Directory where input images will be saved
    os.makedirs(output_dir2, exist_ok=True) 

    for counter, image_path in enumerate(images_paths):

        orig_image=cv2.imread(image_path)
        path=image_path.split("_")[-2] 
        #deg=image_path.split("_")[-1].split(".j")[-2]
        deg=image_path.split("/")[-1].split(".j")[-2].split("'")[-1]
        #deg=image_path.split("-")[-1].split(".p")[0]
        deg=deg.replace(f"decdeg","deg")        #If images have wrong name, change it
        print(deg)
        #images.append(orig_image)
        
        #Convert image to grayscale
        image= cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)

        #Store the input=original images
        #output_path_original_image= os.path.join(output_dir2, f"{dir_name}_original_{path}.png")
        output_path_original_image= os.path.join(output_dir2, f"original_{deg}.png")

        cv2.imwrite(output_path_original_image, image)

        # Convert the image to a PyTorch tensor
        image_tensor = torch.from_numpy(image).float()          # Convert to float32 tensor
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)   # Add batch and channel dimensions

        #Resize the tensor
        resized_image=cv2.resize(image, (224,224))

        #Input to the neural network
        input_nn = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).float().cuda()

        with torch.no_grad():
                #Model outputs
                outputs, _, _, _  = net(input_nn)
                #Model predictions
                out = torch.sigmoid(outputs).squeeze()
                pred = out.cpu().detach().numpy()

                if x != patch_size[0] or y != patch_size[1]:
                    pred = cv2.resize(out, (y, x), interpolation = cv2.INTER_NEAREST)
                
                #Create a binary mask from predicitons
                a = 1.0*(pred>0.5)
                prediction = a.astype(np.uint8)
                #Resize to original image (fix - autmate)
                prediction=cv2.resize(prediction, (image.shape[1],image.shape[0]))
                prediction = cv2.normalize(prediction, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                predictions.append(prediction)
                #Store masks
                #output_path = os.path.join(output_dir, f"{dir_name}_slice_{path}.png")
                output_path = os.path.join(output_dir, f"mask_{deg}.png")

                

                #Find contours on predicted masks (used for visualization)
                contours, hierarchy = cv2.findContours(prediction,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #Smooth the contours
                smoothened = []
                for contour in contours:
                    
                    x_1,y_1 = contour.T
                    # Convert from numpy arrays to normal arrays
                    x_1 = x_1.tolist()[0]
                    y_1 = y_1.tolist()[0]
                    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
                    tck, u = splprep([x_1,y_1], u=None, s=0.0, k=1, per=1)
                    # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
                    u_new = np.linspace(u.min(), u.max(), 100)

                    
                    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
                    x_new, y_new = splev(u_new, tck, der=0)
                    # Convert it back to numpy format for opencv to be able to display it
                    res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
                    smoothened.append(np.asarray(res_array, dtype=np.int32))

                #Show only the biggest contour
                cv2.drawContours(orig_image, smoothened, 0, (0,0,255),3)

                #Smooth the conoturs and store them
                prediction[:] = 0  
                cv2.drawContours(prediction, smoothened, 0, (255,255,255),-1)
                cv2.imwrite(output_path, prediction)

                #Store the images with found contours
                #output_path1 = os.path.join(output_dir1, f"{dir_name}_segmentation_{path}.png")
                output_path1 = os.path.join(output_dir1, f"segmentation_{deg}.png")

                cv2.imwrite(output_path1, orig_image)


                
                

