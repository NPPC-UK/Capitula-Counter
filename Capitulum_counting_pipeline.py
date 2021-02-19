#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!/usr/bin/env python
# coding: utf-8

#Install any packages are not already available on your computer
#You can copy paste the names of the packages below into the install line on numpy's position
#These include:
# numpy, pandas, matplotlib, opencv-python, plantcv, scikit-image, glob3 (if it does not work while importing)
import sys
get_ipython().system('{sys.executable} -m pip install numpy')


#1
#Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from plantcv import plantcv as pcv
import glob
from skimage import exposure, feature
import json
import os
import sys
import PySimpleGUI as sg
from PIL import Image


#2
#Set the source folder to the folder that contains your image ('C:/abc/dfg/')
#Set filename to the name of the image ('image.jpg')
#Set the destination folder to the folder where you want the results to be saved
#Set the lower limit of the threshold. You can find a table with ranges in the guide
source_folder = 'C:/Users/abc/def/' #Use / in the file path, otherwise Python does not understand
fileformat = '.jpg' #The format of your images. the format (.jpg) will help to load in the correct images
destination_folder = 'C:/Users/abc/def/ghi/' #making a separate folder for your data makes it easier to find



#2
#Define functions for every step of the pipeline. You do not have to change anything here!
#order_points will find the corners of the box
def order_points(pts):
    # initialise a list of coordinates that will be ordered such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

#cropouter performs multiple steps to automatically crop the image to the box edges
def cropouter(image):
    #First it will blur the image, convert it to HSV colour space after which the blue colour of the box is defined,
    #which will be used to mask everything but the blue box
    blur = cv2.blur(image, (50,50))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100,100,40])
    upper_blue = np.array([130,255,255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue).astype('uint8')
    newmask = cv2.erode(mask, kernel = (1,1), iterations = 10)
    dfmask = pd.DataFrame(newmask != 0)

    #The mask is used to determine the coordinations of the box and find the corners
    coord = np.where(dfmask)
    x = coord[0].reshape(1,coord[0].shape[0])
    y = coord[1].reshape(1,coord[1].shape[0])
    box = np.concatenate([y,x])
    box1 = np.swapaxes(box, 0,1).reshape(box.shape[1], 2)
    rect = order_points(box1)
    lu, ru, rb, lb = rect
    
    #Some of the images that were used to develop the pipeline had a blue demarcation in the lower right corner (was later removed)
    #with the pythagorean theorem you can always find the true corner when there is an artifact such as this
    #This code finds the right lower corner this way
    w = np.diff([lu[0], ru[0]])
    diffy = np.diff([lu[1], ru[1]])
    h = np.diff([lu[1], lb[1]])
    diffx = np.diff([lu[0], lb[0]])
    if diffy < 0:
        if diffx < 0:
            diffx = -diffx
        else:
            diffx = diffx
        widthbox = np.sqrt(w **2 + diffy ** 2)
        heightbox = np.sqrt(h ** 2 + diffx **2)
    else:
        if diffx > 0:
            diffx = -diffx
        else:
            diffx = diffx
        widthbox = np.sqrt(w **2 + diffy ** 2)
        heightbox = np.sqrt(h ** 2 + diffx **2)

    #The corners are then used to alter the perspective such that it is always a topview and crops the image
    #to the edges of the blue box
    lu = [lu[0], lu[1]]
    ru = [ru[0], ru[1]]
    lb = [lb[0], lb[1]]
    rb = [ru[0] + diffx, ru[1] + h]
    rect = np.array([lu, ru, rb, lb], dtype = "float32")
    dst = np.array([
            [0, 0],
            [int(widthbox) - 1, 0],
            [int(widthbox) - 1, int(heightbox) - 1],
            [0, int(heightbox) - 1]], dtype = "float32")
    M2 = cv2.getPerspectiveTransform(rect, dst)
    cropout = cv2.warpPerspective(image, M2, (int(widthbox), int(heightbox)))
    return cropout

#3
#Import the image you want to annotate
filenames = [f for f in sorted(os.listdir(source_folder)) if ((str(f))[-4:] == fileformat.lower()) or ((str(f))[-4:] == fileformat.upper())]
images=[]
for file in filenames:
    filepath = source_folder + file
    image = cv2.imread(filepath)
    images.append(image)

#4
#This part will crop the images to the edges of the box
interface=True
upper_threshold = 256
lower_threshold = 125

layout = [[sg.Text('Select version'), sg.Text('')],
              [sg.Radio('Use UI Version', 1, default= True, key='UI_Version')],
              [sg.Radio('Use Automated Version', 1, key='Automated_Version'),
              sg.InputText(upper_threshold, size=(8, 1), key='input_upper_threshold'), sg.Text('Upper Threshold'),
              sg.InputText(lower_threshold, size=(8, 1), key='input_lower_threshold'), sg.Text('Lower Threshold')],
              [sg.B('OK',key='okay button')]]
            
window = sg.Window('Select Version', layout)
    
    # Event loop
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == 'input_upper_threshold':
        window['input_upper_threshold'].update('')
    if event == 'input_lower_threshold':
        window['input_lower_threshold'].update('')
    if event == 'okay button':
        if values['UI_Version']:
            interface=True
        else:
            interface=False
        
        upper_threshold = int(values['input_upper_threshold'])
        lower_threshold = int(values['input_lower_threshold'])
        break

        
    #This deletes the used preview image and therefore exits the GUI section
window.close()

i = 0
regioncounts = []
files = []
for im in images:
    cropout = cropouter(im)
    
    #5
    #This part standardizes the crop to a certain width and height
    stnd = cv2.resize(cropout, (1100,1350), interpolation = cv2.INTER_LINEAR)

    #6
    #This part corrects the image using the white balance correction
    roi = (390, 250, 40, 40)
    corrected_img = pcv.white_balance(stnd, mode='hist', roi= roi)
    
    #7
    #This part crops the edges of the box off
    y1 = 351
    x1 = 121
    fcrop = corrected_img[y1:y1+800, x1:x1+800, :]
    filenamecrop = destination_folder + 'crop_' + filenames[i]
    cv2.imwrite(filenamecrop, fcrop)

     #8
    if interface:
       #This is the part where you mask all the non-moss material
        preview_image_size=500
        #This creates an initial preview image for the GUI
        im1 = Image.open(filenamecrop)
        im1.thumbnail((preview_image_size, preview_image_size), Image.ANTIALIAS)
        im1.save('preview_threshold.png')

        #This resizes the fcrop and hsv for preview images
        preview_fcrop = cv2.resize(fcrop, (preview_image_size, preview_image_size), interpolation = cv2.INTER_LINEAR)
        preview_hsv = cv2.cvtColor(preview_fcrop, cv2.COLOR_BGR2HSV)
        (preview_h, preview_s, preview_v) = (preview_hsv[:,:,0], preview_hsv[:,:,1], preview_hsv[:,:,2])

        #Layout of the interactive GUI
        layout = [[sg.Text('Select Threshold'), sg.Text('')],
                  [sg.T('upper threshold'),
                   sg.B('<', key='decrease_ut'),
                   sg.Slider(range=(0,256), default_value=upper_threshold, key = 'upper threshold', enable_events= True, 
                         size=(20,15), orientation='horizontal', font=('Helvetica', 12)),
                   sg.B('>', key='increase_ut')],
                  [sg.T('lower threshold'),
                   sg.Button('<', key='decrease_lt'),
                   sg.Slider(range=(0,256), default_value=lower_threshold, key = 'lower threshold', enable_events= True, 
                         size=(20,15), orientation='horizontal', font=('Helvetica', 12)),
                   sg.B('>', key='increase_lt')],
                  [sg.B('OK',key='okay button')],
                  [sg.Image('preview_threshold.png', key = 'image')]]

        window = sg.Window('Image Preview', layout)
    
        # Event loop
        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == 'Exit':
                break
            if event == 'okay button':
                lower_threshold = int(values['lower threshold'])
                upper_threshold = int(values['upper threshold']) 
                break
            if event == 'decrease_ut':
                window['upper threshold'].update(values['upper threshold'] - 1)
            if event == 'increase_ut':
                window['upper threshold'].update(values['upper threshold'] + 1)
            if event == 'decrease_lt':
                window['lower threshold'].update(values['lower threshold'] - 1)
            if event == 'increase_lt':
                window['lower threshold'].update(values['lower threshold'] + 1)
        
            #This creates and updates the preview image for the GUI
            preview_mask = cv2.inRange(preview_s, int(values['lower threshold']), int(values['upper threshold']))
            preview_noleaves = cv2.bitwise_and(preview_fcrop, preview_fcrop, mask = preview_mask)
            cv2.imwrite('preview_threshold.png', preview_noleaves)
        
            window['image'].update('preview_threshold.png')
        
        #This deletes the used preview image and therefore exits the GUI section
        window.close()
        os.remove('preview_threshold.png') 
    
    
    #This is the part where you continue with masking all the non-moss material
    hsv = cv2.cvtColor(fcrop, cv2.COLOR_BGR2HSV)
    (h, s, v) = (hsv[:,:,0], hsv[:,:,1], hsv[:,:,2])
    mask = cv2.inRange(s, lower_threshold, upper_threshold)
    noleaves = cv2.bitwise_and(fcrop, fcrop, mask = mask)

   

    #13
    #This is the part that does the annotating
    #The original image is converted from BGR to RGB
    #The thresholded image is converted to the colour space for annotating
    rgb = cv2.cvtColor(fcrop, cv2.COLOR_BGR2RGB)
    ycrcb = cv2.cvtColor(noleaves, cv2.COLOR_BGR2YCrCb)
    (y, cr, cb) =(ycrcb[:,:, 0], ycrcb[:,:, 1], ycrcb[:,:, 2])
    #Add correct channel to find capitula
    inv= np.subtract(255, cb)
    #This is the function that will perform the annotation
    blobs_dog = feature.blob_dog(inv, min_sigma = 8, max_sigma = 11, threshold = 0.002, overlap = 0.5, sigma_ratio=1.1)

    #Then the annotations are plotted on the original image and saved
    fig, axes = plt.subplots(figsize=(10, 10))
    axes.imshow(rgb)
    for blob in blobs_dog:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='yellow', linewidth=1, fill=False)
        axes.add_patch(c)
    plt.axis('off')
    plt.tight_layout()
    newfile = destination_folder + 'anno_' + filenames[i]
    fig.savefig(newfile)
    plt.close('all')

    #Then the count and individual annotations are extracted
    region_count = len(blobs_dog)
    regioncounts.append(region_count)
    filesize = os.stat(filenamecrop).st_size
    regions = []
    for idx, (blob) in enumerate(blobs_dog):
        region_id = idx
        y, x, r = blob
        region_shape_attributes = json.dumps({"name":"circle","cx":int(x),"cy":int(y),"r":"{:.2f}".format(round(r, 3))})
        regions.append((region_id, region_shape_attributes))
    regionid = []
    regionshape = []
    for region, shape in regions:
        regionid.append(region)
        regionshape.append(shape)
    fileattributes = ({})
    regionattributes = ({})
    
    #then the file with annotations is made and saved
    split = '.' + filenames[i].split('.')[-1]
    file = filenames[i].replace(split, '')
    files.append(file)
    anno = {'filename': file, 'file_size': filesize, 'file_attributes': fileattributes, 'region_count' : region_count, 
            'region_id' : regionid, 'region_shape_attributes' : regionshape, 'region_attributes': regionattributes}
    annotations = pd.DataFrame(pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in anno.items() ])))
    annotations = annotations.ffill()
    annotations['file_attributes'] = '{}'
    annotations['region_attributes'] = '{}'
    annotations['file_size'] = annotations['file_size'].astype('int64')
    annotations['region_count'] = annotations['region_count'].astype('int64')
    newfile = destination_folder + 'anno_' + file + '.csv'
    annotations.to_csv(newfile, index=False, header = True)
    i += 1

#Then a file is made with all the automated and corrected counts and is saved with the correct filenames
newfilenames = pd.Series(files)
regioncounts = pd.Series(regioncounts)
correctedcount = round(np.multiply(regioncounts, 0.81))
count = pd.concat([newfilenames, regioncounts, correctedcount], axis = 'columns')
count = count.rename(columns = {0:'Filename', 1:'Automated_count', 2: 'Corrected_count'})
count.to_csv(destination_folder + 'All_counts.csv', header = True, index = False)

