# Capitulum Counter Pipeline

This pipeline automatically counts the number of Sphagnum capitula in your images and produces crops of all the images, annotated images, which can be used to check the annotations, a CSV-file of the annotations,
which can be used in annotation software (i.e. VGG), and a CSV-file with the automated count and the corrected count for all images.

It is important to note that all the libraries required to run the code need to be loaded in (by running cell 2).
If any of the packages are not yet installed on your computer, you will need to install those first (the code on how to do that is provided in cell 1).

To run the pipeline successfully, you need to set:

- a source folder. This is the folder where your image is located. Make sure you end the path with /.
- a file format. This is the file type i.e. .jpg.
- a destination folder. This is the folder where you want to save the annotated image and the CSV-file. Make sure you end the path with /.

File path syntax:
The file path to folder for both retreiving of images and storing of data files needs to be carefully inputted.
The syntax on how to do it correctly is explained with the following example:
The correct way of naming the source folder is:
source_folder = 'C:/Users/abc/def/' Wich ensures Python searches within dfg for your image. However if you write:
source_folder = 'C:/Users/abc/def' Python searches within abc, where it most likely will not find your image and will throw an error.
It is also important to note that for the destination folder the same principle applies. 
However Python will NOT throw an error when the incorrect file path ('C:/Users/abc/dfg') is provided, but will instead store your files in abc.

When you run the code, a box will pop up that allows you to select whether you want to use the version with a GUI for manual thresholding, or if you want to use an automated version
with manually entered threshold values (see Supplemental Table 1 for examples), which may be preferable if you have a large set of images of the same quality and lighting.
If you select the version with GUI, a new window will pop up whenever it reaches the thresholding step that allows you to modify the threshold with two sliders for each image separately.

Supplemental Table 1. General ranges of lower threshold values for channel HSV S. 
Threshold values are mostly dependent on the amount of non-moss material in an image, lighting and the colouration of the Sphagnum. 

|Image characterisation				|Lower threshold range			|Lower threshold mean			|Upper threshold |
| --------------					| ------------- 				| -------------- 				| -------------- |
|Pale discoloured Sphagnum, bright image 	|80-140					|		107 					|	256			|
|Wet saturated, dark image					|50-200					|		157						|	256 |
|Intermediates								|50-180					|		110						|	256 |
|Mean value (default)						|						|		125						|	256 |
