# Computer-Vision-Assignment
Classifying Dog Images / Training a Classifier from Scratch / Transfer Learning in Convolutional Neural Networks

In this project, there are three parts and five MATLAB m. files.
MATLAB 2018b is used.

These are;
1. all.command.m
2. PART1.m
3. PART2_no_data_aug.m
4. PART2_dataAug.m
5. PART3_resnet18.m

You load lgarph_1.mat to work ResNet18.

###all.command.m ->>> SHORTCUT
In this matlab code, you can see four matlab code.
Copy of first code and paste it ccommand line and run it. PART1.m will work.

Copy of second code and paste it ccommand line and run it. PART2_no_data_aug.m will work.

Copy of third code and paste it ccommand line and run it. PART2_dataAug.m will work.

Copy of last code and paste it ccommand line and run it. PART3_resnet18.m will work.


### PART1.m
In this code, we do the data split.

First Data Split Option. %60 train, %20 Validation and %20 test
[imageTrain,imageValidation,imageTest] = splitEachLabel(images,0.6,0.2,'randomize');

%Second Data Split Option. . %70 train, %15 Validation and %15 test

[imageTrain11,imageValidation11,imageTest11] = splitEachLabel(images,0.7,0.15,'randomize');

You can choose one of them.

You can write this code on the command line and PART1.m works.
[imageTrain,imageValidation,imageTest,imageTrain2,imageValidation2,imageTest2] = PART1()

It returns original size of data set and resizing of train, validation and test image set.

### PART2_no_data_aug.m
In this code, we build a deep net and train the net without data augmentation..
We set the layers, options of the net and monitorize the curves.
Finally, we calculate the validation accuracy, confisuon matrix and show the validation and loss curves.

 
You can write this code on the command line and PART2_no_data_aug.m works.
[layers,options,net2,accuracy,plots] = PART2_no_data_aug(imageTest,imageTrain2,imageValidation2,imageTest2)


### PART2_dataAug.m
In this code, we build a deep net and train the net with data augmentation.
We set the layers, options of the net and monitorize the curves.
Finally, we calculate the validation accuracy, confisuon matrix and show the validation and loss curves.

 
You can write this code on the command line and PART2_no_dataAug.m works.
[imageTrain12,imageValidation12,imageTest12,layers,options2,net3,accuracy2,plots2] = PART2_dataAug(imageTrain,imageValidation,imageTest)


### PART3_resnet18.m
In this code, we build a ResNet18 deep net and train the net with data augmentation.
We set the layers, options of the net and monitorize the curves.
Finally, we calculate the validation accuracy, confisuon matrix and show the validation and loss curves.

 
You can write this code on the command line and PART3_resnet18.m works.
load('lgarph_1.mat') %this code loads the layers of ResNet18.
[imageTrain22,imageValidation22,imageTest22,options3,net4,accuracy3,plots3] = PART3_resnet18()
