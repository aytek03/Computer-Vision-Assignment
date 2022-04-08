function [imageTrain,imageValidation,imageTest,imageTrain2,imageValidation2,imageTest2] = PART1()

% AYTEKÝN YILDIZHAN
% N18147923
% CMP719 Computer Vision Homework 1
% In this part, we do the data split.

%deepNetworkDesigner

%----------------------------------------------------------------


%-----------This is the dog dataset's path.
dogPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','dog'); 

%-----------loading the images
images = imageDatastore(dogPath, ... 
    'IncludeSubfolders',true,'LabelSource','foldernames');

labelCount = countEachLabel(images); % Classes of Dogs 

%First Data Split. %60 train, %20 Validation and %20 test
[imageTrain,imageValidation,imageTest] = splitEachLabel(images,0.7,0.15,'randomize');

%Second Data Split. %70 train, %15 Validation and %15 test

%[imageTrain11,imageValidation11,imageTest11] = splitEachLabel(images,0.7,0.15,'randomize');


%Without Data Augmentation, just resizing for first data split
%----------------------------------------------------------------

imageTrain2 = augmentedImageDatastore([256,256],imageTrain);

imageValidation2 = augmentedImageDatastore([256,256],imageValidation);

imageTest2 = augmentedImageDatastore([256,256],imageTest);

%Without Data Augmentation, just resizing for second data split
%----------------------------------------------------------------

% imageTrain12 = augmentedImageDatastore([256,256],imageTrain11);
 
% imageValidation12 = augmentedImageDatastore([256,256],imageValidation11);

% imageTest12 = augmentedImageDatastore([256,256],imageTest11);


% layers = [
%     
%     imageInputLayer([256 256 3])
%     
%     convolution2dLayer(3,8,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     dropoutLayer
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     dropoutLayer
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     dropoutLayer
%     maxPooling2dLayer(2,'Stride',2)
%     
%     fullyConnectedLayer(20)
%     softmaxLayer
%     classificationLayer];
% 

% options = trainingOptions('sgdm', ...
%     'InitialLearnRate',0.0001, ...
%     'MaxEpochs',30, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',imdsValidation2, ...
%     'ValidationFrequency',30, ...
%     'Verbose',false, ...
%     'Plots','training-progress');
%  

% 
% net2 = trainNetwork(imageTrain2,layers,options);
% 

end