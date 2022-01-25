function [layers,options,net2,accuracy,plots] = PART2_no_data_aug(imageTest,imageTrain2,imageValidation2,imageTest2)

% AYTEKÝN YILDIZHAN
% N18147923
% CMP719 Computer Vision Homework 1
% In this part, we train the net without data augmentation.

%deepNetworkDesigner

%----------------------------------------------------------------

% dogPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
%     'nndatasets','dog');
% images = imageDatastore(dogPath, ...
%     'IncludeSubfolders',true,'LabelSource','foldernames');

%[imageTrain2,imageValidation2,imageTest2] = PART1();

% 
% [imageTrain,imageValidation,imageTest] = splitEachLabel(images,0.6,0.2,'randomize');
% 
% 
% imageTrain2 = augmentedImageDatastore([256,256],imageTrain);
% 
% imageValidation2 = augmentedImageDatastore([256,256],imageValidation);
% 
% imageTest2 = augmentedImageDatastore([256,256],imageTest);

%building the layers
layers = [
    
    imageInputLayer([256 256 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
   % dropoutLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
   % dropoutLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
   % dropoutLayer
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(20)
    softmaxLayer
    classificationLayer];

%analyzeNetwork(layers) %Analyzing the layers of the net.


%tuning the hyperparameters
options = trainingOptions('adam', ...  
    'MiniBatchSize',64, ...    
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imageValidation2, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%train the net
net2 = trainNetwork(imageTrain2,layers,options);

%Calculate the validation accuracy
YTest = classify(net2,imageTest2);
TTest = imageTest.Labels;
 
accuracy = sum(YTest == TTest)/numel(TTest);

%plot the confusion matrix
plots = plotconfusion(TTest,YTest);
 
% Show the some of 4 example
% idx = randperm(numel(imageValidation.Files),4);
% figure
% for i = 1:4
%     subplot(2,2,i)
%     I = readimage(imageValidation,idx(i));
%     imshow(I)
%     label = YTest(idx(i));
%     title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
% end
% 
 end