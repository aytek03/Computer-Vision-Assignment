function [imageTrain22,imageValidation22,imageTest22,options3,net4,accuracy3,plots3] = PART3_resnet18()

% AYTEKÝN YILDIZHAN
% N18147923
% CMP719 Computer Vision Homework 1
% In this part, we train the ResNEt18 with data augmentation.


%-----------This is the dog dataset's path.
dogPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','dog'); 

%-----------loading the images
images = imageDatastore(dogPath, ... 
    'IncludeSubfolders',true,'LabelSource','foldernames');

load('lgarph_1.mat'); %loading the ResNet18 layers

% [imageTrain,imageValidation,imageTest] = splitEachLabel(images,0.6,0.2,'randomize');
% 
% imageTrain2 = augmentedImageDatastore([224,224],imageTrain);
% 
% imageValidation2 = augmentedImageDatastore([224,224],imageValidation);
% 
% imageTest2 = augmentedImageDatastore([224,224],imageTest);

%We do the data augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXReflection',true, ...
    'RandXTranslation',[-30 30], ...
    'RandYTranslation',[-30 30], ...
    'RandXScale',[0.9 1.1], ...
    'RandYScale',[0.9 1.1]);

[imageTrain11,imageValidation11,imageTest11] = splitEachLabel(images,0.7,0.15,'randomize');

imageTrain22 = augmentedImageDatastore([224,224],imageTrain11, ...
    'DataAugmentation',imageAugmenter);

 imageValidation22 = augmentedImageDatastore([224,224],imageValidation11, ...
     'DataAugmentation',imageAugmenter);

 imageTest22 = augmentedImageDatastore([224,224],imageTest11, ...
     'DataAugmentation',imageAugmenter);

%Show the first eight augmented images
minibatch = preview(imageTrain22);
imshow(imtile(minibatch.input));


%tuning the hyperparameters
options3 = trainingOptions('adam', ...  
    'MiniBatchSize',64, ...    
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imageValidation22, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% train the network, our layer is lgarph_1.mat

net4 = trainNetwork(imageTrain22,lgraph_1,options3);

%Calculate the validation accuracy
YTest = classify(net4,imageTest22);
TTest = imageTest11.Labels;

accuracy3 = sum(YTest == TTest)/numel(TTest);
%plot the confusion matrix
plots3=plotconfusion(TTest,YTest);


%  
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