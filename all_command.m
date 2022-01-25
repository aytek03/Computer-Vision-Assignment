function []= all_command()
% AYTEKÝN YILDIZHAN
% N18147923
% CMP719 Computer Vision Homework 1
% In this matlab function, it consists of four command line.
% copy and run one by one in order.
%
%
%
[imageTrain,imageValidation,imageTest,imageTrain2,imageValidation2,imageTest2] = PART1();

[layers,options,net2,accuracy,plots] = PART2_no_data_aug(imageTest,imageTrain2,imageValidation2,imageTest2);

[imageTrain12,imageValidation12,imageTest12,layers,options2,net3,accuracy2,plots2] = PART2_dataAug(imageTrain,imageValidation,imageTest);

[imageTrain22,imageValidation22,imageTest22,options3,net4,accuracy3,plots3] = PART3_resnet18();

end