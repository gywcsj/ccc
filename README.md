# 基于深度学习的猫狗图像分类问题方案
# 1、项目介绍
    项目属于计算机视觉领域中的图像分类问题。图像分类的过程非常明确：给定已经标记的数据集，提取特征，训练得到分类器。项目使用猫狗数据集，任务是对给定的猫和狗的图片进行分类，因此是二分类问题。
    项目要解决的问题是使用12500张猫和12500张狗的图片作为测试集，训练一个合适的模型，能够在给定的12500张未见过的图像中分辨猫和狗。在机器学习领域，用于分类的策略，包括K均值聚类、支持向量机等，均能够用于处理该二分类问题。但在图像分类领域，神经网络技术具有更加明显的优势，特别是深度卷积神经网络，已成功地应用于图像识别领域。
    所以，利用GoogLeNet卷积神经网络（CNN）进行深度学习，通过模型迁移，使用预训练过的深度学习网络作为主干来提取图片的底层特征，编写代码，使用给定数据集，合理设置网络参数，训练得到新模型，计算出模型准确率指标并分析。
# 2、配置环境
## 2.1 所需应用
    本项目基于matlab2020b版本运行，利用GoogLeNet卷积神经网络（CNN）进行迁移学习，需要在matlab的APP区Deep Network Designer中加载预训练网格GoogLeNet。
## 2.2 硬件资源
    默认情况下，如果GPU可用，则trainNetwork会将其用于训练。尽量保证有单gpu资源，可以保证得以运行。在GPU上训练需要具有计算能力的支持CUDA的GPU。
## 2.3 运行环境
    运行平台Windows10，安装10.2版本的cuda库。
# 3、运行代码
## 3.1 加载数据
    clc;close all;clear;
    Location = 'C:\Users\chenlab3\Desktop\机器学习算法作业\结课作业\dog_and_cat_12500';%输入自己的数据集
    imds = imageDatastore(Location ,...  %若使用自己的数据集则改为Location（不加单引号）;图像数据的数据存储
                       'IncludeSubfolders',true,...%子文件夹包含标记
                       'LabelSource','foldernames');%提供标签数据的源
    [imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');%将数据集按7:3的比例分为训练集和测试集
## 3.2 加载预训练网络
    net = googlenet;
## 3.3 从训练有素的网络中提取图层，并绘制图层图
    lgraph = layerGraph(net);%从训练网络中提取layer graph
    %绘制layer graph
    figure('Units','normalize','Position',[0.1 0.1 0.8 0.8]);
    plot(lgraph)
    net.Layers(1)
    
    inputSize = net.Layers(1).InputSize;
## 3.4 替换最终图层
    为了训练Googlenet去分类新的图像，取代网络的最后三层。这三层为'loss3-classifier', 'prob', 和'output'，包含如何将网络的提取的功能组合为类概率和标签的信息。在层次图中添加三层新层： a fully connected layer, a softmax layer, and a classification output layer将全连接层设置为同新的数据集中类的数目相同的大小，为了使新层比传输层学习更快，增加全连接层的学习因子。
    lgraph = removeLayers(lgraph,{'loss3-classifier','prob','output'});
    numClasses = numel(categories(imdsTrain.Labels));
    newLayers = [
              fullyConnectedLayer(numClasses,'Name','fc','weightLearnRateFactor',10,'BiasLearnRateFactor',10)
              softmaxLayer('Name','softmax')
              classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);

    %将网络中最后一个传输层（pool5-drop_7x7_s1）连接到新层
    lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');

    % 绘制新的图层
    figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph)；
    ylim([0,10])；
## 3.5 冻结初始图层
    这个网络现在已经准备好训练新的图像集。或者你可以通过设置这些层的学习速率为0来“冻结”网络中早期层的权重在训练过程中trainNetwork不会更新冻结层的参数，因为冻结层的梯度不需要计算，冻结大多数初始层的权重对网络训练加速很重要。如果新的数据集很小，冻结早期网络层也可以防止新的数据集过拟合。
    layers = lgraph.Layers;
    connections = lgraph.Connections;  
    layers(1:110) = freezeWeights(layers(1:110));%调用freezeWeights函数，设置开始的110层学习速率为0
    lgraph = createLgraphUsingConnections(layers,connections);%调用createLgraphUsingConnections函数，按原始顺序重新连接所有的层。
## 3.6 训练网络
### 3.6.1 数据处理
    pixelRange = [-30 30];
    % 图像数据增强器为图像增强配置了一组预处理选项，如调整大小、旋转和反射。
    imageAugmenter = imageDataAugmenter(...
                                    'RandXReflection',true,...
                                    'RandXTranslation',pixelRange,...  %应用于输入图像的水平平移范围。平移距离以像素为单位。
                                    'RandYTranslation',pixelRange);
    %对输入数据进行数据加强;可以减少过拟合
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
       'DataAugmentation',imageAugmenter);
    %  自动调整验证图像大小    
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
## 3.6.2 设置训练参数
    options = trainingOptions('sgdm', ...
      'MiniBatchSize',64, ...    %调整参数,指定小批量的大小
      'MaxEpochs',6, ...      %调整参数,指定要训练的最大轮数
      'InitialLearnRate',1e-4, ...%指定全局学习率
      'ValidationData',augimdsValidation, ...%指定验证数据
      'ValidationFrequency',3, ...  %设置验证频率
      'ValidationPatience',Inf, ...%启用自动验证停止:在验证损失停止减少时自动停止训练
      'Verbose',true ,...
      'Plots','training-progress');
### 3.6.3 开始训练网络
    googlenetTrain = trainNetwork(augimdsTrain,lgraph,options);
## 3.7 使用训练好的模型随机显示分类的图片及其标签和概率
    [YPred,scores] = classify(googlenetTrain,augimdsValidation);
     m=30;
     idx = randperm(numel(imdsValidation.Files),m);
     figure
    for i = 1:m
      subplot(5,6,i)
      I = readimage(imdsValidation,idx(i));
      imshow(I)
      label = YPred(idx(i));
      title(string(label) + ";" + num2str(100*max(scores(idx(i),:)),3) + "%");
    end
## 3.8 对验证图像进行分类
    [YPred,probs] = classify(googlenetTrain,augimdsValidation);%使用训练好的网络进行分类
    accuracy = mean(YPred == imdsValidation.Labels)%计算网络的精确度
## 3.9 保存训练好的模型
    save googlenet_12500 googlenetTrain;
    % save  x  y;  保存训练好的模型y（注意：y为训练的模型，即y = trainNetwork()），取名为x
# 4、实例结果
    accuracy =
      0.9875
