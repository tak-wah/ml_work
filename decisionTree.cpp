//
// Created by zysC on 2018/1/8.
//
#include <iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgcodecs/imgcodecs.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;
using namespace ml;

int main(int argc, char **argv)
{
    const int kWidth = 512, kHeight = 512; // 显示分类结果图像的高度，宽度
    Vec3b red(0, 0, 255), green(0, 255, 0), blue(255, 0, 0); // 显示分类结果的3种颜色
    Mat image = Mat::zeros(kHeight, kWidth, CV_8UC3);

    // 为3类训练样本标签赋值
    int labels[30];
    for (int i  = 0 ; i < 10; i++)
        labels[i] = 1;
    for (int i = 10; i < 20; i++)
        labels[i] = 2;
    for (int i = 20; i < 30; i++)
        labels[i] = 3;
    Mat trainResponse(30, 1, CV_32SC1, labels);

    // 用随机数生成训练样本特征向量数组
    float trainDataArray[30][2];
    RNG rng;
    for (int i = 0; i < 10; i++)
    {
        trainDataArray[i][0] = 250 + static_cast<float>(rng.gaussian(30));
        trainDataArray[i][1] = 250 + static_cast<float>(rng.gaussian(30));
    }

    for (int i = 10; i < 20; i++)
    {
        trainDataArray[i][0] = 150 + static_cast<float>(rng.gaussian(30));
        trainDataArray[i][1] = 150 + static_cast<float>(rng.gaussian(30));
    }

    for (int i = 20; i < 30; i++)
    {
        trainDataArray[i][0] = 320 + static_cast<float>(rng.gaussian(30));
        trainDataArray[i][1] = 150 + static_cast<float>(rng.gaussian(30));
    }

    Mat trainData(30, 2, CV_32FC1, trainDataArray);

    //CvDTree dtree;
    // 决策树的训练参数，在后面的源代码分析中会详细讲解
    //CvDTreeParams params(5, 1, 0, true, 2, 0, true, true, NULL);
    // 训练决策树
    Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainResponse);
    Ptr<DTrees> dtree = DTrees::create();
    dtree->setMaxDepth(8);   //设置最大深度
    dtree->setMinSampleCount(2);
    dtree->setUseSurrogates(false);
    dtree->setCVFolds(0); //交叉验证
    dtree->setUse1SERule(false);
    dtree->setTruncatePrunedTree(false);
    dtree->train(tData);

    // 对图像平面内所有点进行预测，根据分类结果显示不同的颜色
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            Mat sampleMat = (Mat_<float>(1, 2) << j, i);
// 用决策树进行预测，返回分类结果
            float response = dtree->predict(sampleMat);
// 根据分类结果显示不同的颜色
            if (response == 1)
                image.at<Vec3b>(i, j) = red;
            else if (response == 2)
                image.at<Vec3b>(i, j) = green;
            else
                image.at<Vec3b>(i, j) = blue;
        }
    }

// 用不同的亮度显示3类训练样本
    for (int i = 0; i < trainData.rows; i++)
    {
        const float* v = trainData.ptr<float>(i);
        Point pt = Point((int)v[0], (int)v[1]);

        if (labels[i] == 1)
            circle(image, pt, 5, Scalar::all(0), -1, 8);
        else if (labels[i] == 2)
            circle(image, pt, 5, Scalar::all(128), -1, 8);
        else
            circle(image, pt, 5, Scalar::all(255), -1, 8);
    }

    imshow("Decision tree classifier demo", image);
    waitKey(0);
    return 0;
}


