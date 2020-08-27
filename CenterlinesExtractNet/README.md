可运行的！！！

将一个UNet分割网络改为DeepCL中心线提取网络，未进行实验。

仅用果蝇细胞图像运行！！！


1、data
果蝇细胞数据集，一共160对原图和分割图。
data/
             centerlines/    #分割图
             imgs/             #原图
             points/           #分割图
             test/               #测试图片

2、train
cd DeepCenterLines
python train.py -e 50 -b 2 -l 0.0001 
-e       总批次数epoch
-b       批次大小batch-size
-l        学习率learn-rate

3、test
python predict.py -m checkpoints/CP_epoch1.pth -i data/test/1.bmp -o1 data/test/output1.bmp -o2 data/test/output2.bmp

-m      模型的路径
-i        输入图像
-o1     输出图像1
-o2     输出图像2