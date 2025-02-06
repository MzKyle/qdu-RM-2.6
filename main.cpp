//此代码将第四行red为检测蓝色装甲板 blue为检测红色装甲板
//思路：分离通道过滤留下对应颜色像素，然后合并，转灰度图像，并二值化，使用canny算法显示边缘，然后进行形态学操作使灯条完整显示，再绘制目标检测矩形，算出装甲板中心点
/*
推荐值(已经将推荐值改为默认值):
检测红色装甲板:
二值化阈值0 腐蚀次数0 膨胀次数2 红蓝差值100
检测蓝色装甲板:
二值化阈值0 腐蚀次数0 膨胀次数2 红蓝差值130
*/
#define red
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;
#ifdef red
int erzhihua_yuzhi=0;//二值化阈值
int pengzhang_cishu=2;//膨胀次数
int fushi_cishu=0;//腐蚀次数
int chazhi=100;//红蓝差值
int main(){
    VideoCapture video(0);//打开摄像头
    Mat Light_double; //创建保存图像的矩阵
    Mat Light_double_tongdao[3];//创建保存图像三通道的矩阵数组
    
    namedWindow("二值化");//创建二值化窗口
    namedWindow("形态学操作");//创建形态学窗口
    createTrackbar("二值化阈值","二值化",&erzhihua_yuzhi,255);//创建二值化阈值改变滑块
    createTrackbar("腐蚀次数","形态学操作",&fushi_cishu,15);//创建腐蚀次数改变滑块
    createTrackbar("膨胀次数","形态学操作",&pengzhang_cishu,15);//创建膨胀次数改变滑块
    createTrackbar("红蓝差值","二值化",&chazhi,255);//创建膨胀次数改变滑块

    Mat Light_double_erzhihua;//创建矩阵保存二值化后的图像
    Mat kernel=getStructuringElement(1, Size(3, 3));//生成膨胀腐蚀中的结构元素
    Mat Light_Canny;//创建矩阵保存边缘检测的图像
    Mat result;//创建矩阵保存最后显示追踪灯条的效果
    while(1){
        video>>Light_double;//摄像头图片输入到Light_double矩阵中，方便后续操作
        Light_double.copyTo(result);//深拷贝原图，用于最后检测显示
        imshow("原图",Light_double);//显示原图
        split(Light_double,Light_double_tongdao);//分离图像的BGR三通道

        //遍历图像所有像素值,当图像红蓝差值小于设定时，保留像素值，否则设为0改成黑色
        for(int i=0;i<Light_double.rows;i++){
            for(int j=0;j<Light_double.cols;j++){
                if(Light_double_tongdao[2].at<uchar>(i,j)-Light_double_tongdao[0].at<uchar>(i,j)<chazhi){
                    Light_double_tongdao[0].at<uchar>(i,j)=0;
                    Light_double_tongdao[1].at<uchar>(i,j)=0;
                    Light_double_tongdao[2].at<uchar>(i,j)=0;
                }
            }
        }
        
        merge(Light_double_tongdao,3,Light_double);//合并图像BGR三通道
        imshow("保留红色像素",Light_double);//显示只留红色像素的图片

        GaussianBlur(Light_double, Light_double, Size(9, 9), 10, 20);//高斯滤波，过滤小噪点

        cvtColor(Light_double,Light_double,COLOR_BGR2GRAY);//图像由BGR图像转为灰度图像
        imshow("灰度图",Light_double);//显示灰度图

        threshold(Light_double,Light_double_erzhihua,erzhihua_yuzhi,255,THRESH_BINARY);//将图像二值化
        imshow("二值化",Light_double_erzhihua);//显示二值化的图像

        Canny(Light_double_erzhihua,Light_Canny,100,300,3);//边缘检测图像
        imshow("边缘检测",Light_Canny);//显示二值化的图像

        erode(Light_Canny,Light_Canny,kernel,Point(-1,-1),fushi_cishu);//图像腐蚀操作
        dilate(Light_Canny,Light_Canny,kernel,Point(-1,-1),pengzhang_cishu);//图像膨胀操作
        imshow("形态学操作",Light_Canny);//显示形态学操作后的图像

        vector<vector<Point>> contours;//创建vector<vector<Point>>型变量保存目标检测轮廓
        findContours(Light_Canny, contours, 0, 2, Point());//检测目标轮廓
        Point2f cpt;//第一个矩形的中心点
        Point2f cpt1;//第二个矩形的中心点

        //画出包裹灯条的最小矩形，下面的if...else...判断用于检测没有相应颜色灯条时直接显示结果，从而防止程序卡住
        if(contours.size()!=0){
            for(int n=0;n<contours.size();n++){
                    RotatedRect rrect = minAreaRect(contours[n]);//创建RotatedRect变量保存矩形信息
                    Point2f points[4];//创建Point2f变量保存包裹灯条最小矩阵的四个角的坐标
                    cpt1=cpt;//cpt1保存上一个cpt矩形
                    rrect.points(points); //读取最小外接矩形的 4 个顶点
                    cpt = rrect.center; //最小外接矩形的中心
                    
                    // 绘制矩形与装甲板中心位置
                    for (int i = 0; i < 4; i++)
                    {
                        if (i == 3){
                            line(result,points[i],points[0],Scalar(0,0,255),2,8);
                            break;
                            }//最后一个点连接开始的点，从而形成封闭矩形
                        
                        line(result,points[i],points[i+1],Scalar(0,0,255),2,8);//描矩形四角的点连线画矩形

                       //去除第一次算出的装甲板中心点，因为此处原理是用上一次第一个矩形的点cpt1加这次第二个矩形的点cpt求平均值得出装甲板
                       //中心坐标，第一次算的时候，cpt1值为0，导致第一次算不准，所以去掉
                        if(cpt1.x!=0&&cpt1.y!=0){
                        circle(result,Point2f((cpt.x+cpt1.x)/2,(cpt.y+cpt1.y)/2),6,Scalar(0,0,255),-1);}                         
                    }
                    imshow("结果",result);//显示最后检测结果
                }
        }else {imshow("结果",result);}//没有对应颜色的灯条时直接显示原图像，防止程序堵塞
        
        if(waitKey(10)=='q'){
            destroyAllWindows();
            return 0;
        }//按键操作，按Q键退出
        
    }
}
#endif

#ifdef blue
int erzhihua_yuzhi=0;//二值化阈值
int pengzhang_cishu=2;//膨胀次数
int fushi_cishu=0;//腐蚀次数
int chazhi=130;//红蓝差值
int main(){
    VideoCapture video(0);//打开摄像头
    Mat Light_double; //创建保存图像的矩阵
    Mat Light_double_tongdao[3];//创建保存图像三通道的矩阵数组
    
    namedWindow("二值化");//创建二值化窗口
    namedWindow("形态学操作");//创建形态学窗口
    createTrackbar("二值化阈值","二值化",&erzhihua_yuzhi,255);//创建二值化阈值改变滑块
    createTrackbar("腐蚀次数","形态学操作",&fushi_cishu,15);//创建腐蚀次数改变滑块
    createTrackbar("膨胀次数","形态学操作",&pengzhang_cishu,15);//创建膨胀次数改变滑块
    createTrackbar("红蓝差值","二值化",&chazhi,255);//创建膨胀次数改变滑块

    Mat Light_double_erzhihua;//创建矩阵保存二值化后的图像
    Mat kernel=getStructuringElement(1, Size(3, 3));//生成膨胀腐蚀中的结构元素
    Mat Light_Canny;//创建矩阵保存边缘检测的图像
    Mat result;//创建矩阵保存最后显示追踪灯条的效果
    while(1){
        video>>Light_double;//摄像头图片输入到Light_double矩阵中，方便后续操作
        Light_double.copyTo(result);//深拷贝原图，用于最后检测显示
        imshow("原图",Light_double);//显示原图
        split(Light_double,Light_double_tongdao);//分离图像的BGR三通道

        //遍历图像所有像素值,当图像红蓝差值小于设定时，保留像素值，否则设为0改成黑色
        for(int i=0;i<Light_double.rows;i++){
            for(int j=0;j<Light_double.cols;j++){
                if(Light_double_tongdao[0].at<uchar>(i,j)-Light_double_tongdao[2].at<uchar>(i,j)<chazhi){
                    Light_double_tongdao[0].at<uchar>(i,j)=0;
                    Light_double_tongdao[1].at<uchar>(i,j)=0;
                    Light_double_tongdao[2].at<uchar>(i,j)=0;
                }
            }
        }
        
        merge(Light_double_tongdao,3,Light_double);//合并图像BGR三通道
        imshow("保留蓝色像素",Light_double);//显示只留红色像素的图片

        GaussianBlur(Light_double, Light_double, Size(9, 9), 10, 20);//高斯滤波，过滤小噪点

        cvtColor(Light_double,Light_double,COLOR_BGR2GRAY);//图像由BGR图像转为灰度图像
        imshow("灰度图",Light_double);//显示灰度图

        threshold(Light_double,Light_double_erzhihua,erzhihua_yuzhi,255,THRESH_BINARY);//将图像二值化
        imshow("二值化",Light_double_erzhihua);//显示二值化的图像

        Canny(Light_double_erzhihua,Light_Canny,100,300,3);//边缘检测图像
        imshow("边缘检测",Light_Canny);//显示二值化的图像

        erode(Light_Canny,Light_Canny,kernel,Point(-1,-1),fushi_cishu);//图像腐蚀操作
        dilate(Light_Canny,Light_Canny,kernel,Point(-1,-1),pengzhang_cishu);//图像膨胀操作
        imshow("形态学操作",Light_Canny);//显示形态学操作后的图像

        vector<vector<Point>> contours;//创建vector<vector<Point>>型变量保存目标检测轮廓
        findContours(Light_Canny, contours, 0, 2, Point());//检测目标轮廓
        Point2f cpt;//第一个矩形的中心点
        Point2f cpt1;//第二个矩形的中心点

        //画出包裹灯条的最小矩形，下面的if...else...判断用于检测没有相应颜色灯条时直接显示结果，从而防止程序卡住
        if(contours.size()!=0){
            for(int n=0;n<contours.size();n++){
                    RotatedRect rrect = minAreaRect(contours[n]);//创建RotatedRect变量保存矩形信息
                    Point2f points[4];//创建Point2f变量保存包裹灯条最小矩阵的四个角的坐标
                    cpt1=cpt;//cpt1保存上一个cpt矩形
                    rrect.points(points); //读取最小外接矩形的 4 个顶点
                    cpt = rrect.center; //最小外接矩形的中心
                    
                    // 绘制矩形与装甲板中心位置
                    for (int i = 0; i < 4; i++)
                    {
                        if (i == 3){
                            line(result,points[i],points[0],Scalar(0,0,255),2,8);
                            break;
                            }//最后一个点连接开始的点，从而形成封闭矩形
                        
                        line(result,points[i],points[i+1],Scalar(0,0,255),2,8);//描矩形四角的点连线画矩形

                       //去除第一次算出的装甲板中心点，因为此处原理是用上一次第一个矩形的点cpt1加这次第二个矩形的点cpt求平均值得出装甲板
                       //中心坐标，第一次算的时候，cpt1值为0，导致第一次算不准，所以去掉
                        if(cpt1.x!=0&&cpt1.y!=0){
                        circle(result,Point2f((cpt.x+cpt1.x)/2,(cpt.y+cpt1.y)/2),6,Scalar(0,0,255),-1);}                         
                    }
                    imshow("结果",result);//显示最后检测结果
                }
        }else {imshow("结果",result);}//没有对应颜色的灯条时直接显示原图像，防止程序堵塞
        
        if(waitKey(10)=='q'){
            destroyAllWindows();
            return 0;
        }//按键操作，按Q键退出
        
    }
}
#endif
