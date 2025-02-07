#include<opencv2/opencv.hpp>
#include<iostream>
#include<cstring>
#include<math.h>
#include<vector>

using namespace cv;
using namespace std;

Mat dst;

string PIC_PATH = "/home/find/code/cpp/decter_armor/imgs/";

int hmin = 0,hmax = 255,smin = 0,smax = 179,vmin = 194,vmax = 255;
int open = 9,close = 3,er = 3,di = 3;

float WIDTH = 780,HEIGH = 540;
float WIDTH_HEIGHT_RATIO = 1.5;
float AREA = WIDTH * HEIGH;

float WIDTH_HEIGHT_RATIO_MAX = 10;
float WIDTH_HEIGHT_RATIO_MIN = 0.2;

float ANGLE = 90;
float ANGLE_ABS = 45;

float AREA_MAX = 0.01 * AREA;
float AREA_MIN = 0.0005 * AREA;    // 210 < area < 421

float PLATE_WIDTH_HIGH_RATIO = 0.5;

void nothing(int,void*){}

void do_create_all_trackbar(){

    createTrackbar("hmin", "color_adjust", &hmin, 255, nothing);
    createTrackbar("hmax", "color_adjust", &hmax, 255, nothing);
    createTrackbar("smin", "color_adjust", &smin, 255, nothing);
    createTrackbar("smax", "color_adjust", &smax, 255, nothing);
    createTrackbar("vmin", "color_adjust", &vmin, 255, nothing);
    createTrackbar("vmax", "color_adjust", &vmax, 255, nothing);

    createTrackbar("open", "mor_adjust", &open, 30, nothing);
    createTrackbar("close", "mor_adjust", &close, 30, nothing);
    createTrackbar("erode", "mor_adjust", &er, 30, nothing);
    createTrackbar("dilate", "mor_adjust", &di, 30, nothing);
}
Mat do_mask_color(Mat color_adjust_resized){
    Mat lower_hsv = (Mat_<uchar>(1,3)<<hmin,smin,vmin);
    Mat upper_hsv = (Mat_<uchar>(1,3)<<hmax,smax,vmax);
    inRange(color_adjust_resized, lower_hsv, upper_hsv,dst);
    return dst;
}

Mat open_binary(Mat binary, int x, int y){
    Mat kernel = getStructuringElement(MORPH_RECT, Size(x,y));
    morphologyEx(binary, dst, MORPH_OPEN, kernel);
    return dst;
}
Mat close_binary(Mat binary, int x, int y){
    Mat kernel = getStructuringElement(MORPH_RECT, Size(x,y));
    morphologyEx(binary, dst, MORPH_CLOSE, kernel);
    return dst;
}
Mat erode_binary(Mat binary, int x, int y){
    Mat kernel = getStructuringElement(MORPH_RECT, Size(x,y));
    erode(binary, dst, kernel);
    return dst;
}
Mat dilate_binary(Mat binary, int x, int y){
    Mat kernel = getStructuringElement(MORPH_RECT, Size(x,y));
    dilate(binary, dst, kernel);
    return dst;
}
Mat do_mask_mor(Mat binary){
    open_binary(binary, open, open);
    close_binary(binary, close, close);
    erode_binary(binary, er, er);
    dilate_binary(binary, di, di);
    return dst;
}

int main(){
    
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout<< "无法打开摄像头" << endl;
        return -1;
    }
    
    /**********截图，调参**********/
    int i = -1;
    while (1){
        Mat frame;
        cap>>frame;
        if(frame.empty()){
            cout<<"return -1"<<endl;
            return -1;
        }
        imshow("video",frame);
        int k = waitKey(1);
        if (k == 27){
            destroyAllWindows();
            break;
            }
        else if (k == 's'){
            imwrite(PIC_PATH+to_string(++i)+".jpg",frame);
            }
    }
    Mat img = imread(PIC_PATH + to_string(i) + ".jpg");
    //Mat img = imread(PIC_PATH + "3.jpg");//////////////////////////////////////////////////
    //Mat img = imread("/home/find/code/cpp/测评 24-12-28/images/test1.png");

    Mat color_adjust_resized;
    cvtColor(img, color_adjust_resized, COLOR_BGR2HSV);
    resize(color_adjust_resized, color_adjust_resized, Size(WIDTH,HEIGH), INTER_CUBIC);
    namedWindow("color_adjust");
    namedWindow("mor_adjust");
    do_create_all_trackbar();
    while (1){
        do_mask_color(color_adjust_resized);
        imshow("dst_color", dst);
        do_mask_mor(dst);
        imshow("dst_mor", dst);
        int k = waitKey(1);
        if (k == 27){
            destroyAllWindows();
            break;
            }
    }

/*********************识别主函数********************/
   while (1){
        double t1 = getTickCount();
        Mat frame;
        cap>> frame;
        flip(frame,frame,1);
        //frame = imread(PIC_PATH + to_string(3) + ".jpg");/////////////////////////////
        resize(frame,frame, Size(WIDTH,HEIGH));

        Mat color_adjust_resized;
        Mat frame_copy(frame);
        cvtColor(frame_copy, color_adjust_resized, COLOR_BGR2HSV);
        do_mask_color(color_adjust_resized);
        do_mask_mor(dst);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        vector<Point2f>box_centers;
        findContours(dst, contours, hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE);

        for (size_t i = 0; i < contours.size(); i++){
            RotatedRect rect = minAreaRect(contours[i]);
            Point2f box_center = rect.center;
            float width = rect.size.width;
            float height = rect.size.height;
            float angle = rect.angle;
            float width_height_ratio = width / height;
            float area = width * height;
            
            cout<< width_height_ratio<< endl;
            cout<< angle<< endl;
            cout<< area<< endl;
            cout<< ""<< endl;

            if (WIDTH_HEIGHT_RATIO_MIN < width_height_ratio && width_height_ratio< WIDTH_HEIGHT_RATIO_MAX 
                && (abs(angle - ANGLE) < 20 || abs(angle - ANGLE) > 70) 
                && AREA_MIN < area && area < AREA_MAX)
            {
                box_centers.push_back(box_center);
            }
        }

        if (box_centers.size() == 2){
            /***********打击中心***************/
            Point2f center;
            center.x = (box_centers[0].x + box_centers[1].x) / 2;
            center.y = (box_centers[0].y + box_centers[1].y) / 2;
            circle(frame, center, 10,Scalar(0, 0, 255));

            /***********绘制装甲板*************/
            int plate_x1 = box_centers[0].x;
            int plate_x2 = box_centers[1].x;
            int PLATE_LONG = abs(plate_x1 - plate_x2);

            int plate_y1 = center.y - (PLATE_WIDTH_HIGH_RATIO * PLATE_LONG / 2);
            int plate_y2 = center.y + (PLATE_WIDTH_HIGH_RATIO * PLATE_LONG / 2);

            Point plate_0(plate_x1, plate_y1);
            Point plate_1(plate_x2, plate_y1);
            Point plate_2(plate_x2, plate_y2);
            Point plate_3(plate_x1, plate_y2);

            line(frame_copy, plate_0, plate_1, Scalar(255, 0, 0), 2);
            line(frame_copy, plate_1, plate_2, Scalar(255, 0, 0), 2);
            line(frame_copy, plate_2, plate_3, Scalar(255, 0, 0), 2);
            line(frame_copy, plate_3, plate_0, Scalar(255, 0, 0), 2);
        }
        else{
            //cout<< box_centers.size()<< endl;
            //cout<< contours.size()<< endl;
            //return -1;
        }
        imshow("dst",dst);
        imshow("frame",frame);
        double t2 = getTickCount();
        double time = (t1 - t2) /  getTickFrequency();
        cout<< time<<endl;
        int k = waitKey(1);
        if (k == 27){
            cap.release();
            destroyAllWindows();
            break;
        }
    }
    return 0;
}