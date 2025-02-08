#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

enum EnermyColor{
    RED,BLUE
};
EnermyColor energyColor = RED;

char RoboType[5][5] = {"1", "2", "3", "4", "SB"}; // 1-4号、哨兵。
string dataDir = "./dataset/", dataId = "1"; // 定义数据集目录和采集数据时指定的装甲板编号
bool getdata = false;
Net net = readNetFromONNX("./digit.onnx"); // 加载onnx模型

Mat src, srcImg;
int color_threshold = 105, count1 = 2, id_count = 0; // 颜色相减阈值，开闭次数，数据采集计数
VideoCapture cap(0);


class LightBar { // 灯条类
public:
    RotatedRect rect;
    Point center;
    double angle;
    LightBar(RotatedRect rect){
        this->rect = rect;
        center = rect.center;
        angle = rect.angle;
    }
    void findLights(Mat srcImg_binary, vector<LightBar>& lights);
};

/**
 * 识别装甲板的函数
 * @param Armor 输入的装甲板图像
 * @return 识别出的装甲板编号，如果识别失败则返回-1
 */
int Recognize(Mat Armor) {
    Mat ArmorImage;
    cvtColor(Armor, ArmorImage, COLOR_BGR2GRAY);
    if (ArmorImage.empty()) {
        cerr << "加载图片失败" << endl;
        return -1;
    }
    cout << "图像类型:" << ArmorImage.type() << endl;
    imshow("截取装甲板", ArmorImage);

    resize(ArmorImage, ArmorImage, Size(20, 20));

    if(getdata){
        if(id_count<=1511){
            Mat id_number = ArmorImage.clone();
            imwrite(dataDir + dataId + to_string(id_count++) + ".png", id_number);
            cout << "已保存装甲板数据" << endl;
        }
    }

    Mat inputBlob;
    ArmorImage.convertTo(inputBlob, CV_32F, 1.0 / 255.0);
    Scalar mean = Scalar(0.5);
    Scalar std = Scalar(0.5);
    inputBlob = (inputBlob - mean[0]) / std[0];

    inputBlob = dnn::blobFromImage(inputBlob, 1.0, Size(20, 20), Scalar(0), true, false);

    net.setInput(inputBlob);
    Mat output = net.forward();

    for (int i = 0; i < output.total(); i++) {
        cout << output.at<float>(i) << " ";
    }
    cout << endl;

    Point classIdPoint;
    double confidence;
    minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);
    int predictedClass = classIdPoint.x;
    cout << "装甲板编号: " << predictedClass << endl;
    return predictedClass;
}

/**
 * @brief 在图像上绘制旋转矩形
 * 
 * @param[in,out] mask 输入的图像，绘制后的结果会直接修改此图像
 * @param[in] rotatedrect 要绘制的旋转矩形
 * @param[in] color 绘制矩形的颜色
 * @param[in] thickness 矩形边框的厚度，如果为负数，则表示填充矩形
 * @param[in] lineType 线条的类型，例如8连接、4连接等
 * 
 * @details 此函数将旋转矩形的四个顶点转换为点集，然后使用OpenCV的polylines函数在图像上绘制旋转矩形。
 */
void DrawRotatedRect(Mat mask, RotatedRect &rotatedrect,const Scalar color,int thickness, int lineType){
	Point2f ps[4];
	rotatedrect.points(ps);
 
	vector<vector<Point>> tmpContours;
	vector<Point> contours;
	for (int i = 0; i != 4; i++) {
		contours.emplace_back(Point2i(ps[i]));
	}
	tmpContours.insert(tmpContours.end(), contours);
 
    polylines(mask, vector<vector<Point>>{contours}, true, color, thickness, lineType);
}

/**
 * @brief 检测图像中的灯条
 * 
 * @param[in] srcImg_binary 输入的二值化图像
 * @param[in,out] lights 存储检测到的灯条
 * 
 * @details 此函数对输入的二值化图像进行高斯模糊处理，然后查找图像中的轮廓。
 *          对于每个轮廓，如果其面积大于1000且轮廓点数量大于5，则将其拟合为椭圆，
 *          并检查椭圆的角度是否在45度到135度之间。如果角度不在此范围内，则将其视为灯条，
 *          并将其添加到lights向量中。最后，对lights向量进行排序，以便后续处理。
 */
void findLights(Mat srcImg_binary, vector<LightBar>& lights) {
    vector<vector<Point>> lightContours;

    GaussianBlur(srcImg_binary, srcImg_binary, Size(5, 5), 0);
    findContours(srcImg_binary, lightContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (const auto& lightContour : lightContours) {
        if (lightContour.size() > 5 && contourArea(lightContour) > 1000) {
            RotatedRect lightRect = fitEllipse(lightContour);
            if (abs(lightRect.angle) > 45.0 && abs(lightRect.angle) < 135.0) continue;
            lights.emplace_back(LightBar(lightRect));
        }
    }
    if (lights.size() < 2) {
        return;
    }
    sort(lights.begin(), lights.end(), [](LightBar &a, LightBar &b) { return a.center.x < b.center.x; });
}

/**
 * @brief 设置图像并进行处理
 * 
 * @param[in,out] src 输入的原始图像，处理后会被修改
 * @param[in,out] srcImg 用于存储处理后的图像
 * @param[in] color_threshold 颜色阈值，用于图像二值化
 * @param[in] count 形态学操作的次数，默认为1
 * 
 * @details 此函数对输入的原始图像进行颜色通道相减和二值化处理，然后进行形态学闭操作，
 *          接着检测图像中的灯条，根据检测到的灯条对装甲板进行识别和绘制。
 */
void setImg(Mat &src, Mat &srcImg, int color_threshold, int count = 1) {
    src.copyTo(srcImg);
    Mat srcImg_binary = Mat::zeros(srcImg.size(), CV_8UC1);

    uchar *pdata = srcImg.data;
    uchar *qdata = srcImg_binary.data;
    int srcData = srcImg.rows * srcImg.cols;

    if(energyColor == BLUE){
        for (int i = 0; i < srcData; i++) {
            if (*(pdata) - *(pdata+2) > color_threshold) *qdata = 255;
            pdata += 3;
            qdata++;
        }
    }
    else{
        for (int i = 0; i < srcData; i++) {
            if (*(pdata+2) - *(pdata) > color_threshold) *qdata = 255;
            pdata += 3;
            qdata++;
        }
    }
    
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(srcImg_binary, srcImg_binary, MORPH_CLOSE, kernel, Point(-1, -1), count);
    vector<LightBar> lights;
    findLights(srcImg_binary, lights);

    if(lights.size()%2==0){
        Point2f p1, p2;
        for (size_t i = 0; i < lights.size(); i+=2){
            p1 = lights[i].center;
            p2 = lights[i+1].center;
            int length = abs(p1.x - p2.x);
            int width = lights[i].rect.size.height+lights[i+1].rect.size.height;
            Point2f intersection((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
            Point2f cent(intersection.x-length/2, intersection.y-width/2);

            Rect rect(cent, Size(length, width));
            if(rect.x < 0 || rect.x + rect.width > srcImg.cols || rect.y < 0 || rect.y + rect.height > srcImg.rows) {
                cerr << "绘制的装甲板超出相机边界" << endl;
                continue;
            }
            int number = Recognize(srcImg(rect));
            if(number!=-1) putText(src, RoboType[number], cent, 1, 2, Scalar(0, 0, 255), 2);

            rectangle(src, rect, Scalar(0, 255, 0), 2, 8);

            circle(src, intersection, 5, Scalar(0, 0, 255), 2, 8);
        }   
    }
    for (auto& light : lights) {
        DrawRotatedRect(src, light.rect, Scalar(0, 255, 0), 2, 8);
    }
}

void callback(int, void*) {
        cap >> src;
        resize(src, src, Size(1440, 1080));
        GaussianBlur(src, src, Size(5, 5), 0);
        setImg(src, srcImg, color_threshold, count1);
        imshow("src", src);
}

int main() {
    namedWindow("src");
    createTrackbar("通道相减阈值", "src", &color_threshold, 255, callback);
    createTrackbar("开闭次数", "src", &count1, 10, callback);
    net.setPreferableBackend(0);
    net.setPreferableTarget(0);

    cout << "设置敌方颜色: 1.RED  2.BLUE" << endl;
    int color;
    cin >> color;
    if(color == 2){
        energyColor = BLUE;
        cout << "Set BLUE" << endl;
    }
    else cout << "Set RED" << endl;

    while(1){
        cap >> src;
        resize(src, src, Size(1440, 1080));
        if (src.empty()) {
            cout << "Cap error." << endl;
            break;
        }
        GaussianBlur(src, src, Size(5, 5), 0);
        setImg(src, srcImg, color_threshold);
        imshow("src", src);
        if (waitKey(30) == 27) {
            break;
        }
    }
    destroyAllWindows();
    cap.release();
    return 0;
}


