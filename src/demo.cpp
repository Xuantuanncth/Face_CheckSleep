#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include "dlib/image_processing/shape_predictor.h"
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv.h>
#include <dlib/geometry/rectangle.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <wiringPi.h>


using namespace dlib;
using namespace std;
using namespace cv;

int rightIndex[6] = {36, 37, 38, 39, 40, 41};
int leftIndex[6] = {42, 43, 44, 45, 46, 47};

int topIndex[6] = {50, 51, 52, 61, 62, 63};
int bottomIndex[6] = {56, 57, 58, 65, 66, 67};
time_t now = time(0);
int year = 0, month = 0, day = 0, hour= 0, minute = 0;

/*22/4/2020 =================> TuanDX add =================> */

void getTime(char time[], int select);
void calibData();
void writeLog(char data);

ofstream Log;
struct face_Deafult{
    double yaw[120];
    double pitch[120];
    double roll[120];
    double s_face[120];
};
double s_face_default = 0;
int yaw_max = 0, yaw_min = 0, roll_max = 0, roll_min = 0, pitch_max = 0, pitch_min = 0;
int counter = 0;
bool index_t = false;
struct face_Deafult face_info;

/*======================================================>*/
void draw_polyline(cv::Mat &img, const dlib::full_object_detection& d, const int start, const int end, bool isClosed = false){
    std::vector <cv::Point> points;
    for (int i = start; i <= end; ++i)
    {
        circle(img,cv::Point(d.part(i).x(), d.part(i).y()),1,cv::Scalar(0,255,255), -1);
    }
    
}

void render_face (cv::Mat &img, const dlib::full_object_detection& d){
    DLIB_CASSERT
    (
     d.num_parts() == 68,
     "\n\t Invalid inputs were given to this function. "
     << "\n\t d.num_parts():  " << d.num_parts()
     );
     
    //draw_polyline(img, d, 36, 41, true);    // Left eye
    //draw_polyline(img, d, 42, 47, true);    // Right Eye
     draw_polyline(img, d,  0, 17, true);    //Face

    //draw_polyline(img, d, 48, 59, true);    // Outer lip
    //draw_polyline(img, d, 60, 67, true);    // Inner lip
     
}
/*
 * Tinh mat nham va mo
 */
float eye_aspect_ratio (const dlib::full_object_detection& d, bool isLeft) {
    int *index = rightIndex;
    if (isLeft) {
        index = leftIndex;
    }
    float A = sqrt(pow(d.part(index[1]).x() - d.part(index[5]).x(), 2) + pow(d.part(index[1]).y() - d.part(index[5]).y(), 2) * 1.0);
    float B = sqrt(pow(d.part(index[2]).x() - d.part(index[4]).x(), 2) + pow(d.part(index[2]).y() - d.part(index[4]).y(), 2) * 1.0);

    float C = sqrt(pow(d.part(index[0]).x() - d.part(index[3]).x(), 2) + pow(d.part(index[0]).y() - d.part(index[3]).y(), 2) * 1.0);

    return (A + B) / (2.0 * C);
}

void face_ratio(const dlib::full_object_detection& d) {
    // cout << "A_0(x,y) =  " << "(" << d.part(1).x() << "," << d.part(1).y()<<")"<< endl;
    // cout << "A_17(x,y) =  " << "(" << d.part(17).x() << "," << d.part(17).y()<<")"<< endl;
    // cout << "A_44(x,y) =  " << "(" << d.part(44).x() << "," << d.part(44).y()<<")"<< endl;
    // cout << "A_54(x,y) =  " << "(" << d.part(54).x() << "," << d.part(54).y()<<")"<< endl;
    float A = sqrt(pow(d.part(1).x() - d.part(17).x(), 2) + pow(d.part(1).y() - d.part(17).y(), 2) * 1.0);
    cout << "Khoang cach khuon mat: " << A << endl;
}

float lip_center (const dlib::full_object_detection& d, bool isBottom) {
    int *index = topIndex;
    int count = 0;
    if (isBottom) {
        index = bottomIndex;
    }
    for (int i = 0; i < 6; i++) {
        count += d.part(index[i]).y();
    }
    return count / 6.0;
}

void dlib_point2cv_Point(const dlib::full_object_detection& S,std::vector<cv::Point>& L,double& scale)
{
    for(unsigned int i = 0; i<S.num_parts();++i)
    {
        L.push_back(cv::Point(S.part(i).x()*(1/scale),S.part(i).y()*(1/scale)));
    }
}

void getEulerAngles(cv::Mat &rotCamerMatrix,cv::Vec3d &eulerAngles)
{
    cv::Mat cameraMatrix,rotMatrix,transVect,rotMatrixX,rotMatrixY,rotMatrixZ;
    double* _r = rotCamerMatrix.ptr<double>();
    double projMatrix[12] = {_r[0],_r[1],_r[2],0,
                            _r[3],_r[4],_r[5],0,
                            _r[6],_r[7],_r[8],0};

    decomposeProjectionMatrix(
        cv::Mat(3,4,CV_64FC1,projMatrix),
                        cameraMatrix,
                        rotMatrix,
                        transVect,
                        rotMatrixX,
                        rotMatrixY,
                        rotMatrixZ,
                        eulerAngles);
}

bool headPosEstimate(cv::Mat & faceImg, const dlib::full_object_detection& d)
{
    std::vector<cv::Point> landmarks;//,R_Eyebrow,L_Eyebrow,L_Eye,R_Eye,Mouth,Jaw_Line,Nose;

    double scale = 1;
    dlib_point2cv_Point(d,landmarks,scale);

    std::vector<cv::Point2d> image_points;
    image_points.push_back(landmarks[30]);    // Nose tip
    image_points.push_back(landmarks[8]);    // Chin
    image_points.push_back(landmarks[45]);     // Left eye left corner
    image_points.push_back(landmarks[36]);    // Right eye right corner
    image_points.push_back(landmarks[54]);    // Left Mouth corner
    image_points.push_back(landmarks[48]);    // Right mouth corner

    // 3D model points.
    std::vector<cv::Point3d> model_points;
    model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
    model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
    model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
    model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
    model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner
    model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // Right mouth corner

    // Camera internals
    double focal_length = faceImg.cols; // Approximate focal length.
    cv::Point2d center = cv::Point2d(faceImg.cols/2,faceImg.rows/2);
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
    cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion

    // Output rotation and translation
    cv::Mat rotation_vector; // Rotation in axis-angle form
    cv::Mat translation_vector;
    cv::Mat rotCamerMatrix1;

    // Solve for pose
    solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

    Rodrigues(rotation_vector,rotCamerMatrix1);
    cv::Vec3d eulerAngles;
    getEulerAngles(rotCamerMatrix1,eulerAngles);

    if (sizeof(eulerAngles) >= 1) {
        double yaw   = eulerAngles[1];
        double pitch = eulerAngles[0] + 10;
        double roll  = eulerAngles[2];
        if (roll > 90)
            roll -= 180;
        if (roll < -90)
            roll += 180;
        if(index_t) {
            face_info.yaw[counter] = yaw;
            face_info.pitch[counter] = pitch;
            face_info.roll[counter] = roll;
        }

        if((pitch_max != 0 || pitch_min != 0) && (yaw_max !=0 || yaw_min != 0) && (roll_max != 0 || roll_min != 0)) {
            if(pitch > pitch_max){
                cout << "Cui dau" << endl;
                cv::putText(faceImg, "Cui dau", cv::Point(50,70), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,0,255), 2);
            } else if(pitch < pitch_min) {
                cout << " Ngua dau " << endl;
            } else if(roll > roll_max) {
                cout << "Nghieng sang trai" << endl;
            } else if(roll < roll_min) {
                cout << "Nghieng sang phai" << endl;
            } else if(yaw > yaw_max) {
                cout << "Quay phai" << endl;
            } else if (yaw < yaw_min) {
                cout << "Quay trai" << endl;
            } 
        }
        cv::putText(faceImg, "yaw_1: " + to_string(yaw), cv::Point(10,280), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0,0,255), 1);
        cv::putText(faceImg, "pitch_1: " + to_string(pitch), cv::Point(10,300), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0,0,255), 1);
        cv::putText(faceImg, "roll_1: " + to_string(roll), cv::Point(10,320), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0,0,255), 1);
    } else {
        if(index_t) {
            face_info.yaw[counter] = 0;
            face_info.pitch[counter] = 0;
            face_info.roll[counter] = 0;
        }
    }
    return false;
}


void camera_test(int frames)
{
    cv::Mat image;
    cv::Mat im_small;
    cv::VideoCapture cap(0);

    float fps = cap.get(cv::CAP_PROP_FPS);
    int top = 0, bottom = 0, left = 0, right = 0, s_face = 0, time_in = 0,s_face_err = 0;
    char time_err[20];
    char path_file[50];
    char dateTime[20];

    if(!cap.isOpened())
        cout<<"fail to open!"<<endl;

    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 320);

    cap>>image;

    if(!image.data) {
        cout << "unable to open camera" << endl;
        return;
    }
    cout << "fps:" << fps << endl;
    cout << "cols:" << image.cols << endl;
    cout << "rows:" << image.rows << endl;
    
    float EYE_AR_THRESH = 0.3;
    int EYE_AR_CONSEC_FRAMES = 24;

    int EYE_COUNTER = 0;
    int LIP_COUNTER = 0;
    bool ALARM_ON = false;

    int stop = frames;
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    string path = "shape_predictor_68_face_landmarks.dat";
    //string str = "omxplayer /home/pi/test.mp3"; 
    
    getTime(dateTime,1);
    cout << dateTime << endl;
    sprintf(path_file,"/home/pi/logs/%s.txt",dateTime);
    Log.open(path_file,ios_base::app);

    deserialize(path.c_str()) >> sp;

    while(1){
        /*=======================> calib ==================> */
        if(digitalRead(15) == 0) {
            index_t = true;
        }
        if(index_t) {
            if(counter > 118){
                counter = 0;
                time_in = 0;
                index_t = false;
                digitalWrite(16,1);
                calibData();
            } else {
                if(time_in > 30){
                    counter ++;
                    digitalWrite(16,0);
                    cout << "Counter : " << counter << endl;
                }
                time_in++;
            }
        }
        cap>>image;

        dlib::cv_image<dlib::bgr_pixel> cimg(image);
        cv::Mat des;
        cv::Mat roi;
        std::vector<cv::Mat> rects;
        std::vector<dlib::rectangle> faces = detector(cimg);

        for(std::vector<dlib::rectangle>::iterator it=faces.begin(); it!=faces.end();it++){
            top = (*it).top();
            bottom = (*it).bottom();
            left = (*it).left();
            right = (*it).right();

            /*Ve hinh vuong khuon mat*/
            //cv::rectangle(image, Point((*it).left(), (*it).top()), Point((*it).right(), (*it).bottom()), Scalar(0,255,0), 2, 4);

            s_face =  ((right-left)*(bottom-top));
            //cout << "Dien tich khuon mat: " << s_face << endl;
            if(index_t) {
                if(s_face <= 0) {
                    face_info.s_face[counter] = 0;
                } else {
                    face_info.s_face[counter] = s_face;
                }
                cout << "Dien tich khuon mat: " << s_face << endl;
            }
            if(s_face_default != 0) {
                cout << "s_face: " << s_face << endl;
                if(s_face > (s_face_default*1.5)) {
                    cout << "S_Face: " << s_face << " s_face_default: "<< s_face_default <<endl;
                    cv::putText(image, " Ngoi sai tu the", cv::Point(50,30), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,0,255), 2); 
                    if(s_face_err == 40) {
                        getTime(time_err,2);
                        Log << time_err << " Ngoi sai tu the" << endl;    
                        cv::putText(image, " Waring !!! ", cv::Point(50,50), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,0,255), 1); 
                        s_face_err = 0;
                    }
                    s_face_err++;
                }
            }
        }
        
        std::vector<full_object_detection> shapes;

        for (unsigned long j = 0; j < faces.size(); ++j)
        {
            dlib::full_object_detection shape = sp(cimg, faces[j]);
		    float leftEAR = eye_aspect_ratio(shape, true);
		    float rightEAR = eye_aspect_ratio(shape, false);
            face_ratio(shape);
            // printf("Mat trai: %f, Mat phai: %f\n",leftEAR,rightEAR);
		    // average the eye aspect ratio together for both eyes
		    float ear = (leftEAR + rightEAR) / 2.0;

            float top_lip_center = lip_center(shape, false);
            float bottom_lip_center = lip_center(shape, true);
            float mouth_height = abs(top_lip_center - bottom_lip_center);
            float mouth_width = abs(shape.part(48).x() - shape.part(54).x());


            if (ear < EYE_AR_THRESH) {
                EYE_COUNTER++;

                if (EYE_COUNTER > EYE_AR_CONSEC_FRAMES) {

                    cv::putText(image, "WAKE UP!!!!", cv::Point(10,30), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,143,143), 2);
                }
            }
            else
            {
                EYE_COUNTER = 0;
            }
            if (mouth_height / mouth_width > 0.5) {
                LIP_COUNTER++;
                if (LIP_COUNTER > 12) {
                    cv::putText(image, "YAWNINGGGG!!!!", cv::Point(10,60), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,143,143), 2);
                }
            } else {
                LIP_COUNTER = 0;
            }
            
            // You get the idea, you can get all the face part locations if
            // you want them.  Here we just store them in shapes so we can
            // put them on the screen.
            headPosEstimate(image, shape);
            render_face(image, shape);
        }
        cv::imshow("result", image);
        cv::waitKey(1);
    }
    Log.close();
    cout<<"end"<<endl;
}

void getTime(char *time, int select) {
    tm *ltm = localtime(&now);
    year = 1900 + ltm->tm_year;
    month = 1+ ltm->tm_mon;
    day = ltm->tm_mday ;
    hour = ltm->tm_hour;
    minute = ltm-> tm_min;
    if(select == 1) {
        sprintf(time,"%d_%d_%d",year,month,day);
    } else if (select == 2) {
        sprintf(time,"%d_%d_%d_%d_%d",year,month,day,hour,minute);
    }

}

void setup_gpio(){
    cout << "Raspberry setting gpio" <<endl;
    if (wiringPiSetup () == -1){
        return; 
    }
    pinMode(15,INPUT);
    pinMode(16,OUTPUT);
    digitalWrite(16,1);
}
void calibData() {
    double sum = 0;
    int count_temp = 0;
    int chot = 0;
    sort(face_info.s_face, face_info.s_face+120);
    //sort(face_info.yaw, face_info.yaw+120);
    //sort(face_info.roll, face_info.roll+120);
    //sort(face_info.pitch, face_info.pitch+120);
    for(int i = 0; i < 120 ; i++) {
        if(face_info.s_face[i] > 0 ) {
            sum += face_info.s_face[i];
            count_temp++;
        }
    }
    cout << "count_temp:" << count_temp << endl;
    cout << "Sum: " << sum << endl;
    s_face_default = sum/count_temp;
    cout << "Face_deafaul: "<< s_face_default << endl;
}

int main()
{   
    setup_gpio();
    camera_test(1);
    return 0;
}
