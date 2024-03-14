#include <opencv2/opencv.hpp>
#include <iostream>

/* Ahmet 13-Feb-2024

Sample program for capturing and displaying a frame, and printing the pixel
values at a given position.
To compile, copy the following lines into a file, say "compile.bat", and
save it in the directory where this file is stored. Then, run it in a Windows terminal.

  g++ -o opencv_cam opencv_cam.cpp -std=c++17^
    -I "C:\msys64\mingw64\include\opencv4"^
    -L "C:\msys64\mingw64\bin"^
    -lopencv_core-409 -lopencv_highgui-409 -lopencv_imgcodecs-409^
    -lopencv_imgproc-409 -lopencv_videoio-409
*/

int main() {
    cv::Mat frame = cv::imread("../DEMO_circle_fish_star_02.jpg",cv::IMREAD_COLOR);
    if (frame.empty()) {
        std::cerr << "Error: Could not open or find the image!\n";
        return -1;
    }
    
////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //Convert to HSV
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    // Define lower and upper bounds for the color range
    cv::Scalar lower(0, 0, 0); // Lower bound (Hue, Saturation, Value)
    cv::Scalar upper(255, 80, 255); // Upper bound (Hue, Saturation, Value)

    // Threshold the HSV image to get only specified colors
    cv::Mat mask;
    cv::inRange(hsv, lower, upper, mask);

    // Apply the mask to the original image
    cv::Mat result;
    cv::bitwise_and(frame, frame, result, mask);

    // Convert the result back to grayscale
    cv::cvtColor(result, frame, cv::COLOR_BGR2GRAY);

    // Apply thresholding to obtain binary image
    int threshold_type = cv::THRESH_BINARY + cv::THRESH_OTSU;
    cv::threshold(frame, frame, 0, 255, threshold_type);
    cv::bitwise_not(frame, frame);

    //Remove tiny blobs
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25));
    cv::morphologyEx(frame, frame, cv::MORPH_OPEN, kernel);

    //Join blobs that are close together
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30));
    cv::morphologyEx(frame, frame, cv::MORPH_CLOSE, kernel);

    // Connected component analysis
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(frame, labels, stats, centroids);

    // Assign random color to each blob
    std::vector<cv::Vec3b> colors(numComponents);
    for (int i = 0; i < numComponents; ++i) {
        colors[i] = cv::Vec3b(rand() & 255, rand() & 255, rand() & 255);
    }

    // Colorize the blobs
    cv::Mat coloredImage(frame.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    for (int y = 0; y < frame.rows; ++y) {
        for (int x = 0; x < frame.cols; ++x) {
            int label = labels.at<int>(y, x);
            if (label > 0) {
                coloredImage.at<cv::Vec3b>(y, x) = colors[label];
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //Stat calculations
    for (int label = 1; label < numComponents; ++label) {
        //Centre of mass
        double centroidX = centroids.at<double>(label, 0);
        double centroidY = centroids.at<double>(label, 1);

        cv::circle(coloredImage, cv::Point(centroidX, centroidY), 20, cv::Scalar(0, 0, 255), -1);

        //Moment calculations
        double m20 = 0.0, m02 = 0.0, m11 = 0.0;
        for (int y = 0; y < frame.rows; ++y) {
            for (int x = 0; x < frame.cols; ++x) {
                if (labels.at<int>(y, x) == label) {
                    m20 += (x - centroidX) * (x - centroidX);
                    m02 += (y - centroidY) * (y - centroidY);
                    m11 += (x - centroidX) * (y - centroidY);
                }
            }
        }

        // Compute the axis of rotation
        double theta = 0.5 * atan2(2 * m11, m20 - m02);

        // Draw the axis of rotation on the image
        cv::Point2d axis(centroidX - 150 * cos(theta), centroidY - 150 * sin(theta));
        cv::line(coloredImage, cv::Point(centroidX, centroidY), axis, cv::Scalar(0, 255, 0), 8);
    }

    //Display image
    cv::namedWindow("image", cv::WINDOW_NORMAL);
    cv::resizeWindow("image", 800, 600); 
    cv::imshow("image", coloredImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}