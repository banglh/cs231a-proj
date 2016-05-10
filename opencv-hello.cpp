#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
  String path = "test.png";
  Mat img = imread(path, CV_LOAD_IMAGE_COLOR);
  imshow("test2", img);
  waitKey(0);
  return 0;
}
