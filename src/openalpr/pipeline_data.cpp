#include "pipeline_data.h"

using namespace cv;
using namespace std;

namespace alpr
{

  PipelineData::PipelineData(Mat colorImage, Rect regionOfInterest, Config* config)
  {
    Mat grayImage;

    if (colorImage.channels() > 2)
    {
      cvtColor(colorImage, grayImage, CV_BGR2GRAY);
    }
    else
    {
      grayImage = colorImage;
    }

    this->init(colorImage, grayImage, expandRect(regionOfInterest, 0, 0, colorImage.cols, colorImage.rows), config); //1/26/2016 adt, modified to fix out-of-bounds regions
  }
  
  PipelineData::PipelineData(Mat colorImage, Mat grayImg, Rect regionOfInterest, Config* config)
  {
    this->init(colorImage, grayImg, expandRect(regionOfInterest, 0, 0, colorImage.cols, colorImage.rows), config); //1/26/2016 adt, modified to fix out-of-bounds regions

  }

  PipelineData::~PipelineData()
  {
    clearThresholds();
  }

  void PipelineData::clearThresholds()
  {
    for (unsigned int i = 0; i < thresholds.size(); i++)
    {
      thresholds[i].release();
    }
    thresholds.clear();
  }

  void PipelineData::init(cv::Mat colorImage, cv::Mat grayImage, cv::Rect regionOfInterest, Config *config) {
    this->colorImg = colorImage;
    this->grayImg = grayImage;
    this->regionOfInterest = regionOfInterest;
    this->config = config;
    this->region_confidence = 0;
    this->plate_inverted = false;
    this->disqualified = false;
    this->disqualify_reason = "";
  }
}
