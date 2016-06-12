/*
 * Copyright (c) 2015 OpenALPR Technology, Inc.
 * Open source Automated License Plate Recognition [http://www.openalpr.com]
 *
 * This file is part of OpenALPR.
 *
 * OpenALPR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License
 * version 3 as published by the Free Software Foundation
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <cstdio>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "tclap/CmdLine.h"
#include "support/filesystem.h"
#include "support/timing.h"
#include "support/platform.h"
#include "video/videobuffer.h"
#include "motiondetector.h"
#include "alpr.h"

using namespace alpr;

const std::string MAIN_WINDOW_NAME = "ALPR main window";

const bool SAVE_LAST_VIDEO_STILL = false;
const std::string LAST_VIDEO_STILL_LOCATION = "/tmp/laststill.jpg";
MotionDetector motiondetector;
bool do_motiondetection = true;

/** Function Headers */
bool detectandshow(Alpr* alpr, cv::Mat frame, std::string region, bool writeJson, std::vector<int> regionCoords);
bool is_supported_image(std::string image_file);

bool measureProcessingTime = false;
std::string templatePattern;

// This boolean is set to false when the user hits terminates (e.g., CTRL+C )
// so we can end infinite loops for things like video processing.
bool program_active = true;

int main( int argc, const char** argv )
{
  std::vector<std::string> filenames;
  std::vector<int> regionCoords; // 1/6/2016 adt, define box coordinates for scan
  std::string configFile = "";
  bool outputJson = false;
  int seektoms = 0;
  bool detectRegion = false;
  std::string country;
  int topn;
  bool debug_mode = false;

  TCLAP::CmdLine cmd("OpenAlpr Command Line Utility", ' ', Alpr::getVersion());

  TCLAP::UnlabeledMultiArg<std::string>  fileArg( "image_file", "Image containing license plates", true, "", "image_file_path"  );
  TCLAP::ValueArg<std::string> regionArg("r","region","Region of image to scan for license plates with (0,0) in top left. Region will be reduced to remain in image.  Format (parantheses required): \"x y width height\". Default=\"0 0 imageWidth imageHeight\"", false, "","\"x y width height\"");


  TCLAP::ValueArg<std::string> countryCodeArg("c","country","Country code to identify (either us for USA or eu for Europe).  Default=us",false, "us" ,"country_code");
  TCLAP::ValueArg<int> seekToMsArg("","seek","Seek to the specified millisecond in a video file. Default=0",false, 0 ,"integer_ms");
  TCLAP::ValueArg<std::string> configFileArg("","config","Path to the openalpr.conf file",false, "" ,"config_file");
  TCLAP::ValueArg<std::string> templatePatternArg("p","pattern","Attempt to match the plate number against a plate pattern (e.g., md for Maryland, ca for California)",false, "" ,"pattern code");
  TCLAP::ValueArg<int> topNArg("n","topn","Max number of possible plate numbers to return.  Default=10",false, 10 ,"topN");

  TCLAP::SwitchArg jsonSwitch("j","json","Output recognition results in JSON format.  Default=off", cmd, false);
  TCLAP::SwitchArg debugSwitch("","debug","Enable debug output.  Default=off", cmd, false);
  TCLAP::SwitchArg detectRegionSwitch("d","detect_region","Attempt to detect the region of the plate image.  [Experimental]  Default=off", cmd, false);
  TCLAP::SwitchArg clockSwitch("","clock","Measure/print the total time to process image and all plates.  Default=off", cmd, false);
  TCLAP::SwitchArg motiondetect("", "motion", "Use motion detection on video file or stream.  Default=off", cmd, false);

  try
  {
    cmd.add( templatePatternArg );
    cmd.add( seekToMsArg );
    cmd.add( topNArg );
    cmd.add( configFileArg );
    cmd.add( fileArg );
    cmd.add( countryCodeArg );
    cmd.add( regionArg); // 1/6/2016 adt, added to allow region for scan from commandline

    if (cmd.parse( argc, argv ) == false)
    {
      // Error occurred while parsing.  Exit now.
      return 1;
    }

    filenames = fileArg.getValue();

    country = countryCodeArg.getValue();
    seektoms = seekToMsArg.getValue();
    outputJson = jsonSwitch.getValue();
    debug_mode = debugSwitch.getValue();
    configFile = configFileArg.getValue();
    detectRegion = detectRegionSwitch.getValue();
    templatePattern = templatePatternArg.getValue();
    topn = topNArg.getValue();
    measureProcessingTime = clockSwitch.getValue();
	  do_motiondetection = motiondetect.getValue();
    // 1/6/2016 adt, parse regionArg string into regionCoords vector
    std::istringstream iss(regionArg.getValue()); // 1/6/2016 adt, added to allow region for scan from commandline
    int n;
    while (iss >> n) {
        regionCoords.push_back(n);
    }
  }
  catch (TCLAP::ArgException &e)    // catch any exceptions
  {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    return 1;
  }

  
  cv::Mat frame;

  Alpr alpr(country, configFile);
  alpr.setTopN(topn);
  
  if (debug_mode)
  {
    alpr.getConfig()->setDebug(true);
  }

  if (detectRegion)
    alpr.setDetectRegion(detectRegion);

  if (templatePattern.empty() == false)
    alpr.setDefaultRegion(templatePattern);

  if (alpr.isLoaded() == false)
  {
    std::cerr << "Error loading OpenALPR" << std::endl;
    return 1;
  }

  for (unsigned int i = 0; i < filenames.size(); i++)
  {
    std::string filename = filenames[i];

    if (filename == "stdin")
    {
      std::string filename;
      while (std::getline(std::cin, filename))
      {
        if (fileExists(filename.c_str()))
        {
          frame = cv::imread(filename);
          detectandshow(&alpr, frame, "", outputJson, regionCoords); // 1/6/2016 adt, modified to allow region for scan from commandline
        }
        else
        {
          std::cerr << "Image file not found: " << filename << std::endl;
        }

      }
    }
    else if (filename == "webcam")
    {
      int framenum = 0;
      cv::VideoCapture cap(0);
      if (!cap.isOpened())
      {
        std::cout << "Error opening webcam" << std::endl;
        return 1;
      }

      while (cap.read(frame))
      {
        if (framenum == 0)
          motiondetector.ResetMotionDetection(&frame);
        detectandshow(&alpr, frame, "", outputJson, regionCoords); // 1/6/2016 adt, modified to allow region for scan from commandline
        sleep_ms(10);
        framenum++;
      }
    }
    else if (startsWith(filename, "http://") || startsWith(filename, "https://"))
    {
      int framenum = 0;

      VideoBuffer videoBuffer;

      videoBuffer.connect(filename, 5);

      cv::Mat latestFrame;

      while (program_active)
      {
        std::vector<cv::Rect> regionsOfInterest;
        int response = videoBuffer.getLatestFrame(&latestFrame, regionsOfInterest);

        if (response != -1)
        {
          if (framenum == 0)
            motiondetector.ResetMotionDetection(&latestFrame);
          detectandshow(&alpr, latestFrame, "", outputJson, regionCoords); // 1/6/2016 adt, modified to allow region for scan from commandline
        }

        // Sleep 10ms
        sleep_ms(10);
        framenum++;
      }

      videoBuffer.disconnect();

      std::cout << "Video processing ended" << std::endl;
    }
    else if (hasEndingInsensitive(filename, ".avi") || hasEndingInsensitive(filename, ".mp4") ||
                                                       hasEndingInsensitive(filename, ".webm") ||
                                                       hasEndingInsensitive(filename, ".flv") || hasEndingInsensitive(filename, ".mjpg") ||
                                                       hasEndingInsensitive(filename, ".mjpeg") ||
             hasEndingInsensitive(filename, ".mkv")
        )
    {
      if (fileExists(filename.c_str()))
      {
        int64_t framenum = 0;
        double frameTime = 0; //added 12/15/2015 adt to output video time and absolute frame
        int64_t vidFrame = 0;
        bool plate_found = false;//2016/06/03 adt, variable to determine if any plates found
        cv::VideoCapture cap = cv::VideoCapture();
        cap.open(filename);
        cap.set(CV_CAP_PROP_POS_MSEC, seektoms);

        while (cap.read(frame))
        {
          if (SAVE_LAST_VIDEO_STILL)
          {
            cv::imwrite(LAST_VIDEO_STILL_LOCATION, frame);
          }
          //Output additional video data video frame and current video time 12/15/2015 adt
          frameTime = cap.get(CV_CAP_PROP_POS_MSEC);
          vidFrame = cap.get(CV_CAP_PROP_POS_FRAMES);
          alpr.setFrame(vidFrame); //2016/06/05 adt, pass in current vidFrame
          alpr.setTime(frameTime); //2016/06/05 adt, pass in current videotime
          std::cout << "Processing Frame: " << framenum << " VideoFrame: " << vidFrame << " VideoTime (ms) " << frameTime << std::endl;
          if (framenum == 0)
            motiondetector.ResetMotionDetection(&frame);
          plate_found = detectandshow(&alpr, frame, "", outputJson, regionCoords) || plate_found;
          //create a 1ms delay
          sleep_ms(1);
          framenum++;
        }
        if (plate_found){
          std::cout << alpr.platesToCSV() << std::endl;
          std::cout << alpr.groupsToCSV() << std::endl;
        }
      }
      else
      {
        std::cerr << "Video file not found: " << filename << std::endl;
      }
    }
    else if (is_supported_image(filename))
    {
      if (fileExists(filename.c_str()))
      {
        frame = cv::imread(filename);

        bool plate_found = detectandshow(&alpr, frame, "", outputJson, regionCoords);

        if (!plate_found && !outputJson)
          std::cout << "No license plates found." << std::endl;
      }
      else
      {
        std::cerr << "Image file not found: " << filename << std::endl;
      }
    }
    else if (DirectoryExists(filename.c_str()))
    {
      std::vector<std::string> files = getFilesInDir(filename.c_str());

      std::sort(files.begin(), files.end(), stringCompare);

      for (int i = 0; i < files.size(); i++)
      {
        if (is_supported_image(files[i]))
        {
          std::string fullpath = filename + "/" + files[i];
          std::cout << fullpath << std::endl;
          frame = cv::imread(fullpath.c_str());
          if (detectandshow(&alpr, frame, "", outputJson, regionCoords))
          {
            //while ((char) cv::waitKey(50) != 'c') { }
          }
          else
          {
            //cv::waitKey(50);
          }
        }
      }
    }
    else
    {
      std::cerr << "Unknown file type" << std::endl;
      return 1;
    }
  }

  return 0;
}

bool is_supported_image(std::string image_file)
{
  return (hasEndingInsensitive(image_file, ".png") || hasEndingInsensitive(image_file, ".jpg") || 
	  hasEndingInsensitive(image_file, ".tif") || hasEndingInsensitive(image_file, ".bmp") ||  
	  hasEndingInsensitive(image_file, ".jpeg") || hasEndingInsensitive(image_file, ".gif"));
}


bool detectandshow( Alpr* alpr, cv::Mat frame, std::string region, bool writeJson, std::vector<int> regionCoords)
{

  timespec startTime;
  getTimeMonotonic(&startTime);

  std::vector<AlprRegionOfInterest> regionsOfInterest;
  if (do_motiondetection)
  {
	  cv::Rect rectan = motiondetector.MotionDetect(&frame);
	  if (rectan.width>0) regionsOfInterest.push_back(AlprRegionOfInterest(rectan.x, rectan.y, rectan.width, rectan.height));
  }
  else if (regionCoords.size() >= 4)  // 1/6/2016 adt, modified to allow region for scan from commandline
  {
    //Any additional integers beyond 4 are ignored. Bounds checking handled by AlprImpl::recognizeFullDetails.
    regionsOfInterest.push_back(AlprRegionOfInterest(regionCoords[0],regionCoords[1],regionCoords[2],regionCoords[3]));
  }
  else regionsOfInterest.push_back(AlprRegionOfInterest(0, 0, frame.cols, frame.rows));
  AlprResults results;
  if (regionsOfInterest.size()>0) results = alpr->recognize(frame.data, frame.elemSize(), frame.cols, frame.rows, regionsOfInterest);

  timespec endTime;
  getTimeMonotonic(&endTime);
  double totalProcessingTime = diffclock(startTime, endTime);
  if (measureProcessingTime)
    std::cout << "Total Time to process image: " << totalProcessingTime << "ms." << std::endl;
  
  
  if (writeJson)
  {
    std::cout << alpr->toJson( results ) << std::endl;
  }
  else
  {
    for (int i = 0; i < results.plates.size(); i++)
    {
      std::cout << "plate" << i << ": " << results.plates[i].topNPlates.size() << " results";
      if (measureProcessingTime)
        std::cout << " -- Processing Time = " << results.plates[i].processing_time_ms << "ms.";
      std::cout << std::endl;

      if (results.plates[i].regionConfidence > 0)
        std::cout << "State ID: " << results.plates[i].region << " (" << results.plates[i].regionConfidence << "% confidence)" << std::endl;
      
      for (int k = 0; k < results.plates[i].topNPlates.size(); k++)
      {
        // Replace the multiline newline character with a dash
        std::string no_newline = results.plates[i].topNPlates[k].characters;
        std::replace(no_newline.begin(), no_newline.end(), '\n','-');
        
        std::cout << "    - " << no_newline << "\t confidence: " << results.plates[i].topNPlates[k].overall_confidence;
        if (templatePattern.size() > 0 || results.plates[i].regionConfidence > 0)
          std::cout << "\t pattern_match: " << results.plates[i].topNPlates[k].matches_template;
        
        std::cout << std::endl;
      }
    }
  }



  return results.plates.size() > 0;
}

