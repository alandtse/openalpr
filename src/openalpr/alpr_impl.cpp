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

#include "alpr_impl.h"
#include "result_aggregator.h"


void plateAnalysisThread(void* arg);

using namespace std;
using namespace cv;

namespace alpr
{
  AlprImpl::AlprImpl(const std::string country, const std::string configFile, const std::string runtimeDir)
  {

    timespec startTime;
    getTimeMonotonic(&startTime);

    config = new Config(country, configFile, runtimeDir);

    prewarp = ALPR_NULL_PTR;


    // Config file or runtime dir not found.  Don't process any further.
    if (config->loaded == false)
    {
      return;
    }

    for (unsigned int i = 0; i < config->loaded_countries.size(); i++)
    {
      config->setCountry(config->loaded_countries[i]);

      AlprRecognizers recognizer;
      recognizer.plateDetector = createDetector(config);
      recognizer.ocr = new OCR(config);

      recognizer.stateDetector = new StateDetector(this->config->country, this->config->config_file_path, this->config->runtimeBaseDir);

      recognizers[config->country] = recognizer;

    }

    setNumThreads(0);

    setDetectRegion(DEFAULT_DETECT_REGION);
    this->topN = DEFAULT_TOPN;
    setDefaultRegion("");

    prewarp = new PreWarp(config);
    frame_number = 0; //1/24/2016, adt adding frame_number for historical measuring
    timespec endTime;
    getTimeMonotonic(&endTime);
    if (config->debugTiming)
      cout << "OpenALPR Initialization Time: " << diffclock(startTime, endTime) << "ms." << endl;

  }

  AlprImpl::~AlprImpl()
  {
    delete config;

    typedef std::map<std::string, AlprRecognizers>::iterator it_type;
    for(it_type iterator = recognizers.begin(); iterator != recognizers.end(); iterator++) {

      delete iterator->second.plateDetector;
      delete iterator->second.stateDetector;
      delete iterator->second.ocr;
    }

    delete prewarp;
  }

  bool AlprImpl::isLoaded()
  {
    return config->loaded;
  }


  AlprFullDetails AlprImpl::recognizeFullDetails(cv::Mat img, std::vector<cv::Rect> regionsOfInterest)
  {
    timespec startTime;
    getTimeMonotonic(&startTime);


    AlprFullDetails response;

    int64_t start_time = getEpochTimeMs();

    //Stop processing early if no image
    if (!img.data)
    {
      // Invalid image
      if (this->config->debugGeneral)
        std::cerr << "Invalid image" << std::endl;

      return response;
    }

    for (unsigned int i = 0; i < regionsOfInterest.size(); i++)
    {
      // Fix regions of interest in case they extend beyond the bounds of the image
      regionsOfInterest[i] = expandRect(regionsOfInterest[i], 0, 0, img.cols, img.rows);
      response.results.regionsOfInterest.push_back(AlprRegionOfInterest(regionsOfInterest[i].x, regionsOfInterest[i].y,
              regionsOfInterest[i].width, regionsOfInterest[i].height));
    }

    // Convert image to grayscale if required
    Mat grayImg = img;
    if (img.channels() > 2)
      cvtColor( img, grayImg, CV_BGR2GRAY );

    // Prewarp the image and ROIs if configured]
    std::vector<cv::Rect> warpedRegionsOfInterest = regionsOfInterest;
    // Warp the image if prewarp is provided
    grayImg = prewarp->warpImage(grayImg);
    warpedRegionsOfInterest = prewarp->projectRects(regionsOfInterest, grayImg.cols, grayImg.rows, false);

    // Iterate through each country provided (typically just one)
    // and aggregate the results if necessary
    ResultAggregator aggregator;
    for (unsigned int i = 0; i < config->loaded_countries.size(); i++)
    {
      if (config->debugGeneral)
        cout << "Analyzing: " << config->loaded_countries[i] << " frame_number: " << frame_number << endl;

      config->setCountry(config->loaded_countries[i]);
      AlprFullDetails sub_results = analyzeSingleCountry(img, grayImg, warpedRegionsOfInterest);

      sub_results.results.epoch_time = start_time;
      sub_results.results.img_width = img.cols;
      sub_results.results.img_height = img.rows;

      aggregator.addResults(sub_results);
    }
    response = aggregator.getAggregateResults();
    //1/23/2016 adt, save response into priorResults
    if (!response.results.plates.empty())
      priorResults.push_back(response.results);
    frame_number++; //1/24/2016, adt increment frame_number after processing

    timespec endTime;
    getTimeMonotonic(&endTime);
    if (config->debugTiming)
    {
      cout << "Total Time to process image: " << diffclock(startTime, endTime) << "ms." << endl;
    }

    if (config->debugGeneral && config->debugShowImages)
    {
      for (unsigned int i = 0; i < regionsOfInterest.size(); i++)
      {
        rectangle(img, regionsOfInterest[i], Scalar(0,255,0), 2);
      }

      for (unsigned int i = 0; i < response.plateRegions.size(); i++)
      {
        rectangle(img, response.plateRegions[i].rect, Scalar(0, 0, 255), 2);
      }

      for (unsigned int i = 0; i < response.results.plates.size(); i++)
      {
        // Draw a box around the license plate
        for (int z = 0; z < 4; z++)
        {
          AlprCoordinate* coords = response.results.plates[i].plate_points;
          Point p1(coords[z].x, coords[z].y);
          Point p2(coords[(z + 1) % 4].x, coords[(z + 1) % 4].y);
          line(img, p1, p2, Scalar(255,0,255), 2);
        }

        // Draw the individual character boxes
        for (int q = 0; q < response.results.plates[i].bestPlate.character_details.size(); q++)
        {
          AlprChar details = response.results.plates[i].bestPlate.character_details[q];
          line(img, Point(details.corners[0].x, details.corners[0].y), Point(details.corners[1].x, details.corners[1].y), Scalar(0,255,0), 1);
          line(img, Point(details.corners[1].x, details.corners[1].y), Point(details.corners[2].x, details.corners[2].y), Scalar(0,255,0), 1);
          line(img, Point(details.corners[2].x, details.corners[2].y), Point(details.corners[3].x, details.corners[3].y), Scalar(0,255,0), 1);
          line(img, Point(details.corners[3].x, details.corners[3].y), Point(details.corners[0].x, details.corners[0].y), Scalar(0,255,0), 1);
        }
      }


      displayImage(config, "Main Image", img);

      // Sleep 1ms
      sleep_ms(1);

    }
    // Pause if response includes any plates; works best with debugPostProcess enabled to view results
    if (config->debugPauseOnPlates && response.results.plates.size() > 0){
      config->debugPauseOnFrame = true;
    }
    char keypress;
    if (config->debugPauseOnFrame)
    {
      // Pause indefinitely until they press a key
      while ((char) (keypress = cv::waitKey(50)) == -1)
      {
      }
      // Add keypress processing 11/7/2015 adt
      // Adding continue option
      switch (keypress) {
        case 'c':
        cout << "Continue pressed " << keypress << endl;
        config->debugPauseOnFrame = false;
        break;
        default:
        cout << "Press \"c\" to continue" << endl;
        break;
      }
    }

    return response;
  }

  AlprFullDetails AlprImpl::analyzeSingleCountry(cv::Mat colorImg, cv::Mat grayImg, std::vector<cv::Rect> warpedRegionsOfInterest)
  {
    AlprFullDetails response;
    response.results.frame_number = frame_number;  //1/24/2016, adt update AlprResult frame_number

    AlprRecognizers country_recognizers = recognizers[config->country];
    timespec startTime;
    getTimeMonotonic(&startTime);
    //1/23/2016 adt, adding historical data check.  If priorResults is not empty and image size is the same
    bool usePriorResults = (!priorResults.empty() && (priorResults.back().img_width == colorImg.cols && priorResults.back().img_height == colorImg.rows));
    if (config->debugGeneral) cout << "usePriorResults: " << usePriorResults << " size: " << priorResults.size() << endl;
    
    vector<PlateRegion> warpedPlateRegions;
    // Find all the candidate regions
    if (config->skipDetection == false)
    {
      warpedPlateRegions = country_recognizers.plateDetector->detect(grayImg, warpedRegionsOfInterest);
    }
    else
    {
      // They have elected to skip plate detection.  Instead, return a list of plate regions
      // based on their regions of interest
      for (unsigned int i = 0; i < warpedRegionsOfInterest.size(); i++)
      {
        PlateRegion pr;
        pr.rect = cv::Rect(warpedRegionsOfInterest[i]);
        warpedPlateRegions.push_back(pr);
      }
    }
    if (false && usePriorResults){ //1/26/2015 adt, calculate next potential license plate location.  Accuracy is bad so disabling.
      ResultAggregator aggregator;
      AlprFullDetails tempFull, aggregate;
      std::string lastPlate, thisPlate;
      for (int i = priorResults.size(); i--; ){
        thisPlate = priorResults[i].plates[0].bestPlate.characters;
        cout << "prior[" << i << "]:frame(" << priorResults[i].frame_number << ")\t" << 
            "x["<< min (priorResults[i].plates[0].plate_points[0].x, priorResults[i].plates[0].plate_points[3].x) << "," <<
            max (priorResults[i].plates[0].plate_points[1].x, priorResults[i].plates[0].plate_points[2].x) << "]" << 
            "y["<< min (priorResults[i].plates[0].plate_points[0].y, priorResults[i].plates[0].plate_points[1].y) << "," <<
            max (priorResults[i].plates[0].plate_points[2].y, priorResults[i].plates[0].plate_points[3].y) << "]" << 
        "\t"<< thisPlate << "\t\t" << priorResults[i].plates[0].bestPlate.overall_confidence << endl;
        if ((frame_number == priorResults[i].frame_number + 2) || (frame_number == priorResults[i].frame_number + 1)){
          //only add last to priorResults preceding this frame.
          tempFull.results = priorResults[i];
          aggregator.addResults(tempFull);
        }
      }
      std::vector<PlateRegion> nextPrs = aggregator.calcNextPlateRegions();    
      warpedPlateRegions.reserve(warpedPlateRegions.size() + nextPrs.size());
      warpedPlateRegions.insert(warpedPlateRegions.end(), nextPrs.begin(), nextPrs.end());
    }
    queue<PlateRegion> plateQueue;
    for (unsigned int i = 0; i < warpedPlateRegions.size(); i++)
      plateQueue.push(warpedPlateRegions[i]);

    int platecount = 0;
    while(!plateQueue.empty())
    {
      PlateRegion plateRegion = plateQueue.front();
      plateQueue.pop();

      PipelineData pipeline_data(colorImg, grayImg, plateRegion.rect, config);
      pipeline_data.prewarp = prewarp;

      timespec platestarttime;
      getTimeMonotonic(&platestarttime);

      LicensePlateCandidate lp(&pipeline_data);

      lp.recognize();

      bool plateDetected = false;
      if (pipeline_data.disqualified && config->debugGeneral)
      {
        cout << "Disqualify reason: " << pipeline_data.disqualify_reason << endl;
      }
      if (!pipeline_data.disqualified)
      {
        AlprPlateResult plateResult;

        plateResult.country = config->country;

        // If there's only one pattern for a country, use it.  Otherwise use the default
        if (country_recognizers.ocr->postProcessor.getPatterns().size() == 1)
          plateResult.region = country_recognizers.ocr->postProcessor.getPatterns()[0];
        else
          plateResult.region = defaultRegion;

        plateResult.regionConfidence = 0;
        plateResult.plate_index = platecount++;
        plateResult.requested_topn = topN;

        // If using prewarp, remap the plate corners to the original image
        vector<Point2f> cornerPoints = pipeline_data.plate_corners;
        cornerPoints = prewarp->projectPoints(cornerPoints, true);

        for (int pointidx = 0; pointidx < 4; pointidx++)
        {
          plateResult.plate_points[pointidx].x = (int) cornerPoints[pointidx].x;
          plateResult.plate_points[pointidx].y = (int) cornerPoints[pointidx].y;
        }

        if (detectRegion && country_recognizers.stateDetector->isLoaded())
        {
          std::vector<StateCandidate> state_candidates = country_recognizers.stateDetector->detect(pipeline_data.color_deskewed.data,
                                                                               pipeline_data.color_deskewed.elemSize(),
                                                                               pipeline_data.color_deskewed.cols,
                                                                               pipeline_data.color_deskewed.rows);

          if (state_candidates.size() > 0)
          {
            plateResult.region = state_candidates[0].state_code;
            plateResult.regionConfidence = (int) state_candidates[0].confidence;
          }
        }

        if (plateResult.region.length() > 0 && country_recognizers.ocr->postProcessor.regionIsValid(plateResult.region) == false)
        {
          std::cerr << "Invalid pattern provided: " << plateResult.region << std::endl;
          std::cerr << "Valid patterns are located in the " << config->country << ".patterns file" << std::endl;
        }

        country_recognizers.ocr->performOCR(&pipeline_data);
        
        country_recognizers.ocr->postProcessor.analyze(plateResult.region, topN);

        timespec resultsStartTime;
        getTimeMonotonic(&resultsStartTime);

        const vector<PPResult> ppResults = country_recognizers.ocr->postProcessor.getResults();

        int bestPlateIndex = 0;

        cv::Mat charTransformMatrix = getCharacterTransformMatrix(&pipeline_data);
        bool isBestPlateSelected = false;
        for (unsigned int pp = 0; pp < ppResults.size(); pp++)
        {

          // Set our "best plate" match to either the first entry, or the first entry with a postprocessor template match
          if (isBestPlateSelected == false && ppResults[pp].matchesTemplate){
            bestPlateIndex = plateResult.topNPlates.size();
            isBestPlateSelected = true;
          }

          AlprPlate aplate;
          aplate.characters = ppResults[pp].letters;
          aplate.overall_confidence = ppResults[pp].totalscore;
          aplate.matches_template = ppResults[pp].matchesTemplate;
          aplate.method = "ocr"; //2016/05/23 adt, default method is ocr, no heuristics

          // Grab detailed results for each character
          for (unsigned int c_idx = 0; c_idx < ppResults[pp].letter_details.size(); c_idx++)
          {
            AlprChar character_details;
            Letter l = ppResults[pp].letter_details[c_idx];

            character_details.character = l.letter;
            character_details.confidence = l.totalscore;
            cv::Rect char_rect = pipeline_data.charRegionsFlat[l.charposition];
            std::vector<AlprCoordinate> charpoints = getCharacterPoints(char_rect, charTransformMatrix );
            for (int cpt = 0; cpt < 4; cpt++)
              character_details.corners[cpt] = charpoints[cpt];
            aplate.character_details.push_back(character_details);
          }
          plateResult.topNPlates.push_back(aplate);
        }

        if (plateResult.topNPlates.size() > bestPlateIndex)
        {
          AlprPlate bestPlate;
          bestPlate.characters = plateResult.topNPlates[bestPlateIndex].characters;
          bestPlate.matches_template = plateResult.topNPlates[bestPlateIndex].matches_template;
          bestPlate.overall_confidence = plateResult.topNPlates[bestPlateIndex].overall_confidence;
          bestPlate.character_details = plateResult.topNPlates[bestPlateIndex].character_details;
          bestPlate.method = "ocr"; //2016/05/24 store method
          plateResult.bestPlate = bestPlate;
          plateResult.methodPlates[bestPlate.method] = bestPlate; //2016/05/24 store method
        }

        timespec plateEndTime;
        getTimeMonotonic(&plateEndTime);
        plateResult.processing_time_ms = diffclock(platestarttime, plateEndTime);
        if (config->debugTiming)
        {
          cout << "Result Generation Time: " << diffclock(resultsStartTime, plateEndTime) << "ms." << endl;
        }
        
        if (false && (plateResult.topNPlates.size() > 0) && usePriorResults){ //1/26/2015 adt, add character information from same cluster.
          ResultAggregator aggregator;
          AlprFullDetails tempFull, aggregate;
          std::string thisPlate;
          for (int i = priorResults.size(); i--; ){
            for (int k = 0; k < priorResults[i].plates.size(); k++ ){
              thisPlate = priorResults[i].plates[k].bestPlate.characters;
              // cout << "prior[" << i << "]:frame(" << priorResults[i].frame_number << ")\t" << 
              //     "x["<< min (priorResults[i].plates[k].plate_points[0].x, priorResults[i].plates[k].plate_points[3].x) << "," <<
              //     max (priorResults[i].plates[k].plate_points[1].x, priorResults[i].plates[k].plate_points[2].x) << "]" << 
              //     "y["<< min (priorResults[i].plates[k].plate_points[0].y, priorResults[i].plates[k].plate_points[1].y) << "," <<
              //     max (priorResults[i].plates[k].plate_points[2].y, priorResults[i].plates[0].plate_points[3].y) << "]" << 
              // "\t"<< thisPlate << "\t\t" << priorResults[i].plates[k].bestPlate.overall_confidence << endl;
              tempFull.results = priorResults[i];
              aggregator.addResults(tempFull);
            
            }
          }
          std::vector<std::vector<AlprPlateResult> > clusters = aggregator.findClusters();
          int cluster_index = aggregator.overlaps(plateResult, clusters); // match to all regions with overlap
          if (cluster_index >= 0){ //found a potential cluster
            for (int i = 0; i < clusters[cluster_index].size(); i++){ //go through every plate in cluster
              AlprPlateResult pr = clusters[cluster_index][i]; 
              for (int k = 0; k < pr.bestPlate.character_details.size(); k++){ //go through every characters
                for (int j = 0; j < plateResult.bestPlate.character_details.size(); j++){ //review every character in initialPlateResult
                  float confidence =pr.bestPlate.character_details[k].confidence; 
                  std::string choice = pr.bestPlate.character_details[k].character;
                  if (plateResult.bestPlate.character_details[j].character == choice){ //only add a character if it matched a potential choice     
                    country_recognizers.ocr->postProcessor.addLetter(string(choice), 0, k+1, confidence); // charposition appears to already start at 1.
                    if (config->debugGeneral) cout << "plate[" << i << "] adding " << choice << " confidence: " << confidence << " at " << k << endl;
                  }
                }
              }
            }  
          }
          country_recognizers.ocr->postProcessor.analyze(plateResult.region, topN);
          AlprPlateResult newPlateResult;
          timespec resultsStartTime;
          getTimeMonotonic(&resultsStartTime);

          const vector<PPResult> ppResults = country_recognizers.ocr->postProcessor.getResults();

          int bestPlateIndex = 0;
          bool isBestPlateSelected = false;
          for (unsigned int pp = 0; pp < ppResults.size(); pp++)
          {

            // Set our "best plate" match to either the first entry, or the first entry with a postprocessor template match
            if (isBestPlateSelected == false && ppResults[pp].matchesTemplate){
              bestPlateIndex = newPlateResult.topNPlates.size();
              isBestPlateSelected = true;
            }

            AlprPlate aplate;
            aplate.characters = ppResults[pp].letters;
            aplate.overall_confidence = ppResults[pp].totalscore;
            aplate.matches_template = ppResults[pp].matchesTemplate;
            aplate.method = "ocr"; //2016/05/23 adt, default method is ocr, no heuristics

            // Grab detailed results for each character
            for (unsigned int c_idx = 0; c_idx < ppResults[pp].letter_details.size(); c_idx++)
            {
              AlprChar character_details;
              Letter l = ppResults[pp].letter_details[c_idx];

              character_details.character = l.letter;
              character_details.confidence = l.totalscore;
              cv::Rect char_rect = pipeline_data.charRegionsFlat[l.charposition];
              std::vector<AlprCoordinate> charpoints = getCharacterPoints(char_rect, charTransformMatrix );
              for (int cpt = 0; cpt < 4; cpt++)
                character_details.corners[cpt] = charpoints[cpt];
              aplate.character_details.push_back(character_details);
            }
            newPlateResult.topNPlates.push_back(aplate);
          }

          if (newPlateResult.topNPlates.size() > bestPlateIndex)
          {
            AlprPlate bestPlate;
            bestPlate.characters = newPlateResult.topNPlates[bestPlateIndex].characters;
            bestPlate.matches_template = newPlateResult.topNPlates[bestPlateIndex].matches_template;
            bestPlate.overall_confidence = newPlateResult.topNPlates[bestPlateIndex].overall_confidence;
            bestPlate.character_details = newPlateResult.topNPlates[bestPlateIndex].character_details;

            plateResult.bestPlate = bestPlate;
            cout << "newBestPlate found : " << " plateResult: " << newPlateResult.bestPlate.characters << "\t" << newPlateResult.bestPlate.overall_confidence << endl;
          
          }

      

          timespec plateEndTime;
          getTimeMonotonic(&plateEndTime);
          plateResult.processing_time_ms = diffclock(platestarttime, plateEndTime);
          
          if (plateResult.bestPlate.overall_confidence >= newPlateResult.bestPlate.overall_confidence){
            cout << "Initial found better: " << plateResult.bestPlate.characters << "\t" << plateResult.bestPlate.overall_confidence 
              << " NewPlateResult: " << newPlateResult.bestPlate.characters << "\t" << newPlateResult.bestPlate.overall_confidence << endl;
          }else {
            cout << "NewPlateResult found better: " << plateResult.bestPlate.characters << "\t" << plateResult.bestPlate.overall_confidence 
              << " NewPlateResult: " << newPlateResult.bestPlate.characters << "\t" << newPlateResult.bestPlate.overall_confidence << endl;
          }
          if (config->debugTiming)
          {
            cout << "PriorHistory Result Generation Time: " << diffclock(resultsStartTime, plateEndTime) << "ms." << endl;
          }
        }
        if (plateResult.topNPlates.size() > 0)
        {
          plateDetected = true;
          response.results.plates.push_back(plateResult);
        }
      }

      if (!plateDetected)
      {
        // Not a valid plate
        // Check if this plate has any children, if so, send them back up for processing
        for (unsigned int childidx = 0; childidx < plateRegion.children.size(); childidx++)
        {
          plateQueue.push(plateRegion.children[childidx]);
        }
      }

    }
    //1/24/2016 adt, compare plateResult with priorResults. plateResult has already been added so it will be compared against the result_aggregator
    if (response.results.plates.size() > 0 && usePriorResults){
      ResultAggregator aggregator;
      AlprFullDetails tempFull, aggregate;
      std::string lastPlate, thisPlate;
      //fill the aggregator with prior results
      for (int i = 0; i<priorResults.size(); i++){
        for (int k = 0; k < priorResults[i].plates.size(); k++ ){
          if (config->debugGeneral) {
            lastPlate = (i == 0)? "":priorResults[i-1].plates[k].bestPlate.characters;
            thisPlate = priorResults[i].plates[k].bestPlate.characters;
            cout << "priorResults[" << i << "]:frame(" << priorResults[i].frame_number << ")\t" << 
                "x["<< min (priorResults[i].plates[k].plate_points[0].x, priorResults[i].plates[k].plate_points[3].x) << "," <<
                max (priorResults[i].plates[k].plate_points[1].x, priorResults[i].plates[k].plate_points[2].x) << "]" << 
                "y["<< min (priorResults[i].plates[k].plate_points[0].y, priorResults[i].plates[k].plate_points[1].y) << "," <<
                max (priorResults[i].plates[k].plate_points[2].y, priorResults[i].plates[k].plate_points[3].y) << "]" << 
            "\t"<< thisPlate << "\t\t" << priorResults[i].plates[k].bestPlate.overall_confidence << "\tdistance: " << levenshteinDistance(lastPlate, thisPlate, 10)<< "\tlength: " << thisPlate.length() << endl;
          }
          tempFull.results = priorResults[i];
          aggregator.addResults(tempFull);   
        } 
      }
      std::vector<std::vector<AlprPlateResult> > clusters = aggregator.findClusters();
      for (int i = 0; i < response.results.plates.size(); i++) {
        AlprPlateResult plateResult = response.results.plates[i];
        int cluster_index = aggregator.overlaps(plateResult, clusters, 2); // match to all regions with overlap
        if (cluster_index >= 0){ // compare to aggregate results
          aggregate = aggregator.getAggregateResults();
          AlprResults aggregateResults = aggregate.results;
          for (int j = 0; j < aggregateResults.plates[cluster_index].topNPlates.size(); j++){ //find any region match to promote to top of candidates
            if (aggregateResults.plates[cluster_index].topNPlates[j].matches_template){ //set bestPlate to template match candidate
              if (config->debugGeneral){
                cout << "Promoting template match found with candidate index: " <<  j << " plate: " << aggregateResults.plates[cluster_index].topNPlates[j].characters << "\t" << aggregateResults.plates[cluster_index].topNPlates[j].overall_confidence 
                  << " replacing: " << aggregateResults.plates[cluster_index].bestPlate.characters << "\t" << aggregateResults.plates[cluster_index].bestPlate.overall_confidence << endl;
              }
              aggregateResults.plates[cluster_index].bestPlate = aggregateResults.plates[cluster_index].topNPlates[j];
              break;
            }
          }
          AlprPlateResult aggregatePlate = aggregateResults.plates[cluster_index];
          if (((aggregatePlate.bestPlate.overall_confidence > plateResult.bestPlate.overall_confidence) && (aggregatePlate.bestPlate.matches_template == plateResult.bestPlate.matches_template))
            || (aggregatePlate.bestPlate.matches_template && !plateResult.bestPlate.matches_template)){
            cout << "clusterHistory found better at frame: " << aggregateResults.frame_number << " : " << aggregatePlate.bestPlate.characters << " " << aggregatePlate.bestPlate.overall_confidence << " match:" << aggregatePlate.bestPlate.matches_template
              << "\t vs frame: " << response.results.frame_number << " : " << plateResult.bestPlate.characters << " " << plateResult.bestPlate.overall_confidence << " match:" << plateResult.bestPlate.matches_template << endl;
            //TODO:Figure out what aggregateResults since character details won't necessarily be at the same coords
            //aggregatePlate.bestPlate.character_details = plateResult.bestPlate.character_details;
            // aggregatePlate.plate_points[0] = plateResult.plate_points[0];
            // aggregatePlate.plate_points[1] = plateResult.plate_points[1];
            // aggregatePlate.plate_points[2] = plateResult.plate_points[2];
            // aggregatePlate.plate_points[3] = plateResult.plate_points[3];
            response.results.plates[i].bestPlate.characters = aggregatePlate.bestPlate.characters;  
            response.results.plates[i].bestPlate.matches_template = aggregatePlate.bestPlate.matches_template;  
            response.results.plates[i].bestPlate.overall_confidence = aggregatePlate.bestPlate.overall_confidence;  
            response.results.plates[i].bestPlate.method = (aggregatePlate.bestPlate.matches_template == 1) ? "clusterHistory_template_match" : aggregatePlate.bestPlate.method = "clusterHistory";  
            response.results.plates[i].topNPlates = aggregatePlate.topNPlates;  
            response.results.plates[i].group_id = cluster_index;
            aggregateResults.plates[cluster_index] = aggregatePlate;

          }else {
          cout << "clusterHistory result worse at frame: " << aggregateResults.frame_number << " : " << aggregatePlate.bestPlate.characters << " " << aggregatePlate.bestPlate.overall_confidence << " match:" << aggregatePlate.bestPlate.matches_template
            << "\t vs frame: " << response.results.frame_number << " : " << plateResult.bestPlate.characters << " " << plateResult.bestPlate.overall_confidence << " match:" << plateResult.bestPlate.matches_template << endl;
          }
          if (config->debugGeneral){
              for (int i = 0; i<aggregateResults.plates.size(); i++){
                lastPlate = (i == 0)? "":aggregateResults.plates[i-1].bestPlate.characters;
                thisPlate = aggregateResults.plates[i].bestPlate.characters;
                cout << "aggregate[" <<i <<"]:\t" << 
                  "x["<< min (aggregateResults.plates[i].plate_points[0].x, aggregateResults.plates[i].plate_points[3].x) << "," <<
                  max (aggregateResults.plates[i].plate_points[1].x, aggregateResults.plates[i].plate_points[2].x) << "]" << 
                  "y["<< min (aggregateResults.plates[i].plate_points[0].y, aggregateResults.plates[i].plate_points[1].y) << "," <<
                  max (aggregateResults.plates[i].plate_points[2].y, priorResults[i].plates[0].plate_points[3].y) << "]" << 
              "\t"<< aggregateResults.plates[i].bestPlate.characters << "\t\t" << aggregateResults.plates[i].bestPlate.overall_confidence << "\tdistance: " << levenshteinDistance(lastPlate, thisPlate, 10)<<endl;
              }
            if (config->debugGeneral) cout << toJson(response.results) << endl;
          }
        }


      }
      
    }
    // Unwarp plate regions if necessary
    prewarp->projectPlateRegions(warpedPlateRegions, grayImg.cols, grayImg.rows, true);
    response.plateRegions = warpedPlateRegions;

    timespec endTime;
    getTimeMonotonic(&endTime);
    response.results.total_processing_time_ms = diffclock(startTime, endTime);

    return response;
  }

  AlprResults AlprImpl::recognize( std::vector<char> imageBytes)
  {
    try
    {
      cv::Mat img = cv::imdecode(cv::Mat(imageBytes), 1);
      return this->recognize(img);
    }
    catch (cv::Exception& e)
    {
      std::cerr << "Caught exception in OpenALPR recognize: " << e.msg << std::endl;
      AlprResults emptyresults;
      return emptyresults;
    }
  }

  AlprResults AlprImpl::recognize(std::vector<char> imageBytes, std::vector<AlprRegionOfInterest> regionsOfInterest)
  {
    try
    {
      cv::Mat img = cv::imdecode(cv::Mat(imageBytes), 1);

      std::vector<cv::Rect> rois = convertRects(regionsOfInterest);

      AlprFullDetails fullDetails = recognizeFullDetails(img, rois);
      return fullDetails.results;
    }
    catch (cv::Exception& e)
    {
      std::cerr << "Caught exception in OpenALPR recognize: " << e.msg << std::endl;
      AlprResults emptyresults;
      return emptyresults;
    }
  }

  AlprResults AlprImpl::recognize( unsigned char* pixelData, int bytesPerPixel, int imgWidth, int imgHeight, std::vector<AlprRegionOfInterest> regionsOfInterest)
  {

    try
    {
      int arraySize = imgWidth * imgHeight * bytesPerPixel;
      cv::Mat imgData = cv::Mat(arraySize, 1, CV_8U, pixelData);
      cv::Mat img = imgData.reshape(bytesPerPixel, imgHeight);

      if (regionsOfInterest.size() == 0)
      {
        AlprRegionOfInterest fullFrame(0,0, img.cols, img.rows);

        regionsOfInterest.push_back(fullFrame);
      }

      return this->recognize(img, this->convertRects(regionsOfInterest));
    }
    catch (cv::Exception& e)
    {
      std::cerr << "Caught exception in OpenALPR recognize: " << e.msg << std::endl;
      AlprResults emptyresults;
      return emptyresults;
    }
  }

  AlprResults AlprImpl::recognize(cv::Mat img)
  {
    std::vector<cv::Rect> regionsOfInterest;
    regionsOfInterest.push_back(cv::Rect(0, 0, img.cols, img.rows));

    return this->recognize(img, regionsOfInterest);
  }

  AlprResults AlprImpl::recognize(cv::Mat img, std::vector<cv::Rect> regionsOfInterest)
  {
    AlprFullDetails fullDetails = recognizeFullDetails(img, regionsOfInterest);
    return fullDetails.results;
  }


   std::vector<cv::Rect> AlprImpl::convertRects(std::vector<AlprRegionOfInterest> regionsOfInterest)
   {
     std::vector<cv::Rect> rectRegions;
     for (unsigned int i = 0; i < regionsOfInterest.size(); i++)
     {
       rectRegions.push_back(cv::Rect(regionsOfInterest[i].x, regionsOfInterest[i].y, regionsOfInterest[i].width, regionsOfInterest[i].height));
     }

     return rectRegions;
   }

  string AlprImpl::toJson( const AlprResults results )
  {
    cJSON *root, *jsonResults;
    root = cJSON_CreateObject();


    cJSON_AddNumberToObject(root,"version",	2	  );
    cJSON_AddStringToObject(root,"data_type",	"alpr_results"	  );

    cJSON_AddNumberToObject(root,"epoch_time",	results.epoch_time	  );
    cJSON_AddNumberToObject(root,"frame_number",	results.frame_number	  ); //1/24/2016 adt, adding frame_number to json result
    cJSON_AddNumberToObject(root,"img_width",	results.img_width	  );
    cJSON_AddNumberToObject(root,"img_height",	results.img_height	  );
    cJSON_AddNumberToObject(root,"processing_time_ms", results.total_processing_time_ms );

    // Add the regions of interest to the JSON
    cJSON *rois;
    cJSON_AddItemToObject(root, "regions_of_interest", 		rois=cJSON_CreateArray());
    for (unsigned int i=0;i<results.regionsOfInterest.size();i++)
    {
      cJSON *roi_object;
      roi_object = cJSON_CreateObject();
      cJSON_AddNumberToObject(roi_object, "x",  results.regionsOfInterest[i].x);
      cJSON_AddNumberToObject(roi_object, "y",  results.regionsOfInterest[i].y);
      cJSON_AddNumberToObject(roi_object, "width",  results.regionsOfInterest[i].width);
      cJSON_AddNumberToObject(roi_object, "height",  results.regionsOfInterest[i].height);

      cJSON_AddItemToArray(rois, roi_object);
    }


    cJSON_AddItemToObject(root, "results", 		jsonResults=cJSON_CreateArray());
    for (unsigned int i = 0; i < results.plates.size(); i++)
    {
      cJSON *resultObj = createJsonObj( &results.plates[i] );
      cJSON_AddItemToArray(jsonResults, resultObj);
    }

    // Print the JSON object to a string and return
    char *out;
    out=cJSON_PrintUnformatted(root);

    cJSON_Delete(root);

    string response(out);

    free(out);
    return response;
  }



  cJSON* AlprImpl::createJsonObj(const AlprPlateResult* result)
  {
    cJSON *root, *coords, *candidates;

    root=cJSON_CreateObject();

    cJSON_AddStringToObject(root,"plate",		result->bestPlate.characters.c_str());
    cJSON_AddNumberToObject(root,"confidence",		result->bestPlate.overall_confidence);
    cJSON_AddNumberToObject(root,"matches_template",	result->bestPlate.matches_template);
    cJSON_AddStringToObject(root,"method",	result->bestPlate.method.c_str());

    cJSON_AddNumberToObject(root,"plate_index",               result->plate_index);
    //2016/05/25 adt, adding group_id for video clusters
    cJSON_AddNumberToObject(root,"group_id",               result->group_id); 

    cJSON_AddStringToObject(root,"region",		result->region.c_str());
    cJSON_AddNumberToObject(root,"region_confidence",	result->regionConfidence);

    cJSON_AddNumberToObject(root,"processing_time_ms",	result->processing_time_ms);
    cJSON_AddNumberToObject(root,"requested_topn",	result->requested_topn);

    cJSON_AddItemToObject(root, "coordinates", 		coords=cJSON_CreateArray());
    for (int i=0;i<4;i++)
    {
      cJSON *coords_object;
      coords_object = cJSON_CreateObject();
      cJSON_AddNumberToObject(coords_object, "x",  result->plate_points[i].x);
      cJSON_AddNumberToObject(coords_object, "y",  result->plate_points[i].y);

      cJSON_AddItemToArray(coords, coords_object);
    }


    cJSON_AddItemToObject(root, "candidates", 		candidates=cJSON_CreateArray());
    for (unsigned int i = 0; i < result->topNPlates.size(); i++)
    {
      cJSON *candidate_object;
      candidate_object = cJSON_CreateObject();
      cJSON_AddStringToObject(candidate_object, "plate",  result->topNPlates[i].characters.c_str());
      cJSON_AddNumberToObject(candidate_object, "confidence",  result->topNPlates[i].overall_confidence);
      cJSON_AddNumberToObject(candidate_object, "matches_template",  result->topNPlates[i].matches_template);
      cJSON_AddStringToObject(candidate_object, "method",  result->topNPlates[i].method.c_str());

      cJSON_AddItemToArray(candidates, candidate_object);
    }

    return root;
  }

  AlprResults AlprImpl::fromJson(std::string json) {
    AlprResults allResults;

    cJSON* root = cJSON_Parse(json.c_str());

    int version = cJSON_GetObjectItem(root, "version")->valueint;
    allResults.epoch_time = (int64_t) cJSON_GetObjectItem(root, "epoch_time")->valuedouble;
    allResults.frame_number = (int64_t) cJSON_GetObjectItem(root, "frame_number")->valuedouble; //1/24/2016, adt adding frame_number to json
    allResults.img_width = cJSON_GetObjectItem(root, "img_width")->valueint;
    allResults.img_height = cJSON_GetObjectItem(root, "img_height")->valueint;
    allResults.total_processing_time_ms = cJSON_GetObjectItem(root, "processing_time_ms")->valueint;


    cJSON* rois = cJSON_GetObjectItem(root,"regions_of_interest");
    int numRois = cJSON_GetArraySize(rois);
    for (int c = 0; c < numRois; c++)
    {
      cJSON* roi = cJSON_GetArrayItem(rois, c);
      int x = cJSON_GetObjectItem(roi, "x")->valueint;
      int y = cJSON_GetObjectItem(roi, "y")->valueint;
      int width = cJSON_GetObjectItem(roi, "width")->valueint;
      int height = cJSON_GetObjectItem(roi, "height")->valueint;

      AlprRegionOfInterest alprRegion(x,y,width,height);
      allResults.regionsOfInterest.push_back(alprRegion);
    }

    cJSON* resultsArray = cJSON_GetObjectItem(root,"results");
    int resultsSize = cJSON_GetArraySize(resultsArray);

    for (int i = 0; i < resultsSize; i++)
    {
      cJSON* item = cJSON_GetArrayItem(resultsArray, i);
      AlprPlateResult plate;

      //plate.bestPlate = cJSON_GetObjectItem(item, "plate")->valuestring;
      plate.processing_time_ms = cJSON_GetObjectItem(item, "processing_time_ms")->valuedouble;
      plate.plate_index = cJSON_GetObjectItem(item, "plate_index")->valueint;
      //2016/05/25 adt, adding group_id for video clusters
      plate.group_id = cJSON_GetObjectItem(item, "group_id")->valueint;
      plate.region = std::string(cJSON_GetObjectItem(item, "region")->valuestring);
      plate.regionConfidence = cJSON_GetObjectItem(item, "region_confidence")->valueint;
      plate.requested_topn = cJSON_GetObjectItem(item, "requested_topn")->valueint;


      cJSON* coordinates = cJSON_GetObjectItem(item,"coordinates");
      for (int c = 0; c < 4; c++)
      {
        cJSON* coordinate = cJSON_GetArrayItem(coordinates, c);
        AlprCoordinate alprcoord;
        alprcoord.x = cJSON_GetObjectItem(coordinate, "x")->valueint;
        alprcoord.y = cJSON_GetObjectItem(coordinate, "y")->valueint;

        plate.plate_points[c] = alprcoord;
      }

      cJSON* candidates = cJSON_GetObjectItem(item,"candidates");
      int numCandidates = cJSON_GetArraySize(candidates);
      for (int c = 0; c < numCandidates; c++)
      {
        cJSON* candidate = cJSON_GetArrayItem(candidates, c);
        AlprPlate plateCandidate;
        plateCandidate.characters = std::string(cJSON_GetObjectItem(candidate, "plate")->valuestring);
        plateCandidate.overall_confidence = cJSON_GetObjectItem(candidate, "confidence")->valuedouble;
        plateCandidate.matches_template = (cJSON_GetObjectItem(candidate, "matches_template")->valueint) != 0;
        plateCandidate.matches_template = (cJSON_GetObjectItem(candidate, "method")->valuestring);

        plate.topNPlates.push_back(plateCandidate);

        if (c == 0)
        {
          plate.bestPlate = plateCandidate;
        }
      }

      allResults.plates.push_back(plate);
    }


    cJSON_Delete(root);


    return allResults;
  }


  void AlprImpl::setDetectRegion(bool detectRegion)
  {

    this->detectRegion = detectRegion;


  }
  void AlprImpl::setTopN(int topn)
  {
    this->topN = topn;
  }
  void AlprImpl::setDefaultRegion(string region)
  {
    this->defaultRegion = region;
  }

  std::string AlprImpl::getVersion()
  {
    std::stringstream ss;

    ss << OPENALPR_MAJOR_VERSION << "." << OPENALPR_MINOR_VERSION << "." << OPENALPR_PATCH_VERSION;
    return ss.str();
  }

  cv::Mat AlprImpl::getCharacterTransformMatrix(PipelineData* pipeline_data ) {
    std::vector<Point2f> crop_corners;
    crop_corners.push_back(Point2f(0,0));
    crop_corners.push_back(Point2f(pipeline_data->crop_gray.cols,0));
    crop_corners.push_back(Point2f(pipeline_data->crop_gray.cols,pipeline_data->crop_gray.rows));
    crop_corners.push_back(Point2f(0,pipeline_data->crop_gray.rows));

    // Transform the points from the cropped region (skew corrected license plate region) back to the original image
    cv::Mat transmtx = cv::getPerspectiveTransform(crop_corners, pipeline_data->plate_corners);

    return transmtx;
  }

  std::vector<AlprCoordinate> AlprImpl::getCharacterPoints(cv::Rect char_rect, cv::Mat transmtx ) {


    std::vector<Point2f> points;
    points.push_back(Point2f(char_rect.x, char_rect.y));
    points.push_back(Point2f(char_rect.x + char_rect.width, char_rect.y));
    points.push_back(Point2f(char_rect.x + char_rect.width, char_rect.y + char_rect.height));
    points.push_back(Point2f(char_rect.x, char_rect.y + char_rect.height));

    cv::perspectiveTransform(points, points, transmtx);

    // If using prewarp, remap the points to the original image
    points = prewarp->projectPoints(points, true);


    std::vector<AlprCoordinate> cornersvector;
    for (int i = 0; i < 4; i++)
    {
      AlprCoordinate coord;
      coord.x = round(points[i].x);
      coord.y = round(points[i].y);
      cornersvector.push_back(coord);
    }

    return cornersvector;
  }


}
