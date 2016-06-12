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
/*
 * Copyright (c) 2016 Alan D. Tse.
 * Extends result_aggregator and is intended to perform aggregation functions on different frames from same video
 */
 
#include "vid_aggregator.h"

using namespace std;
using namespace cv;

namespace alpr
{

  // VidAggregator::VidAggregator()
  // {
  // 
  // }
  // 
  // VidAggregator::~VidAggregator() {
  // 
  // }
  // 
  // void VidAggregator::addResults(AlprFullDetails full_results)
  // {
  //   all_results.push_back(full_results);
  // }
  AlprFullDetails VidAggregator::getAggregateResults()
  {
    assert(all_results.size() > 0);
  
    if (all_results.size() == 1)
      return all_results[0];
  
    timespec startTime; //1/29/2016 adt, adding timing benchmark to aggregateResults
    getTimeMonotonic(&startTime);
    
    AlprFullDetails response;
  
    // Plate regions are needed for benchmarking
    // Copy all detected boxes across all results
    for (unsigned int i = 0; i < all_results.size(); i++)
    {
      for (unsigned int k = 0; k < all_results[i].plateRegions.size(); k++)
        response.plateRegions.push_back(all_results[i].plateRegions[k]);
    }
  
  
    response.results.epoch_time = all_results.back().results.epoch_time; //1/28/2016 adt, setting last added result
    response.results.frame_number = all_results.back().results.frame_number; //1/24/2016 adt, adding frame_number; select last added result
    //cout << "Set frame_number: " << response.results.frame_number << endl;
    response.results.img_height = all_results[0].results.img_height;
    response.results.img_width = all_results[0].results.img_width;
    response.results.total_processing_time_ms = all_results.back().results.total_processing_time_ms; 
    response.results.regionsOfInterest = all_results.back().results.regionsOfInterest;
  
  
    vector<vector<AlprPlateResult> > clusters = findClusters();
  
    // Assume we have multiple results, one cluster for each unique train data (e.g., eu, eu2)
  
    // Now for each cluster of plates, pick the best one
    for (unsigned int i = 0; i < clusters.size(); i++)
    {
      float best_confidence = 0;
      int best_index = 0;
      vector<AlprPlate> newCandidates;
      for (unsigned int k = 0; k < clusters[i].size(); k++)
      {
        if (clusters[i][k].bestPlate.overall_confidence > best_confidence)
        {
          best_confidence = clusters[i][k].bestPlate.overall_confidence;
          best_index = k;
        }
        //2016/05/15 adt, create list of candidates based off all plates in cluster instead of just bestPlate.
        // NOTE: this may break other openalpr implementations that assume candidates will come from a single plate 
        newCandidates.reserve(newCandidates.size() + clusters[i][k].topNPlates.size());
        newCandidates.insert(newCandidates.end(), clusters[i][k].topNPlates.begin(), clusters[i][k].topNPlates.end());
      }
      //2016/05/16 adt, sort clustersCandidates
      std::sort(newCandidates.begin(),newCandidates.end(), greater<AlprPlate>());
      //insert into topNPlates, but only if not seen before
      clusters[i][best_index].topNPlates.clear();
      map<std::string,int> mymap;
      for (unsigned int k = 0; k < newCandidates.size() ; k++)
      {
        if ((clusters[i][best_index].topNPlates.size() < clusters[i][best_index].topNPlates.capacity())
          && (mymap.count(newCandidates[k].characters) == 0))
          clusters[i][best_index].topNPlates.push_back(newCandidates[k]);
          //build methodPlates up again to include the best of each method
          if ((clusters[i][best_index].methodPlates.count(newCandidates[k].method) == 0) ||
              (clusters[i][best_index].methodPlates[newCandidates[k].method].overall_confidence < newCandidates[k].overall_confidence))
            clusters[i][best_index].methodPlates[newCandidates[k].method] = newCandidates[k];
          mymap[newCandidates[k].characters] = 1;
      }
      //cout << "Cluster[" << i << "] BestPlate:" << clusters[i][best_index].bestPlate.characters << " confidence: " << clusters[i][best_index].bestPlate.overall_confidence << endl;
      response.results.plates.push_back(clusters[i][best_index]);
    }
  
    //1/29/2016 adt, setting processing time for aggregator
    timespec endTime;
    getTimeMonotonic(&endTime);
    response.results.total_processing_time_ms += diffclock(startTime, endTime);
    
    return response;
  }

  // Searches all_plates to find overlapping plates
  // Returns an array containing "clusters" (overlapping plates)
  std::vector<std::vector<AlprPlateResult> > VidAggregator::findClusters()
  {
    std::vector<std::vector<AlprPlateResult> > clusters;

    for (unsigned int i = 0; i < all_results.size(); i++)
    {
      for (unsigned int plate_id = 0; plate_id < all_results[i].results.plates.size(); plate_id++)
      {
        AlprPlateResult plate = all_results[i].results.plates[plate_id];

        int cluster_index = overlaps(plate, clusters, 2); //1/24/2016 adt, setting maxLevenStein distance of 2 for overloaded function
        if (cluster_index < 0)
        {
          vector<AlprPlateResult> new_cluster;
          new_cluster.push_back(plate);
          clusters.push_back(new_cluster);
        }
        else
        {
          clusters[cluster_index].push_back(plate);
        }
      }
    }

    return clusters;
  }

  //1/24/2016 adt, adding overlaps that takes Levenshtein_distance to add to a cluster.  This is primarily to allow reuse of overlaps
  // where the plate may have moved beyond the overlap but is sufficiently close. This will start from the latest entries and only process images from the last frame for each cluster.
  // Returns the cluster ID if the plate overlaps or if no overlap, Levenshtein distance is less than maxLDistance.  Otherwise returns -1
  // TODO: Clear out old clusters if overlaps; match to cluster if > 2 spots identical
  int VidAggregator::overlaps(AlprPlateResult plate,
                                 std::vector<std::vector<AlprPlateResult> > clusters,
                                 int maxLDistance)
  {
    // Check the center positions to see how close they are to each other
    // Also compare the size.  If it's much much larger/smaller, treat it as a separate cluster

    PlateShapeInfo psi = getShapeInfo(plate);
    AlprPlateResult plateResult;
    int distance, adjDistance;

    for (unsigned int i = clusters.size(); i-- > 0;) //1/24/2016 adt,reverse order so latest frames first
    {
      for (unsigned int k = clusters[i].size(); k-- > 0;) 
      {
        plateResult = clusters[i][k];
          
        PlateShapeInfo cluster_shapeinfo = getShapeInfo(plateResult);
        
        int diffx = abs(psi.center.x - cluster_shapeinfo.center.x);
        int diffy = abs(psi.center.y - cluster_shapeinfo.center.y);

        // divide the larger plate area by the smaller plate area to determine a match
        float area_diff;
        if (psi.area > cluster_shapeinfo.area)
          area_diff = psi.area / cluster_shapeinfo.area;
        else
          area_diff = cluster_shapeinfo.area / psi.area;

        int max_x_diff = (psi.max_width + cluster_shapeinfo.max_width) / 2;
        int max_y_diff = (psi.max_height + cluster_shapeinfo.max_height) / 2;

        float max_area_diff = 4.0;
        // Consider it a match if center diffx/diffy are less than the average height
        // the area is not more than 4x different
        // calculate levenshteinDistance using both assisted and unassisted plate characters
        std::string plateChars = plate.bestPlate.characters,
          plateOCRChars = plate.methodPlates["ocr"].characters,
          plateResultChars = plateResult.bestPlate.characters,
          plateResultOCRChars = plateResult.methodPlates["ocr"].characters;
        distance = levenshteinDistance(plateOCRChars, plateResultOCRChars, max(plateOCRChars.size(),plateResultOCRChars.size()));
        // calculate adjusted distance which will adjust distance based on lengths of characters in case of occlusions.
        adjDistance = distance - abs((int) plateOCRChars.length() - (int) plateResultOCRChars.length());
        //cout << plateChars << " (" << plateOCRChars <<") vs \t" << plateResultOCRChars << "\tdistance: " << distance <<  "\tadjDistance: " << adjDistance << endl;

        if (diffx <= max_x_diff && diffy <= max_y_diff && area_diff <= max_area_diff){ //no need to check for distance if overlap
          cout << plateChars << " (" << plateOCRChars <<") overlap added to cluster[" << i << "]\t" << plateResultOCRChars << "\tdistance: " << distance <<  "\tadjDistance: " << adjDistance << endl;
          return i;
        }
        //Do a comparison to the last plate in the cluster for levenshteinDistance match using adjDistance
        if (adjDistance <= maxLDistance)
        {
          //cout << "Levenshtein match: " << plate.bestPlate.characters << "\t" << plateResult.bestPlate.characters << "\tdistance: " << distance <<  "\tadjDistance: " << adjDistance << endl;//levenshteinDistance(plateResult.bestPlate.characters, plate.bestPlate.characters,10) << endl;
          cout << plateChars << " (" << plateOCRChars <<") Levenshtein added to cluster[" << i << "]\t" << plateResultOCRChars << "\tdistance: " << distance <<  "\tadjDistance: " << adjDistance << endl;
          return i;
        }
      }


    }

    cout << plate.bestPlate.characters << " added to new cluster[" << clusters.size()  << "]" << endl;
    return -1;
  }
  //1/25/2016 adt, calculate the next potential plate regions for each cluster based on frame in cluster.  If there is
  //only one frame, will assume no movement.  Since clusters do not contain frame_numbers, this function assumes the last 2 plateResults
  //in each cluster are immediately preceding frames.
  //Note: Average over two frames appears to have minimal benefit.
  std::vector<PlateRegion> VidAggregator::calcNextPlateRegions(){
    std::vector<PlateRegion> prs;
    PlateRegion pr;
    vector<vector<AlprPlateResult> > clusters = findClusters();
    for (unsigned int i = clusters.size(); i-- > 0;) //start from last clusters because most likely to exist
    {
      int lastPlatex = -1,
        lastPlatey = -1,
        lastWidth = -1,
        lastHeight = -1;
      for (unsigned int k = clusters[i].size(); k-- > 0;) 
      {
        AlprPlateResult plateResult = clusters[i][k];
        int x0 = plateResult.plate_points[0].x, 
          x1 = plateResult.plate_points[1].x, 
          x2 = plateResult.plate_points[2].x, 
          x3 = plateResult.plate_points[3].x, 
          y0 = plateResult.plate_points[0].y, 
          y1 = plateResult.plate_points[1].y, 
          y2 = plateResult.plate_points[2].y, 
          y3 = plateResult.plate_points[3].y,
          minx = min(x0, x3),
          miny = min(y0, y1),
          maxx = max(x1, x2),
          maxy = max(y2, y3),
          width = maxx - minx,
          height = maxy - miny;
         
        if (clusters[i].size() == 1) {// nothing to compare, just return last rectangle.      
          cout << "Cluster["<<i<<"] returning non-moving rectangle " << minx << " " << miny << " " << width << " " << height << endl;          
          pr.rect = cv::Rect(minx, miny , width, height);
          prs.push_back(pr);
          break;
        }else if (lastPlatex >=0) { //we already processed one frame, so calculate new coords based on velocities 
          float widthProportion = lastWidth,//(abs(lastWidth-width)/width)*lastWidth, TODO: Fix proportional sizing
            heightProportion = lastHeight;//(abs(lastHeight-height)/height)*lastHeight;
          cout << lastPlatex << "," <<lastWidth <<"," << minx << "," <<width <<  "," <<heightProportion << endl;
          cout << lastPlatey << "," <<lastHeight <<"," << miny << "," <<height << "," <<heightProportion << endl;

          int newx =  (minx + width/2)- (lastPlatex + lastWidth/2) + lastPlatex,
          newy =  (miny + height/2) - (lastPlatey + lastHeight/2)+ lastPlatey,
          newWidth = round(widthProportion),
          newHeight = round(heightProportion);
          cout << "Cluster["<<i<<"] returning moving rectangle " << newx << " " << newy << " " << newWidth << " " << newHeight << endl;          
          pr.rect = cv::Rect(newx, newy, newWidth, newHeight);
          prs.push_back(pr);
          break;
        }else{ // still processing last frame, need to grab frame before it
        lastPlatex = minx;
        lastPlatey = miny;
        lastWidth = width;
        lastHeight = height;
        }   
      }
    }
    return prs;
  }

}