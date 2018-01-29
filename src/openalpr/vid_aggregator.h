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
 * Extends result_aggregator and is performs aggregation functions on different frames from same video
 */
 
#ifndef OPENALPR_VIDAGGREGATOR_H
#define OPENALPR_VIDAGGREGATOR_H


#include "alpr_impl.h"
#include "result_aggregator.h"
#include <iomanip>


namespace alpr
{

  class VidAggregator : public ResultAggregator
  {
  public:
    VidAggregator(ResultMergeStrategy merge_strategy, int topn, Config* config) : ResultAggregator (merge_strategy, topn, config)
    {};
    // 
    // virtual ~VidAggregator();
    // 
    void addResults(AlprFullDetails full_results);
    // 
    AlprFullDetails getAggregateResults();
    //  Add a single result
    //1/26/2016 adt, making cluster operations public
    void genClusters();
    std::vector<std::vector<AlprPlateResult> > findClusters();

    //1/24/2016 adt, adding new overlaps that takes Levenshtein_distance to add to a cluster.  
    int overlaps(AlprPlateResult plate, int maxLDistance);
    //1/25/2016 adt, calculate the next potential plate regions for each cluster.
    std::vector<PlateRegion> calcNextPlateRegions();

  private:  
    std::vector<std::vector<AlprPlateResult> > clusters;
    int lastClusterCalc; //keep track of last time clusters saved
  };

}

#endif //OPENALPR_VIDAGGREGATOR_H
