
#include "properties.h"

#include <iostream>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <math.h>
#include <map>
#include <numeric>
#include <opencv2/core.hpp>
//#include "matplotlibcpp.h"

namespace properties {

using namespace std;
//namespace plt = matplotlibcpp;

struct frame_data_ frame_data[10];

int current_frame_index = 0;

void printEvalData() {

    cout << "===============================================\n";
    cout << "Eval Data:\n";
    cout << "===============================================\n";
    cout << "Feature Detector  : " << keypoint_detector_type << endl;
    cout << "Feature Descriptor: " << feature_descriptor_type << endl;   
    cout << "No. of Feature in ROI: ";

    vector<float> ft_cnt;
    //vector<float> match_cnt;
    for (auto frame : frame_data) {
        //cout << frame.features_in_roi << ", ";
        ft_cnt.push_back(frame.features_in_roi);
        //match_cnt.push_back(frame.feature_matches);
    } 
    {
        float min = *min_element(ft_cnt.begin(), ft_cnt.end());
        float max = *max_element(ft_cnt.begin(), ft_cnt.end());
        float avg = accumulate(ft_cnt.begin(), ft_cnt.end(), 0.0f) / ft_cnt.size();
        cout << " [min, max, avg] = [" << min << ", " << max << ", " << avg << "]\n";
    }

    cout << "Keypoint distribution: size (count avg):\n";
    map<float, int> size_distribution;
    map<float, vector<float>> ft_dist;
    for (auto frame : frame_data) {
        //unordered_set<float> size_set;
        set<float> size_set;
        for (auto size : frame.feature_size) {
            size_set.insert((size));
        }
        for (auto size : size_set) {
            int ft_count = count(frame.feature_size.begin(), frame.feature_size.end(), size);
            //cout << size << " (" << feature_count << "), ";
            // auto it = size_distribution.find(size);
            // if (it == size_distribution.end()) {
            //     size_distribution.insert(pair<float, int>(size, feature_count));
            //     cout << "key not found (" << size << "). adding count = " << feature_count << endl;
            // } else {  
            //     size_distribution[size] += feature_count;
            //     cout << "key found (" << size << "). adding count = " << feature_count << ". total: " << size_distribution[size] << endl;           
            // }
            size_distribution[size] += ft_count;
            auto it = ft_dist.find(size);
            if (it == ft_dist.end()) {
                ft_dist.insert(pair<float, vector<float>>(size, {static_cast<float>(ft_count)}));
            } else {
                it->second.push_back(static_cast<float>(ft_count));
            }
        }
        //cout << endl;
    } 
    //cout << endl;
    // int frame_count = sizeof(frame_data)/sizeof(frame_data[0]);
    // for (auto it=size_distribution.begin(); it!=size_distribution.end(); ++it) {
    //     cout << it->first << " (" << static_cast<float>(it->second)/frame_count << "), ";
    // }
    // cout << endl;

    // compute statistics of feature size
    vector<vector<float>> ft_stats;
    for (auto it=ft_dist.begin(); it!=ft_dist.end(); ++it) {
        vector<float>& ft_cnt = it->second;
        float min = *min_element(ft_cnt.begin(), ft_cnt.end());
        float max = *max_element(ft_cnt.begin(), ft_cnt.end());
        float avg = accumulate(ft_cnt.begin(), ft_cnt.end(), 0.0f) / ft_cnt.size();
        cout << it->first << " [" << min << ", " << max << ", " << avg << "]\n"; 
        vector<float> ft = {it->first, min, max, avg};
        ft_stats.push_back(ft);
    }
    cout << endl;

    // plt::title(properties::keypoint_detector_type);
    // plt::plot(ft_cnt);
    // plt::show();

    //{
        //plt::hist({3,4,6,2,1});
        //plt::show();


    //}
    {
        // write statistical data to file
        vector<vector<float>> ft_size;
        for (auto frame : frame_data) {
            ft_size.push_back(frame.feature_size);
        }

        vector<float> match_cnt;
        vector<float> det_time, des_time, match_time;
        for (auto frame : frame_data) {
            match_cnt.push_back(frame.feature_matches);
            det_time.push_back(frame.detector_time);
            des_time.push_back(frame.descriptor_time);
            match_time.push_back(frame.matcher_time);
        } 

        cv::FileStorage fs("eval-detector-" + properties::keypoint_detector_type + "-" +properties::feature_descriptor_type+ ".json", cv::FileStorage::WRITE);
        // cv::FileStorage fs("eval-DET_" + properties::keypoint_detector_type + "-DES_" + properties::feature_descriptor_type + "-" + properties::feature_matcher_type + "-" + properties::match_selector_type + ".json", cv::FileStorage::WRITE);
        fs << "detector_type" << properties::keypoint_detector_type;
        fs << "descriptor_type" << properties::feature_descriptor_type;
        fs << "matcher_type" << properties::feature_matcher_type;
        fs << "selector_type" << properties::match_selector_type;

        fs << "detector_time" << det_time;
        fs << "descriptor_time" << des_time;  
        fs << "matcher_time" << match_time;
 
        fs << "detector_time_avg" << accumulate(det_time.begin(), det_time.end(), 0.0f)/det_time.size();
        fs << "descriptor_time_avg" << accumulate(des_time.begin(), des_time.end(), 0.0f)/des_time.size();
        match_time.erase(match_time.begin());
        fs << "matcher_time_avg" << accumulate(match_time.begin(), match_time.end(), 0.0f)/match_time.size();
        
        fs << "feature_count" << ft_cnt;
        fs << "match_count" << match_cnt; 

        fs << "feature_count_avg" << accumulate(ft_cnt.begin(), ft_cnt.end(), 0.0f)/ft_cnt.size();
        fs << "match_count_avg" << accumulate(match_cnt.begin(), match_cnt.end(), 0.0f)/match_cnt.size(); 

        fs << "feature_size"  << ft_stats;
        fs.release();
    }


}

} // namespace properties