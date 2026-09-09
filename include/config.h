#pragma once

#include <string>

// arithmetics
const float EPSILON = 1e-3;

// directories
const std::string WORKSPACE = "./";
const std::string LOGS = WORKSPACE + "/logs";
const std::string CKPS = WORKSPACE + "/checkpoints";
const std::string STATS = WORKSPACE + "/stats";

const std::string DATASPACE = "/opt/nfs_dcc/chunxy/SVS";
const std::string ATTR = DATASPACE + "/attr";
const std::string GT = DATASPACE + "/gt";

// query paths
const std::string QUERY_CENTER_PATH_TMPL = ATTR + "/{}_query.center.bin";      // {name}
const std::string QUERY_RADIUS_PATH_TMPL = ATTR + "/{}_query.radius.bin";      // {name}
const std::string QUERY_RECT_MIN_PATH_TMPL = ATTR + "/{}_query.rect.min.bin";  // {name}
const std::string QUERY_RECT_MAX_PATH_TMPL = ATTR + "/{}_query.rect.max.bin";  // {name}

// groundtruth paths
const std::string GT_PATH_TMPL = GT + "/{}_{}.gt";  // {name}_{ng}

// workload names
const std::string WORKLOAD_TMPL = "{}_{}";  // {datacard}_{k}
// index-related names
const std::string CHECKPOINT_TMPL = "{}_{}_{}.{}";  // {datacard}_{M}_{efc}.{index_type}