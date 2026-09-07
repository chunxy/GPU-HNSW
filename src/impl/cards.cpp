#include <map>
#include <string>
#include "utils/Pod.h"

static const std::string siftsmall_bpath = "/home/sean/datasets/siftsmall/siftsmall_base.fvecs";
static const std::string siftsmall_qpath = "/home/sean/datasets/siftsmall/siftsmall_query.fvecs";
static const std::string siftsmall_gpath = "/home/sean/datasets/siftsmall/siftsmall_groundtruth.ivecs";

static const std::string sift_bpath = "/home/sean/datasets/sift/sift_base.fvecs";
static const std::string sift_qpath = "/home/sean/datasets/sift/sift_query.fvecs";
static const std::string sift_gpath = "/home/sean/datasets/sift/sift_groundtruth.ivecs";

static const std::string gist_bpath = "/home/sean/datasets/gist/gist_base.fvecs";
static const std::string gist_qpath = "/home/sean/datasets/gist/gist_query.fvecs";
static const std::string gist_gpath = "/home/sean/datasets/gist/gist_groundtruth.ivecs";

static const std::string crawl_bpath = "/home/sean/datasets/crawl/crawl_base.fvecs";
static const std::string crawl_qpath = "/home/sean/datasets/crawl/crawl_query.fvecs";
static const std::string crawl_gpath = "/home/sean/datasets/crawl/crawl_groundtruth.ivecs";

static const std::string glove100_bpath = "/home/sean/datasets/glove100/glove100_base.fvecs";
static const std::string glove100_qpath = "/home/sean/datasets/glove100/glove100_query.fvecs";
static const std::string glove100_gpath = "/home/sean/datasets/glove100/glove100_groundtruth.ivecs";

static const std::string audio_bpath = "/home/sean/datasets/audio/audio_base.fvecs";
static const std::string audio_qpath = "/home/sean/datasets/audio/audio_query.fvecs";
static const std::string audio_gpath = "/home/sean/datasets/audio/audio_groundtruth.ivecs";

static const std::string video_bpath = "/home/sean/datasets/video/video_base.fvecs";
static const std::string video_qpath = "/home/sean/datasets/video/video_query.fvecs";
static const std::string video_gpath = "/home/sean/datasets/video/video_groundtruth.ivecs";

static const std::string audio_dedup_bpath = "/home/sean/datasets/audio-dedup/audio-dedup_base.fvecs";
static const std::string audio_dedup_qpath = "/home/sean/datasets/audio-dedup/audio-dedup_query.fvecs";
static const std::string audio_dedup_gpath = "/home/sean/datasets/audio-dedup/audio-dedup_groundtruth.ivecs";

static const std::string video_dedup_bpath = "/home/sean/datasets/video-dedup/video-dedup_base.fvecs";
static const std::string video_dedup_qpath = "/home/sean/datasets/video-dedup/video-dedup_query.fvecs";
static const std::string video_dedup_gpath = "/home/sean/datasets/video-dedup/video-dedup_groundtruth.ivecs";

static const std::string sift_dedup_bpath = "/home/sean/datasets/sift-dedup/sift-dedup_base.fvecs";
static const std::string sift_dedup_qpath = "/home/sean/datasets/sift-dedup/sift-dedup_query.fvecs";
static const std::string sift_dedup_gpath = "/home/sean/datasets/sift-dedup/sift-dedup_groundtruth.ivecs";

static const std::string gist_dedup_bpath = "/home/sean/datasets/gist-dedup/gist-dedup_base.fvecs";
static const std::string gist_dedup_qpath = "/home/sean/datasets/gist-dedup/gist-dedup_query.fvecs";
static const std::string gist_dedup_gpath = "/home/sean/datasets/gist-dedup/gist-dedup_groundtruth.ivecs";

static const std::string flickr_bpath = "/opt/nfs_dcc/chunxy/SVS/flickr/flickr_base.fvecs";
static const std::string flickr_qpath = "/opt/nfs_dcc/chunxy/SVS/flickr/flickr_query.fvecs";
static const std::string flickr_gpath = "/opt/nfs_dcc/chunxy/SVS/flickr/flickr_groundtruth.ivecs";

static const std::string deep10m_bpath = "/home/sean/datasets/deep10m/deep10m_base.fvecs";
static const std::string deep10m_qpath = "/home/sean/datasets/deep10m/deep10m_query.fvecs";
static const std::string deep10m_gpath = "/home/sean/datasets/deep10m/deep10m_groundtruth.ivecs";

static const std::string word2vec_bpath = "/opt/nfs_dcc/chunxy/datasets/word2vec/word2vec_base.fvecs";
static const std::string word2vec_qpath = "/opt/nfs_dcc/chunxy/datasets/word2vec/word2vec_query.fvecs";
static const std::string word2vec_gpath = "/opt/nfs_dcc/chunxy/datasets/word2vec/word2vec_groundtruth.ivecs";

DataCard siftsmall{
    "siftsmall",
    siftsmall_bpath,
    siftsmall_qpath,
    siftsmall_gpath,
    128,
    10000,
    100,
    10,
};

DataCard sift{
    "sift",
    sift_bpath,
    sift_qpath,
    sift_gpath,
    128,
    1000000,
    10000,
    100
};

DataCard gist{
    "gist",
    gist_bpath,
    gist_qpath,
    gist_gpath,
    960,
    10000,
    1'000'000,
    100,
};

DataCard crawl{
    "crawl",
    crawl_bpath,
    crawl_qpath,
    crawl_gpath,
    300,
    1'989'995,
    10000,
    100,
};

DataCard glove100{
    "glove100",
    glove100_bpath,
    glove100_qpath,
    glove100_gpath,
    100,
    1'183'514,
    10000,
    100,
};

DataCard audio_dedup{
    "audio-dedup",
    audio_dedup_bpath,
    audio_dedup_qpath,
    audio_dedup_gpath,
    128,
    1'000'000,
    10000,
    100,
};

DataCard video_dedup{
    "video-dedup",
    video_dedup_bpath,
    video_dedup_qpath,
    video_dedup_gpath,
    1024,
    1'000'000,
    10000,
    100,
};

DataCard sift_dedup{
    "sift-dedup",
    sift_dedup_bpath,
    sift_dedup_qpath,
    sift_dedup_gpath,
    128,
    1'000'000 - 14538,
    10000,
    100,
};

DataCard gist_dedup{
    "gist-dedup",
    gist_dedup_bpath,
    gist_dedup_qpath,
    gist_dedup_gpath,
    960,
    1'000'000 - 17306,
    10000,
    100,
};

DataCard deep10m{
    "deep10m",
    deep10m_bpath,
    deep10m_qpath,
    deep10m_gpath,
    96,
    10000000,
    10000,
    100,
};

std::map<std::string, DataCard> name_to_card{
    {"siftsmall", siftsmall},
    {"sift", sift},
    {"gist", gist},
    {"crawl", crawl},
    {"glove100", glove100},
    {"audio-dedup", audio_dedup},
    {"video-dedup", video_dedup},
    {"sift-dedup", sift_dedup},
    {"gist-dedup", gist_dedup},
    // {"flickr", flickr},
    {"deep10m", deep10m},
    // {"word2vec", word2vec},
};