// glog
#include <glog/logging.h>

// Rcpp
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
void setup(CharacterVector dir) {
    // Set log destination to somewhere we can read
    std::cout << "Logging to " <<
                 as<std::string>(dir[0]).c_str() <<
                 std::endl;
    google::SetLogDestination(google::GLOG_INFO,
                              as<std::string>(dir[0]).c_str());
    google::InitGoogleLogging("");

    // Turn off buffering so that all output is flushed
    FLAGS_logbufsecs = 0; 
    FLAGS_logbuflevel = google::GLOG_INFO;
}