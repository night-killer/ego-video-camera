#include <System.h>

#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct FrameRow {
  long long frame_id;
  double timestamp;
  std::string image_path;
};

static std::vector<FrameRow> load_manifest(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open frame manifest: " + path);
  std::vector<FrameRow> rows;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::istringstream stream(line);
    std::string frame_id, timestamp, image_path;
    if (!std::getline(stream, frame_id, '\t') ||
        !std::getline(stream, timestamp, '\t') ||
        !std::getline(stream, image_path)) {
      throw std::runtime_error("Malformed frame manifest row");
    }
    rows.push_back({std::stoll(frame_id), std::stod(timestamp), image_path});
  }
  return rows;
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: orb_slam3_monocular VOCAB SETTINGS FRAMES_TSV OUTPUT_TSV\n";
    return 2;
  }
  try {
    const auto rows = load_manifest(argv[3]);
    ORB_SLAM3::System slam(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, false);
    std::ofstream output(argv[4]);
    if (!output) throw std::runtime_error("Cannot open output trajectory");
    output << "frame_id\ttimestamp_sec\tvalid\ttracking_state\ttracked_points";
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c) output << "\tm" << r << c;
    output << '\n' << std::setprecision(17);

    for (const auto& row : rows) {
      const cv::Mat image = cv::imread(row.image_path, cv::IMREAD_UNCHANGED);
      if (image.empty()) throw std::runtime_error("Cannot read image: " + row.image_path);
      const Sophus::SE3f tcw = slam.TrackMonocular(image, row.timestamp);
      const int state = slam.GetTrackingState();
      const std::size_t tracked_points = slam.GetTrackedMapPoints().size();
      const bool valid = state == 2 || state == 5;
      const Eigen::Matrix4f c2w = tcw.inverse().matrix();
      output << row.frame_id << '\t' << row.timestamp << '\t' << (valid ? 1 : 0)
             << '\t' << state << '\t' << tracked_points;
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          const float value = c2w(r, c);
          output << '\t' << (std::isfinite(value) ? value : 0.0f);
        }
      }
      output << '\n';
    }
    slam.Shutdown();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
