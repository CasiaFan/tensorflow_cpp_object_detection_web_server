#include "detect.h"
#include <fstream>
#include <utility>
#include <iostream>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace std;

Status loadGraph(const string &graph_file_name,
                 unique_ptr<tensorflow::Session> *session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}


/** Convert Mat image into tensor of shape (1, height, width, d) where last three dims are equal to the original dims.
TODO: support batch mat input
 */
Tensor readTensorFromMat(const Mat &mat) {
      int height = mat.rows;
      int width = mat.cols;
      int depth = mat.channels();
      Tensor inputTensor(tensorflow::DT_UINT8, tensorflow::TensorShape({1, height, width, depth}));
      auto inputTensorMapped = inputTensor.tensor<tensorflow::uint8, 4>();

      cv::Mat frame;
      mat.convertTo(frame, CV_8UC3);
      const tensorflow::uint8* source_data = (tensorflow::uint8*)frame.data;
      for (int y=0; y<height; y++){
        const tensorflow::uint8* source_row = source_data + (y*width*depth);
        for (int x=0; x<width; x++){
            const tensorflow::uint8* source_pixel = source_row + (x*depth);
            for (int c=0; c<depth; c++){
                const tensorflow::uint8* source_value = source_pixel + c;
                inputTensorMapped(0, y, x, c) = *source_value;
            }
        }
      }
      return inputTensor;
}

double IOU(Rect2f box1, Rect2f box2) {

    float xA = max(box1.tl().x, box2.tl().x);
    float yA = max(box1.tl().y, box2.tl().y);
    float xB = min(box1.br().x, box2.br().x);
    float yB = min(box1.br().y, box2.br().y);

    float intersectArea = abs((xB - xA) * (yB - yA));
    float unionArea = abs(box1.area()) + abs(box2.area()) - intersectArea;

    return 1. * intersectArea / unionArea;
}

/** Return idxs of good boxes (ones with highest confidence score (>= thresholdScore)
 *  and IOU <= thresholdIOU with others).
 */
vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &scores,
                           tensorflow::TTypes<float, 3>::Tensor &boxes,
                           double thresholdScore,
                           double thresholdIOU) {

    vector<size_t> sortIdxs(scores.size());
    iota(sortIdxs.begin(), sortIdxs.end(), 0);

    // Create set of "bad" idxs
    set<size_t> badIdxs = set<size_t> ();
    size_t i = 0;
    while (i < sortIdxs.size()) {
        // get bad idx with low score
        if (scores(sortIdxs.at(i)) < thresholdScore){
            badIdxs.insert(sortIdxs.at(i));
        }
        if (badIdxs.find(sortIdxs.at(i)) != badIdxs.end()){
            i++;
            continue;
        }
        // get bad idx with high iou
        Rect2f box1 = Rect2f(Point2f(boxes(0, sortIdxs.at(i), 1), boxes(0, sortIdxs.at(i), 0)),
                             Point2f(boxes(0, sortIdxs.at(i), 3), boxes(0, sortIdxs.at(i), 2)));
        for (size_t j=i+1; j<sortIdxs.size(); j++){
            if (scores(sortIdxs.at(j)) < thresholdScore) {
                badIdxs.insert(sortIdxs.at(j));
                continue;
            }
            Rect2f box2 = Rect2f(Point2f(boxes(0, sortIdxs.at(j), 1), boxes(0, sortIdxs.at(j), 0)),
                                 Point2f(boxes(0, sortIdxs.at(j), 3), boxes(0, sortIdxs.at(j), 2)));
            if (IOU(box1, box2) > thresholdIOU)
                badIdxs.insert(sortIdxs.at(j));
        }
        i++;
    }

    vector<size_t> goodIdxs = vector<size_t>();
    for (auto it=sortIdxs.begin(); it!=sortIdxs.end(); ++it){
        if (badIdxs.find(sortIdxs.at(*it)) == badIdxs.end())
            goodIdxs.push_back(sortIdxs.at(*it));
    }
    return goodIdxs;
}


int Detector::loadModel(string modelPath){
    Status loadGraphStatus = loadGraph(modelPath, &(this->session));
    if (!loadGraphStatus.ok()) {
        LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
        return -1;
    } else {
        LOG(INFO) << "loadGraph(): Model Loaded" << endl;
        return 0;
    }
}

int Detector::detect(Mat frame, double thresholdScore, double thresholdIOU, vector<float> &outBoxes, vector<float> &outScores, vector<size_t> &outLabels) {
    // convert mat to tensor
    int width = frame.cols;
    int height = frame.rows;
    tensorflow::TensorShape shape = tensorflow::TensorShape();
    shape.AddDim(1);
    shape.AddDim(height);
    shape.AddDim(width);
    shape.AddDim(3);
    Tensor inputTensor;
    inputTensor = readTensorFromMat(frame);
    // run graph on tensor
    string inputLayer = "image_tensor:0";
    vector<string> outputLayer = {"detection_boxes:0", "detection_scores:0", "detection_classes:0"};
    outBoxes.clear();
    outScores.clear();
    outLabels.clear();
    vector<Tensor> outputs;
    Status runStatus = this->session->Run({{inputLayer, inputTensor}}, outputLayer, {}, &outputs);
    if (!runStatus.ok()) {
        LOG(ERROR) << "Running model failed: " << runStatus;
    } else {
        LOG(INFO) << "Running graph done!";
    }

    // extract results
    tensorflow::TTypes<float, 3>::Tensor boxes = outputs[0].flat_outer_dims<float, 3>();
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat labels = outputs[2].flat<float>();
    vector<size_t> goodIdxs = filterBoxes(scores, boxes, thresholdScore, thresholdIOU);
    for(size_t i=0; i!=goodIdxs.size(); i++)
    {
        outScores.push_back(scores(goodIdxs.at(i)));
        outLabels.push_back(labels(goodIdxs.at(i)));
        outBoxes.push_back(boxes(0, goodIdxs.at(i), 0) * height);
        outBoxes.push_back(boxes(0, goodIdxs.at(i), 1) * width);
        outBoxes.push_back(boxes(0, goodIdxs.at(i), 2) * height);
        outBoxes.push_back(boxes(0, goodIdxs.at(i), 3) * width);
    }
    LOG(INFO) << "outBoxes info: " << outBoxes.size();
    return 0;
}

