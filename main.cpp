#include "detect.h"
#include "crow_all.h"
#include "base64.h"
#include <iostream>
#include <fstream>
#include <exception>
#include <utility>

using namespace std;

Detector testDetector;

int main(int argc, char* argv[]) {
    setenv("CUDA_VISIBLE_DEVICES", "", -1);
    string GRAPH = "optimized_face_detector.pb";
    int loadModelStatus = testDetector.loadModel(GRAPH);
    if (loadModelStatus < 0) {
        LOG(ERROR) << "Load model failed!";
        return -1;
    }

    // crow app
    crow::SimpleApp app;
    CROW_ROUTE(app, "/test").methods("POST"_method, "GET"_method)
    ([](const crow::request& req){
        crow::json::wvalue result;
        result["boxes"] = "[]";
        result["scores"] = "[]";
        result["labels"] = "[]";
        result["flag"] = 0;
        result["errorMsg"] = "";
        std::ostringstream os;
        try {
            auto info = crow::json::load(req.body);
            string base64_string = info["detect_img"].s();
            // read in base 64 string
            string decoded_image = base64_decode(base64_string);
            vector<uchar> data(decoded_image.begin(), decoded_image.end());
            Mat frame = imdecode(data, IMREAD_UNCHANGED);
//        cout << "Frame shape: " << frame.size() << endl;
            cvtColor(frame, frame, COLOR_BGR2RGB);
            double thresholdScore = 0.5;
            double thresholdIOU = 0.8;
            vector<float> outBoxes;
            vector<float> outScores;
            vector<size_t> outLabels;

            int detectStatus = testDetector.detect(frame, thresholdScore, thresholdIOU, outBoxes, outScores, outLabels);

            if (detectStatus < 0) {
                LOG(ERROR) << "detect failed!";
                result["errorMsg"] = "Detection Error";
                os << crow::json::dump(result);
                return crow::response(os.str());
            } else {
//                for (size_t i=0;i<outLabels.size(); i++){
//                LOG(INFO) << "label: " << outLabels[i];
//                LOG(INFO) << "score: " << outScores[i];
//                int cur_idx = i * 4;
//                LOG(INFO) << "boxes: " << outBoxes.at(cur_idx) << ","
//                          << outBoxes.at(cur_idx+1) << ","
//                          << outBoxes.at(cur_idx+2) << ","
//                          << outBoxes.at(cur_idx+3) << endl;
//
                result["boxes"] = outBoxes;
                result["scores"] = outScores;
                result["labels"] = outLabels;
                result["flag"] = 1;
                os << crow::json::dump(result);
                return crow::response{os.str()};
            }
        }
        catch (exception& e){
            LOG(ERROR) << "Unpredicted error: " << e.what();
            result["errorMsg"] = "Unknowm Error";
            os << crow::json::dump(result);
            return crow::response(os.str());
        }
    });
    app.port(8181).run();
    return 0;
}



