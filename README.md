This is an example how to run a TensorFlow object detection model as web server in TensorFlow C++ API.

### Brief Introduction
1). Construct a class for TF model like this:
```c++
class Detector {
    std::unique_ptr<tensorflow::Session> session;
    public:
        int loadModel();
        int detect();
};
``` 
First, `loadModel` to initialize and load graph into session, then use 
session in `detect` for prediction.

2). use crow the start a web server in `main` (crow is inspired by python `FLASK`. 
If you are familiar with FLASK, crow is easy to use.)

### Requirements
- [TensorFlow 1.8.0rc1](https://github.com/tensorflow)
- [OpenCV 3.4.0](https://opencv.org/releases.html)
- Boost 1.5.8
- Ubuntu 16.04

### Install 
**Install TensorFlow C++ and OpenCV**: see this [blog](https://medium.com/@fanzongshaoxing/use-tensorflow-c-api-with-opencv3-bacb83ca5683)

**Install Boost** <br>
```bash
sudo apt-get install libboost-all-dev
```

### Usage
1. compile the project
```bash
cmake .
make
```
2. run tf-cpp web service
```bash
./tf_detect_crow
```

3. test with python script
``` 
python test_cpp_api.py
```

### Acknowledgement
Great appreciation to following project and code snippet: 

- [opencv_tensor.cc](https://gist.github.com/kyrs/9adf86366e9e4f04addb)
- [tensorflow object detection cpp](https://github.com/lysukhin/tensorflow-object-detection-cpp)
- [crow example](https://github.com/jolks/crow-template/blob/master/src/example.cpp)
- [Encoding and decoding base 64 with c++](https://renenyffenegger.ch/notes/development/Base64/Encoding-and-decoding-base-64-with-cpp)