#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <thread>
#include <mutex>
#include <deque>
#include <unordered_map>
#include <chrono>
#include <thread>

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <nlohmann/json.hpp>  // Include nlohmann/json for JSON handling

using json = nlohmann::json;

typedef websocketpp::client<websocketpp::config::asio_client> client;
typedef websocketpp::connection_hdl connection_hdl;

std::mutex ws_mutex;  // Mutex to protect WebSocket client and handler
client c;
connection_hdl handler;

void on_open(client* c, websocketpp::connection_hdl hdl) {
    std::lock_guard<std::mutex> guard(ws_mutex);
    handler = hdl; 
    std::cout << "Connected to server." << std::endl;
}

void on_message(client* c, websocketpp::connection_hdl hdl, client::message_ptr msg) {
    std::cout << "Received message: " << msg->get_payload() << std::endl; 
}

void on_close(client* c, websocketpp::connection_hdl hdl) {
    std::cout << "Connection closed." << std::endl;
}

void send_detection_info(json& jsonData) {
    if (jsonData.empty()) {
        return;
    }
    std::string message = jsonData.dump();
    printf("Sending message: %s\n", message.c_str());

    std::lock_guard<std::mutex> guard(ws_mutex);
    c.send(handler, message, websocketpp::frame::opcode::text);
}

std::vector<std::string> load_class_list(const std::string& path) {
    std::vector<std::string> class_list;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "Error opening class names file: " << path << std::endl;
        exit(-1);
    }
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }

    for (size_t i = 0; i < class_list.size(); ++i) {
        std::cout << "Class " << i << ": " << class_list[i] << std::endl;
    }

    return class_list;
}

void load_net(cv::dnn::Net& net, const std::string& model_path, bool is_cuda) {
    auto result = cv::dnn::readNetFromONNX(model_path);
    if (result.empty()) {
        std::cerr << "Failed to load the ONNX model from path: " << model_path << std::endl;
        exit(-1);
    }

    if (is_cuda) {
        std::cout << "Attempting to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

const std::vector<cv::Scalar> colors = { cv::Scalar(128, 128, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0) };
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& classNames) {
    cv::Mat blob;
    auto input_image = format_yolov5(image);
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float* data = (float*)outputs[0].data;
    const int dimensions = 29;                  // 24 class probabilities +  5 model needs
    const int rows = 25200;     

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, classNames.size(), CV_32FC1, classes_scores);
            cv::Point class_id_point;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);
            if (max_class_score > SCORE_THRESHOLD) {
                confidences.push_back(confidence);
                int class_id = class_id_point.x;
                class_ids.push_back(class_id);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += dimensions;  // Move to the next row
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    } 
}

void communication_thread() {
    std::string uri = "ws://localhost:8080";
    //192.168.114.104

    try {
        c.set_access_channels(websocketpp::log::alevel::all);
        c.clear_access_channels(websocketpp::log::alevel::frame_payload);

        c.init_asio();

        c.set_open_handler(bind(&on_open, &c, std::placeholders::_1));
        c.set_message_handler(bind(&on_message, &c, std::placeholders::_1, std::placeholders::_2));
        c.set_close_handler(bind(&on_close, &c, std::placeholders::_1));

        websocketpp::lib::error_code ec;
        client::connection_ptr con = c.get_connection(uri, ec);
        if (ec) {
            std::cout << "Could not create connection because: " << ec.message() << std::endl;
            return;
        }

        c.connect(con);
        c.run();
    }
    catch (websocketpp::exception const& e) {
        std::cerr << "WebSocket++ exception: " << e.what() << std::endl;
    }
    catch (std::exception const& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown exception" << std::endl;
    }
}

int id_transform(int input_id) {        // Function to transform the detected class ID to the corresponding card ID
  
        
    if (input_id == 0) return 10;
    if (input_id == 1) return 15;
    if (input_id == 2) return 1; 
    if (input_id == 3) return 21;
    if (input_id == 4) return 23;
    if (input_id == 5) return 13;
    if (input_id == 6) return 2;
    if (input_id == 7) return 7;
    if (input_id == 8) return 9; 
    if (input_id == 9) return 5; 
    if (input_id == 10) return 22;
    if (input_id == 11) return 12; 
    if (input_id == 12) return 19;
    if (input_id == 13) return 6;
    if (input_id == 14) return 17;
    if (input_id == 15) return 3; 
    if (input_id == 16) return 20;
    if (input_id == 17) return 11;
    if (input_id == 18) return 16;
    if (input_id == 19) return 14; 
    if (input_id == 20) return 24;
    if (input_id == 21) return 8;
    if (input_id == 22) return 4; 
    if (input_id == 23) return 18; 

    printf("input id is not valid");
    return -1;
}


int main(int argc, char** argv) {

    std::thread t1(communication_thread);  // Create a thread to handle the communication with the server

    std::string class_file_path = "C:\\Users\\erent\\source\\repos\\cson123\\cson123\\data\\coco.names";
    std::string model_path = "C:\\Users\\erent\\source\\repos\\cson123\\cson123\\yolov5s.onnx";

    std::vector<std::string> class_list = load_class_list(class_file_path);

    cv::Mat frame;
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        std::cerr << "Error opening video capture\n";
        return -1;
    }

    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
    cv::dnn::Net net;
    load_net(net, model_path, is_cuda);

    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;

    std::deque<json> frameBuffer;  // Buffer to hold the last 10 frames
    const int FRAME_BUFFER_SIZE = 10;
    const int CONSISTENCY_THRESHOLD = 7;

    while (true) {  // Traverses frames
        capture.read(frame);
        if (frame.empty()) {
            std::cout << "End of stream\n";
            break;
        }

        std::vector<Detection> output;
        detect(frame, net, output, class_list);

        frame_count++;
        total_frames++;
        json outputJson;
        outputJson["Array"] = json::array();
        for (const auto& detection : output) {  // Traverses cards
            const auto& box = detection.box;
            int classId = detection.class_id;
            const auto& color = colors[classId % colors.size()];
            cv::rectangle(frame, box, color, 2);
            cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);

            std::string label = "Unknown";
            if (classId >= 0 && classId < class_list.size()) {
                label = class_list[classId] + " " + std::to_string(detection.confidence).substr(0, 4);
            }
            cv::putText(frame, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);

            // Take id x and y to to a push json array
            json detectionJson;
            detectionJson["pos_x"] = box.x;
            detectionJson["pos_y"] = box.y;
            detectionJson["id"] = id_transform(classId);

            outputJson["Array"].push_back(detectionJson);
        }

        frameBuffer.push_back(outputJson);  // Add new frame to the buffer

        if (frameBuffer.size() > FRAME_BUFFER_SIZE) {
            frameBuffer.pop_front();  // Remove oldest frame if buffer exceeds size
        }

        // Count occurrences of each card ID across the buffer
        std::unordered_map<int, std::pair<int, cv::Point>> cardIdCount;
        for (const auto& frame : frameBuffer) {
            for (const auto& detection : frame["Array"]) {
                int cardId = detection["id"];
                cv::Point pos(detection["pos_x"], detection["pos_y"]);
                if (cardIdCount.find(cardId) != cardIdCount.end()) {
                    cardIdCount[cardId].first++;
                    // Update position to average (if needed)
                    cardIdCount[cardId].second = pos;
                }
                else {
                    cardIdCount[cardId] = std::make_pair(1, pos);
                }
            }
        }

        // Find card IDs that are consistent in at least 7 frames
        json consistentJson;
        consistentJson["Array"] = json::array();
        for (const auto& entry : cardIdCount) {
            if (entry.second.first >= CONSISTENCY_THRESHOLD) {
                json detectionJson;
                detectionJson["id"] = entry.first;
                detectionJson["pos_x"] = entry.second.second.x;
                detectionJson["pos_y"] = entry.second.second.y;
                consistentJson["Array"].push_back(detectionJson);
            }
        }

        
          send_detection_info(consistentJson);      // Send the consistent card IDs to the server
        

        // print fps every frame 
          auto end = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> elapsed = end - start;
          if (elapsed.count() > 1.0) {
			  fps = frame_count / elapsed.count();
			  frame_count = 0;
			  start = end;
		  }
          cv::putText(frame, "FPS: " + std::to_string(fps), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);



        cv::imshow("output", frame);
        if (cv::waitKey(1) != -1) {
            break;
        }

        //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    capture.release();

    if (t1.joinable())
        t1.join();  // Ensure the communication thread is joined before exiting main

    return 0;
}
