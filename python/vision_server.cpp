// vision_server.cpp — No Haar, only YOLO body detection
#include <iostream>
#include <sstream>
#include <fstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstring>

#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "libobsensor/ObSensor.hpp"

#define SOCKET_PATH   "/tmp/medpal.sock"
#define HTTP_PORT     8081
#define COLOR_W       1280
#define COLOR_H       720
#define DEPTH_W       640
#define DEPTH_H       480
#define YOLO_SIZE     320
#define FRAME_SKIP    10
#define STREAM_FPS    15
#define JPEG_QUALITY  60
#define FOLLOW_STOP   0.20f
#define CENTER_TOL    0.15f

const std::string YOLO_MODEL = "/home/medpal/MedPalRobotV2/models/yolov8n.onnx";

// Shared state
cv::Mat           g_depth;
std::mutex        g_depth_mtx;
std::atomic<bool> g_depth_new{false};
std::atomic<bool> g_running{true};
std::atomic<bool> g_following{false};
std::string       g_target = "anyone";
std::mutex        g_target_mtx;

std::vector<uchar> g_jpeg;
std::mutex         g_jpeg_mtx;
std::string        g_status_json = "{}";
std::mutex         g_status_mtx;
std::string        g_motor_cmd = "stop";
std::mutex         g_motor_mtx;

int cpu_temp() {
    std::ifstream f("/sys/class/thermal/thermal_zone0/temp");
    int v = 0;
    if (f) f >> v;
    return v / 1000;
}

bool detect_coral() {
    std::ifstream f("/proc/bus/usb/devices");
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("1a6e") != std::string::npos ||
            line.find("18d1") != std::string::npos)
            return true;
    }
    return false;
}

void depth_thread(ob::Pipeline& pipe) {
    std::cout << "[Depth] Thread started.\n";
    while (g_running) {
        try {
            auto fs = pipe.waitForFrames(200);
            if (!fs) continue;
            auto df = fs->depthFrame();
            if (!df) continue;
            int w = df->width(), h = df->height();
            float scale = df->getValueScale();
            cv::Mat raw(h, w, CV_16UC1, (void*)df->data());
            cv::Mat mm;
            raw.convertTo(mm, CV_16UC1, scale);
            std::lock_guard<std::mutex> lk(g_depth_mtx);
            g_depth = mm.clone();
            g_depth_new = true;
        } catch (...) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    std::cout << "[Depth] Thread stopped.\n";
}

float sample_depth(int cx, int cy) {
    std::lock_guard<std::mutex> lk(g_depth_mtx);
    if (g_depth.empty()) return -1.f;
    int dcx = std::clamp((int)(cx * (float)DEPTH_W / COLOR_W), 2, DEPTH_W-3);
    int dcy = std::clamp((int)(cy * (float)DEPTH_H / COLOR_H), 2, DEPTH_H-3);
    auto roi = g_depth(cv::Rect(dcx-2, dcy-2, 5, 5));
    float sum = 0; int cnt = 0;
    for (int r = 0; r < roi.rows; r++)
        for (int c = 0; c < roi.cols; c++) {
            uint16_t v = roi.at<uint16_t>(r, c);
            if (v > 0) { sum += v; cnt++; }
        }
    return cnt > 0 ? sum / cnt / 1000.f : -1.f;
}

void http_stream_thread() {
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(HTTP_PORT);
    bind(srv, (struct sockaddr*)&addr, sizeof(addr));
    listen(srv, 5);
    std::cout << "[HTTP] MJPEG stream on port " << HTTP_PORT << "\n";

    while (g_running) {
        struct sockaddr_in cli{};
        socklen_t cli_len = sizeof(cli);
        int fd = accept(srv, (struct sockaddr*)&cli, &cli_len);
        if (fd < 0) continue;

        char buf[1024] = {};
        recv(fd, buf, sizeof(buf)-1, 0);
        std::string req(buf);

        if (req.find("GET /frame") != std::string::npos) {
            std::vector<uchar> jpg;
            { std::lock_guard<std::mutex> lk(g_jpeg_mtx); jpg = g_jpeg; }
            std::string hdr = "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\n"
                              "Content-Length: " + std::to_string(jpg.size()) +
                              "\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
            send(fd, hdr.c_str(), hdr.size(), 0);
            send(fd, jpg.data(), jpg.size(), 0);
            close(fd);
        } else {
            std::string hdr = "HTTP/1.1 200 OK\r\n"
                              "Content-Type: multipart/x-mixed-replace;boundary=frame\r\n"
                              "Access-Control-Allow-Origin: *\r\n\r\n";
            send(fd, hdr.c_str(), hdr.size(), 0);
            float interval = 1.f / STREAM_FPS;
            while (g_running) {
                auto t0 = std::chrono::steady_clock::now();
                std::vector<uchar> jpg;
                { std::lock_guard<std::mutex> lk(g_jpeg_mtx); jpg = g_jpeg; }
                std::string part = "--frame\r\nContent-Type: image/jpeg\r\n"
                                   "Content-Length: " + std::to_string(jpg.size()) + "\r\n\r\n";
                if (send(fd, part.c_str(), part.size(), MSG_NOSIGNAL) < 0) break;
                if (send(fd, jpg.data(), jpg.size(), MSG_NOSIGNAL) < 0) break;
                if (send(fd, "\r\n", 2, MSG_NOSIGNAL) < 0) break;
                float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
                float sl = interval - elapsed;
                if (sl > 0) std::this_thread::sleep_for(std::chrono::duration<float>(sl));
            }
            close(fd);
        }
    }
    close(srv);
}

void socket_thread() {
    unlink(SOCKET_PATH);
    int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path)-1);
    bind(srv, (struct sockaddr*)&addr, sizeof(addr));
    listen(srv, 1);
    std::cout << "[Socket] Waiting for Python...\n";
    int cli = accept(srv, nullptr, nullptr);
    std::cout << "[Socket] Python connected.\n";

    while (g_running) {
        std::string json;
        { std::lock_guard<std::mutex> lk(g_status_mtx); json = g_status_json + "\n"; }
        if (send(cli, json.c_str(), json.size(), MSG_NOSIGNAL) < 0) break;

        char buf[512] = {};
        int n = recv(cli, buf, sizeof(buf)-1, MSG_DONTWAIT);
        if (n > 0) {
            std::string cmd(buf, n);
            if (cmd.find("follow:on") != std::string::npos) {
                g_following = true;
                std::cout << "[Socket] Follow ON\n";
            }
            if (cmd.find("follow:off") != std::string::npos) {
                g_following = false;
                std::cout << "[Socket] Follow OFF\n";
            }
            if (cmd.find("target:") != std::string::npos) {
                size_t pos = cmd.find("target:") + 7;
                std::string t = cmd.substr(pos);
                t.erase(std::remove_if(t.begin(), t.end(), [](char c){ return c=='\n'||c=='\r'||c==' '; }), t.end());
                std::lock_guard<std::mutex> lk(g_target_mtx);
                g_target = t;
                std::cout << "[Socket] Target → " << t << "\n";
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    close(cli);
    close(srv);
    unlink(SOCKET_PATH);
}

int main() {
    std::cout << "=== MedPal C++ Vision Server ===\n";
    bool coral = detect_coral();
    std::cout << "[Main] Coral TPU: " << (coral ? "YES" : "NO") << "\n";

    // Depth pipeline
    std::cout << "[Main] Starting depth pipeline...\n";
    ob::Pipeline pipe;
    auto profiles = pipe.getStreamProfileList(OB_SENSOR_DEPTH);
    auto dpro = profiles->getVideoStreamProfile(DEPTH_W, DEPTH_H, OB_FORMAT_Y11, 30);
    auto cfg = std::make_shared<ob::Config>();
    cfg->enableStream(dpro);
    pipe.start(cfg);
    std::thread dt(depth_thread, std::ref(pipe));

    // Color camera
    std::cout << "[Main] Opening color camera...\n";
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, COLOR_W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, COLOR_H);
    cap.set(cv::CAP_PROP_FPS, 30);
    std::cout << "[Main] Color: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n";

    // YOLO
    std::cout << "[Main] Loading YOLO: " << YOLO_MODEL << "\n";
    cv::dnn::Net net = cv::dnn::readNetFromONNX(YOLO_MODEL);
    if (coral) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_NPU);
    } else {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        cv::setNumThreads(2);
    }

    // Start HTTP and socket threads
    std::thread http_t(http_stream_thread);
    std::thread sock_t(socket_thread);

    // Placeholder frame
    cv::Mat placeholder(COLOR_H, COLOR_W, CV_8UC3, cv::Scalar(8,17,31));
    cv::putText(placeholder, "MedPal Vision Starting...", {380,360},
                cv::FONT_HERSHEY_SIMPLEX, 1.2, {0,200,255}, 2);
    std::vector<uchar> ph_buf;
    cv::imencode(".jpg", placeholder, ph_buf, {cv::IMWRITE_JPEG_QUALITY, 60});
    { std::lock_guard<std::mutex> lk(g_jpeg_mtx); g_jpeg = ph_buf; }

    int fc = 0;
    float f_int = 1.f / STREAM_FPS;
    std::vector<cv::Rect> last_bodies;
    cv::Mat depth_snap;

    std::cout << "[Main] Vision loop running.\n";

    while (g_running) {
        auto t0 = std::chrono::steady_clock::now();

        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        fc++;

        if (g_depth_new) {
            std::lock_guard<std::mutex> lk(g_depth_mtx);
            depth_snap = g_depth.clone();
            g_depth_new = false;
        }

        // YOLO detection every FRAME_SKIP frames
        if (fc % FRAME_SKIP == 0) {
            cv::Mat small, blob;
            cv::resize(frame, small, {YOLO_SIZE, YOLO_SIZE});
            cv::dnn::blobFromImage(small, blob, 1./255., {YOLO_SIZE,YOLO_SIZE},
                                    {0,0,0}, true, false);
            net.setInput(blob);
            std::vector<cv::Mat> outs;
            net.forward(outs, net.getUnconnectedOutLayersNames());

            cv::Mat out = outs[0].reshape(1, outs[0].size[1]);
            cv::transpose(out, out);
            float sx = (float)COLOR_W/YOLO_SIZE, sy = (float)COLOR_H/YOLO_SIZE;
            std::vector<cv::Rect> boxes;
            std::vector<float> scores;
            for (int i = 0; i < out.rows; i++) {
                float* r = out.ptr<float>(i);
                if (r[4] < 0.4f) continue;
                float cx = r[0]*sx, cy = r[1]*sy, w = r[2]*sx, h = r[3]*sy;
                boxes.push_back({(int)(cx-w/2), (int)(cy-h/2), (int)w, (int)h});
                scores.push_back(r[4]);
            }
            std::vector<int> idx;
            cv::dnn::NMSBoxes(boxes, scores, 0.4f, 0.45f, idx);
            last_bodies.clear();
            for (int i : idx) last_bodies.push_back(boxes[i]);
        }

        // Follow logic (no face detection)
        std::string action = "IDLE";
        float dist_m = -1;
        int dot_x = COLOR_W/2, dot_y = COLOR_H/2;

        if (!last_bodies.empty()) {
            // Pick largest body
            auto& rb = *std::max_element(last_bodies.begin(), last_bodies.end(),
                                         [](auto& a, auto& b){ return a.area() < b.area(); });
            cv::Rect r = rb & cv::Rect(0,0,COLOR_W,COLOR_H);
            if (!r.empty()) {
                dot_x = r.x + r.width/2;
                dot_y = r.y + r.height/2;
                dist_m = sample_depth(dot_x, dot_y);

                // Draw
                cv::rectangle(frame, r, {0,255,0}, 2);
                cv::circle(frame, {dot_x, dot_y}, 6, {0,0,255}, -1);
                cv::line(frame, {dot_x-12, dot_y}, {dot_x+12, dot_y}, {255,255,255}, 1);
                cv::line(frame, {dot_x, dot_y-12}, {dot_x, dot_y+12}, {255,255,255}, 1);
                char dbuf[32];
                if (dist_m > 0) snprintf(dbuf, sizeof(dbuf), "%.2fm", dist_m);
                else snprintf(dbuf, sizeof(dbuf), "?");
                cv::putText(frame, dbuf, {dot_x+10, dot_y-8}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,200,255}, 2);

                if (g_following) {
                    if (dist_m > 0 && dist_m <= FOLLOW_STOP) action = "TOO CLOSE";
                    else if (dist_m < 0) action = "NO DEPTH";
                    else {
                        float frame_cx = COLOR_W / 2.f;
                        float person_cx = r.x + r.width/2.f;
                        float offset = (person_cx - frame_cx) / frame_cx;
                        if (std::abs(offset) > CENTER_TOL)
                            action = offset > 0 ? "TURNING RIGHT" : "TURNING LEFT";
                        else
                            action = "FOLLOWING";
                    }
                }
            }
        } else {
            if (g_following) action = "NO PERSON";
        }

        // Motor command
        {
            std::string mcmd = "stop";
            if (action == "FOLLOWING") mcmd = "forward";
            else if (action == "TURNING RIGHT") mcmd = "right";
            else if (action == "TURNING LEFT") mcmd = "left";
            std::lock_guard<std::mutex> lk(g_motor_mtx);
            g_motor_cmd = mcmd;
        }

        // HUD
        int temp = cpu_temp();
        cv::putText(frame, "P:" + std::to_string((int)last_bodies.size()) +
                    " | " + action + " | " + std::to_string(temp) + "C",
                    {10,40}, cv::FONT_HERSHEY_SIMPLEX, 0.9, {0,255,255}, 2);
        cv::putText(frame, std::string("Follow:") + (g_following ? "ON" : "OFF") +
                    (coral ? " [CORAL]" : " [CPU]"),
                    {10,80}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    g_following ? cv::Scalar{0,255,0} : cv::Scalar{100,100,255}, 2);

        // Encode JPEG
        std::vector<uchar> buf;
        cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, JPEG_QUALITY});
        { std::lock_guard<std::mutex> lk(g_jpeg_mtx); g_jpeg = buf; }

        // Status JSON
        {
            std::string tgt;
            { std::lock_guard<std::mutex> lk(g_target_mtx); tgt = g_target; }
            std::string mcmd;
            { std::lock_guard<std::mutex> lk(g_motor_mtx); mcmd = g_motor_cmd; }
            std::ostringstream js;
            js << "{\"persons\":" << last_bodies.size()
               << ",\"distance\":" << (dist_m > 0 ? dist_m : 0.f)
               << ",\"action\":\"" << action << "\""
               << ",\"following\":" << (g_following ? "true":"false")
               << ",\"temp\":" << temp
               << ",\"coral\":" << (coral ? "true":"false")
               << ",\"target\":\"" << tgt << "\""
               << ",\"motor\":\"" << mcmd << "\"}";
            std::lock_guard<std::mutex> lk(g_status_mtx);
            g_status_json = js.str();
        }

        // Frame pacing
        float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
        float sl = f_int - elapsed;
        if (sl > 0) std::this_thread::sleep_for(std::chrono::duration<float>(sl));
    }

    std::cout << "[Main] Stopping...\n";
    g_running = false;
    dt.join();
    http_t.detach();
    sock_t.detach();
    pipe.stop();
    cap.release();
    std::cout << "[Main] Done.\n";
    return 0;
}
