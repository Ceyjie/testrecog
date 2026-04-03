// vision_server.cpp
// Handles: YOLO body detect, Haar face, depth, person ID, motor control
// Sends status JSON to Python via Unix socket
// Coral TPU auto-detected if plugged in

#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <fstream>
#include <sstream>
#include <chrono>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// OrbbecSDK
#include "libobsensor/ObSensor.hpp"

// ── Config ────────────────────────────────────────────────
#define SOCKET_PATH     "/tmp/medpal.sock"
#define COLOR_W         1280
#define COLOR_H         720
#define DEPTH_W         640
#define DEPTH_H         480
#define YOLO_SIZE       320
#define FRAME_SKIP      10
#define STREAM_FPS      15
#define JPEG_QUALITY    60
#define FOLLOW_STOP     0.20f
#define CENTER_TOL      0.15f
#define FOLLOW_SPEED    0.5f
#define TURN_SPEED      0.4f
#define YOLO_MODEL      "/home/medpal/MedPalRobotV2/models/yolov8n.onnx"
#define FACE_MODEL      "models/face.xml"
#define PERSON_DB       "data/persons/"

// ── GPIO (gpiozero via Python preferred; stub here) ───────
// Motor commands sent as JSON to Python which calls gpiozero

// ── Shared state ──────────────────────────────────────────
cv::Mat           g_depth;
std::mutex        g_depth_mtx;
std::atomic<bool> g_depth_new{false};
std::atomic<bool> g_running{true};
std::atomic<bool> g_following{false};
std::string       g_target_id = "";  // locked person ID

// Latest JPEG frame for streaming
std::vector<uchar> g_jpeg;
std::mutex         g_jpeg_mtx;

// Latest status JSON
std::string g_status_json = "{}";
std::mutex  g_status_mtx;

// ── Depth thread ──────────────────────────────────────────
void depth_thread(ob::Pipeline& pipe) {
    while (g_running) {
        auto fs = pipe.waitForFrames(200);
        if (!fs) continue;
        auto df = fs->depthFrame();
        if (!df) continue;
        int w = df->width(), h = df->height();
        float scale = df->getValueScale();
        cv::Mat raw(h, w, CV_16UC1, (void*)df->data());
        cv::Mat mm; raw.convertTo(mm, CV_16UC1, scale);
        std::lock_guard<std::mutex> lk(g_depth_mtx);
        g_depth = mm.clone();
        g_depth_new = true;
    }
}

// ── Sample depth at point ─────────────────────────────────
float sample_depth(int cx, int cy) {
    std::lock_guard<std::mutex> lk(g_depth_mtx);
    if (g_depth.empty()) return -1;
    int dcx = std::clamp((int)(cx*(float)DEPTH_W/COLOR_W), 2, DEPTH_W-3);
    int dcy = std::clamp((int)(cy*(float)DEPTH_H/COLOR_H), 2, DEPTH_H-3);
    auto roi = g_depth(cv::Rect(dcx-2, dcy-2, 5, 5));
    float sum=0; int cnt=0;
    for (int r=0;r<roi.rows;r++)
        for (int c=0;c<roi.cols;c++){
            uint16_t v=roi.at<uint16_t>(r,c);
            if(v>0){sum+=v;cnt++;}
        }
    return cnt>0 ? sum/cnt/1000.f : -1;
}

// ── CPU temp ──────────────────────────────────────────────
int cpu_temp() {
    std::ifstream f("/sys/class/thermal/thermal_zone0/temp");
    int v=0; f>>v; return v/1000;
}

// ── Check Coral USB ───────────────────────────────────────
bool coral_available() {
    // Check for EdgeTPU device
    return system("lsusb | grep -q '1a6e\\|18d1' 2>/dev/null") == 0;
}

// ── Socket server — sends JSON to Python ─────────────────
int g_sock_fd = -1;
int g_client_fd = -1;

void socket_thread() {
    unlink(SOCKET_PATH);
    g_sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path)-1);
    bind(g_sock_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(g_sock_fd, 1);
    std::cout << "[Socket] Waiting for Python connection...\n";
    g_client_fd = accept(g_sock_fd, nullptr, nullptr);
    std::cout << "[Socket] Python connected.\n";

    while (g_running) {
        std::string json;
        {
            std::lock_guard<std::mutex> lk(g_status_mtx);
            json = g_status_json + "\n";
        }
        if (send(g_client_fd, json.c_str(), json.size(), MSG_NOSIGNAL) < 0)
            break;
        // Read commands from Python
        char buf[256] = {};
        int n = recv(g_client_fd, buf, sizeof(buf)-1, MSG_DONTWAIT);
        if (n > 0) {
            std::string cmd(buf, n);
            if (cmd.find("follow:on") != std::string::npos)
                g_following = true;
            else if (cmd.find("follow:off") != std::string::npos)
                g_following = false;
            else if (cmd.find("target:") != std::string::npos)
                g_target_id = cmd.substr(7);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// ── Main vision loop ──────────────────────────────────────
int main() {
    std::cout << "[MedPal C++] Starting...\n";
    bool coral = coral_available();
    std::cout << "[MedPal C++] Coral TPU: " << (coral?"YES":"NO") << "\n";

    // Depth pipeline
    ob::Pipeline pipe;
    auto profiles = pipe.getStreamProfileList(OB_SENSOR_DEPTH);
    auto dpro = profiles->getVideoStreamProfile(DEPTH_W,DEPTH_H,OB_FORMAT_Y11,30);
    auto cfg = std::make_shared<ob::Config>();
    cfg->enableStream(dpro);
    pipe.start(cfg);
    std::thread dt(depth_thread, std::ref(pipe));

    // Color camera
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  COLOR_W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, COLOR_H);
    cap.set(cv::CAP_PROP_FPS, 30);

    // Load YOLO (Coral if available, else CPU)
    cv::dnn::Net bodyNet = cv::dnn::readNetFromONNX(YOLO_MODEL);
    if (coral) {
        std::cout << "[MedPal C++] Using Coral EdgeTPU backend\n";
        bodyNet.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        bodyNet.setPreferableTarget(cv::dnn::DNN_TARGET_NPU);
    } else {
        bodyNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        bodyNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        cv::setNumThreads(2);
    }

    // Haar face
    cv::CascadeClassifier haar;
    haar.load(FACE_MODEL);

    // Socket thread (communicate with Python)
    std::thread st(socket_thread);

    int fc=0;
    float frame_interval = 1.0f/STREAM_FPS;
    std::vector<cv::Rect> last_bodies;
    std::string last_action;

    std::cout << "[MedPal C++] Vision loop running.\n";

    while (g_running) {
        auto t0 = std::chrono::steady_clock::now();
        cv::Mat frame;
        if (!cap.read(frame)||frame.empty()){
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        fc++;

        // YOLO every FRAME_SKIP frames
        if (fc % FRAME_SKIP == 0) {
            cv::Mat small, blob;
            cv::resize(frame, small, {YOLO_SIZE, YOLO_SIZE});
            cv::dnn::blobFromImage(small,blob,1./255.,{YOLO_SIZE,YOLO_SIZE},
                                   {0,0,0},true,false);
            bodyNet.setInput(blob);
            std::vector<cv::Mat> outs;
            bodyNet.forward(outs, bodyNet.getUnconnectedOutLayersNames());
            cv::Mat out = outs[0].reshape(1, outs[0].size[1]);
            cv::transpose(out, out);
            float sx=(float)COLOR_W/YOLO_SIZE, sy=(float)COLOR_H/YOLO_SIZE;
            std::vector<cv::Rect> boxes; std::vector<float> scores;
            for(int i=0;i<out.rows;i++){
                float* r=out.ptr<float>(i);
                if(r[4]<0.4f) continue;
                float cx=r[0]*sx,cy=r[1]*sy,w=r[2]*sx,h=r[3]*sy;
                boxes.push_back({(int)(cx-w/2),(int)(cy-h/2),(int)w,(int)h});
                scores.push_back(r[4]);
            }
            std::vector<int> idx;
            cv::dnn::NMSBoxes(boxes,scores,0.4f,0.45f,idx);
            last_bodies.clear();
            for(int i:idx) last_bodies.push_back(boxes[i]);
        }

        // Haar faces
        cv::Mat small_h, gray;
        cv::resize(frame, small_h, {320,180});
        cv::cvtColor(small_h, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);
        std::vector<cv::Rect> faces_s;
        haar.detectMultiScale(gray,faces_s,1.1,4,0,{20,20},{150,150});
        float hsx=COLOR_W/320.f, hsy=COLOR_H/180.f;

        // Follow logic
        std::string action="IDLE";
        float dist_m=-1;
        int dot_x=0, dot_y=0;

        if (!last_bodies.empty()) {
            auto& r = *std::max_element(last_bodies.begin(),last_bodies.end(),
                [](auto& a,auto& b){return a.area()<b.area();});
            cv::Rect rb = r & cv::Rect(0,0,COLOR_W,COLOR_H);
            if(!rb.empty()){
                dot_x=(rb.x+rb.width/2); dot_y=(rb.y+rb.height/2);
                for(auto& fs:faces_s){
                    int fcx=(int)(fs.x*hsx+fs.width*hsx/2);
                    int fcy=(int)(fs.y*hsy+fs.height*hsy/2);
                    if(rb.contains({fcx,fcy})&&fcy<rb.y+rb.height/2){
                        dot_x=fcx; dot_y=fcy;
                        cv::rectangle(frame,
                            cv::Rect((int)(fs.x*hsx),(int)(fs.y*hsy),
                                     (int)(fs.width*hsx),(int)(fs.height*hsy)),
                            {255,100,0},2);
                        break;
                    }
                }
                dist_m = sample_depth(dot_x, dot_y);
                cv::rectangle(frame, rb, {0,255,0}, 2);
                cv::circle(frame,{dot_x,dot_y},6,{0,0,255},-1);
                cv::line(frame,{dot_x-12,dot_y},{dot_x+12,dot_y},{255,255,255},1);
                cv::line(frame,{dot_x,dot_y-12},{dot_x,dot_y+12},{255,255,255},1);

                char buf[32];
                if(dist_m>0) snprintf(buf,sizeof(buf),"%.2fm",dist_m);
                else snprintf(buf,sizeof(buf),"?");
                cv::putText(frame,buf,{dot_x+10,dot_y-8},
                    cv::FONT_HERSHEY_SIMPLEX,0.6,{0,200,255},2);
            }
        }

        // HUD
        int temp=cpu_temp();
        cv::putText(frame,
            "Persons:"+std::to_string((int)last_bodies.size())+
            " | "+action+" | "+std::to_string(temp)+"C",
            {10,40},cv::FONT_HERSHEY_SIMPLEX,0.9,{0,255,255},2);
        cv::putText(frame,
            std::string("Follow:")+(g_following?"ON":"OFF")+
            (coral?" [CORAL]":" [CPU]"),
            {10,80},cv::FONT_HERSHEY_SIMPLEX,0.7,
            g_following?cv::Scalar{0,255,0}:cv::Scalar{100,100,255},2);

        // JPEG encode
        std::vector<uchar> buf;
        cv::imencode(".jpg",frame,buf,{cv::IMWRITE_JPEG_QUALITY,JPEG_QUALITY});
        {std::lock_guard<std::mutex> lk(g_jpeg_mtx); g_jpeg=buf;}

        // Status JSON
        {
            std::ostringstream js;
            js<<"{\"persons\":"<<last_bodies.size()
              <<",\"distance\":"<<(dist_m>0?dist_m:0)
              <<",\"action\":\""<<action<<"\""
              <<",\"following\":"<<(g_following?"true":"false")
              <<",\"temp\":"<<temp
              <<",\"coral\":"<<(coral?"true":"false")
              <<"}";
            std::lock_guard<std::mutex> lk(g_status_mtx);
            g_status_json=js.str();
        }

        // Frame pacing
        auto elapsed=std::chrono::steady_clock::now()-t0;
        float ms=std::chrono::duration<float>(elapsed).count();
        float sleep=frame_interval-ms;
        if(sleep>0)
            std::this_thread::sleep_for(
                std::chrono::duration<float>(sleep));
    }

    g_running=false;
    dt.join(); st.join();
    pipe.stop(); cap.release();
    close(g_sock_fd);
    std::cout<<"[MedPal C++] Done.\n";
    return 0;
}
