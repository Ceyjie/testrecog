#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <vector>
#include <string>
#include <memory>

namespace medpal {

class FaceRecognizer {
public:
    FaceRecognizer(const std::string& embeddings_dir);
    ~FaceRecognizer();
    
    std::vector<float> extract_embedding(const cv::Mat& face_roi);
    std::string recognize(const std::vector<float>& embedding, float threshold = 0.5f);
    bool save_embedding(const std::string& name, const std::vector<float>& embedding);
    void load_all_embeddings();
    
private:
    struct PersonEmbedding {
        std::string name;
        std::vector<float> embedding;
    };
    
    std::string embeddings_dir_;
    std::unique_ptr<cv::dnn::Net> embed_net_;
    std::vector<PersonEmbedding> known_embeddings_;
    
    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);
    std::vector<float> preprocess_face(const cv::Mat& face);
};

}
