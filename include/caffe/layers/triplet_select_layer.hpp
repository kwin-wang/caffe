#ifndef CAFFE_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_LOSS_LAYER_HPP_
#include <vector>
#include <cstdlib>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class TripletSelectLayer : public Layer<Dtype> {
 public:
  explicit TripletSelectLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  void Reshape(const vector<Blob<Dtype>*>& bottom,
               const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TripletSelectLayer"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //                          const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //                           const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  inline int random_item_id(int item_id) {
    CHECK_NE(num_img_, 0);
    CHECK_NE(input_batch_size_, 0);
    while(true) {
     int num_items = input_batch_size_ / num_img_;
     int random_id = rand() % num_items;
     if (random_id != item_id) return random_id;
    }
  }

  inline int random_image_id() {
    CHECK_NE(num_img_, 0) << "The num_img_ cannot be 0, now is " << num_img_;
    return rand() % num_img_;
  }

  int num_img_; // the number of images for each item
  int input_batch_size_; // the number of input mini-batch
  int output_batch_size_;

  // 表示archor, positive, neagive数组中与bottom中位置的对应关系
  // 用于bottom每个元素的diff
  std::vector<size_t> archor_map_vec_;
  std::vector<size_t> positive_map_vec_;
  std::vector<size_t> nagetive_map_vec_;


  // TODO(yajun)
  // Dtype num_negative_; // the number of negative sample for each achor-positive pair

}; // class TripletSelectLayer

} // namespace caffe
#endif