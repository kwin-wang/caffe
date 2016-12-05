#ifndef CAFFE_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class DsTripletLossLayer : public LossLayer<Dtype> {
 public:
  explicit DsTripletLossLayer (const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_np_(), diff_ap_(), diff_an_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline const char* type() const { return "DsTripletLoss"; }
  /**
   * Unlike most loss layers, in the DsTripletLossLayer we can backpropagate
   * to the first three inputs.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 3;
  }

 protected:
  /// @copydoc ContrastiveLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_np_;  // cached for backward pass: f(x_i^n) - f(x_i^p)
  Blob<Dtype> diff_ap_;  // cached for backward pass: f(x_i^a) - f(x_i^p)
  Blob<Dtype> diff_an_;  // cached for backward pass: f(x_i^a) - f(x_i^n)
  Blob<Dtype> dist_ap_sq_;  // cached for backward pass :||f(x_i^a)-f(x_i^p)||^2
  Blob<Dtype> dist_an_sq_;  // cached for backward pass :||f(x_i^a)-f(x_i^n)||^2
  Blob<Dtype> diff_ap_sq_;  // tmp storage for gpu forward pass: f(x_i^a)-f(x_i^p).^2 
  Blob<Dtype> diff_an_sq_;  // tmp storage for gpu forward pass: f(x_i^a)-f(x_i^n).^2 
  Blob<Dtype> summer_vec_;  // tmp storage for gpu forward pass: [ 1, 1, 1,...,1 ]
};

}  // namespace caffe

#endif  // CAFFE_TRIPLET_LOSS_LAYER_HPP_
