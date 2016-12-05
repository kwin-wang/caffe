#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/ds_triplet_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class DsTripletLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DsTripletLossLayerTest ()
      : blob_bottom_data_i_(new Blob<Dtype>(512, 2, 1, 1)),
        blob_bottom_data_j_(new Blob<Dtype>(512, 2, 1, 1)),
        blob_bottom_data_k_(new Blob<Dtype>(512, 2, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);  // distances~=1.0 to test both sides of margin
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_i_);
    blob_bottom_vec_.push_back(blob_bottom_data_i_);
    filler.Fill(this->blob_bottom_data_j_);
    blob_bottom_vec_.push_back(blob_bottom_data_j_);
    filler.Fill(this->blob_bottom_data_k_);  
    blob_bottom_vec_.push_back(blob_bottom_data_k_);  
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~DsTripletLossLayerTest () {
    delete blob_bottom_data_i_;
    delete blob_bottom_data_j_;
    delete blob_bottom_data_k_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_data_i_; // f(x_i^a)
  Blob<Dtype>* const blob_bottom_data_j_; // f(x_i^p)
  Blob<Dtype>* const blob_bottom_data_k_; // f(x_i^n)
  Blob<Dtype>* const blob_top_loss_;      // loss
  vector<Blob<Dtype>*> blob_bottom_vec_;  //bottom
  vector<Blob<Dtype>*> blob_top_vec_;     //top
};

TYPED_TEST_CASE(DsTripletLossLayerTest , TestDtypesAndDevices);

TYPED_TEST(DsTripletLossLayerTest , TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DsTripletLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  const Dtype margin = layer_param.ds_triplet_loss_param().margin();
  const int num = this->blob_bottom_data_i_->num();
  const int channels = this->blob_bottom_data_i_->channels();
  Dtype loss(0);

  for (int i = 0; i < num; ++i) {
    Dtype dist_sq_ap(0);
    Dtype dist_sq_an(0);
    for (int j = 0; j < channels; ++j) {
      Dtype diff_ap = this->blob_bottom_data_i_->cpu_data()[i*channels+j] -
          this->blob_bottom_data_j_->cpu_data()[i*channels+j];
      dist_sq_ap += diff_ap * diff_ap;
      Dtype diff_an = this->blob_bottom_data_i_->cpu_data()[i*channels+j] -
          this->blob_bottom_data_k_->cpu_data()[i*channels+j];
      dist_sq_an += diff_an * diff_an;
    }
    loss += std::max(Dtype(0.0), margin + dist_sq_ap - dist_sq_an);
  }
  printf("============================\n");
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6);
  printf("============================\n");
}

TYPED_TEST(DsTripletLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  std::cout << "mode: " << Caffe::mode() << std::endl;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  DsTripletLossLayer<Dtype> layer(layer_param);
  //layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  // check the gradient for the three bottom layers
  printf("============================1\n");
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);  // check gradient for f(x_i^a)
  printf("============================2\n");
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);  // check gradient for f(x_i^p)
  printf("============================3\n");
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 2);  // check gradient for f(x_i^n)
  printf("============================4\n");
}
}  // namespace caffe