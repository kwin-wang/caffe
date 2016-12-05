#include <algorithm>
#include <vector>

#include "caffe/layers/ds_triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DsTripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // bottom[0] : f(x_i^a); bottom[1] : f(x_i^p); bottom[2] : f(x_i^n)
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());  
  CHECK_EQ(bottom[1]->num(), bottom[2]->num());  
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());  
  CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());  
  CHECK_EQ(bottom[0]->height(), 1);  
  CHECK_EQ(bottom[0]->width(), 1);  
  CHECK_EQ(bottom[1]->height(), 1);  
  CHECK_EQ(bottom[1]->width(), 1);  
  CHECK_EQ(bottom[2]->height(), 1);  
  CHECK_EQ(bottom[2]->width(), 1);  

  diff_np_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  
  diff_ap_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  
  diff_an_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  
  dist_ap_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  
  diff_an_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  
  dist_ap_sq_.Reshape(bottom[0]->num(), 1, 1, 1);  
  dist_an_sq_.Reshape(bottom[0]->num(), 1, 1, 1);  
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void DsTripletLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[2]->cpu_data(),  // f(x_i^n)
      bottom[1]->cpu_data(),  // f(x_i^p)
      diff_np_.mutable_cpu_data());  // f(x_i^n)-f(x_i^p)
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // f(x_i^a)
      bottom[1]->cpu_data(),  // f(x_i^p)
      diff_ap_.mutable_cpu_data());  // f(x_i^a)-f(x_i^p)
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // f(x_i^a)
      bottom[2]->cpu_data(),  // f(x_i^n)
      diff_an_.mutable_cpu_data());  // f(x_i^a)-f(x_i^n)     
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.ds_triplet_loss_param().margin();
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_ap_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_ap_.cpu_data() + (i*channels), diff_ap_.cpu_data() + (i*channels));
    dist_an_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_an_.cpu_data() + (i*channels), diff_an_.cpu_data() + (i*channels));
    Dtype trip_dist = std::max(margin + dist_ap_sq_.cpu_data()[i] - dist_an_sq_.cpu_data()[i], Dtype(0.0));  
    loss += trip_dist;  
    if(trip_dist == Dtype(0)){  
        //when ||f(x_i^a)-f(x_i^p)||^2 - ||f(x_i^a)-f(x_i^n)||^2 + margin < 0
        //this triplet has no contribution to loss,so the differential should be zero.
        caffe_set(channels, Dtype(0), diff_np_.mutable_cpu_data() + (i*channels));  
        caffe_set(channels, Dtype(0), diff_ap_.mutable_cpu_data() + (i*channels));  
        caffe_set(channels, Dtype(0), diff_an_.mutable_cpu_data() + (i*channels));     
    }    
  }
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void DsTripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 3; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 1) ? -2 : 2;
      const Dtype alpha = sign * top[0]->cpu_diff()[0];
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[i]->mutable_cpu_diff();
        if (i==0) {  // \frac{\partial(L)}{\partial f(x_i^a)}
              caffe_cpu_axpby(  
                  channels,  
                  alpha,          // 2   
                  diff_np_.cpu_data() + (j*channels),  
                  Dtype(0.0),  
                  bout + (j*channels));  
        } else if (i==1) {  // \frac{\partial(L)}{\partial f(x_i^p)}
              caffe_cpu_axpby(  
                  channels,  
                  alpha,         // -2
                  diff_ap_.cpu_data() + (j*channels),  
                  Dtype(0.0),  
                  bout + (j*channels));  
        } else if (i==2) {  // \frac{\partial(L)}{\partial f(x_i^n)}
              caffe_cpu_axpby(  
                  channels,  
                  alpha,         // 2
                  diff_an_.cpu_data() + (j*channels),  
                  Dtype(0.0),  
                  bout + (j*channels));   
        }
      } 
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DsTripletLossLayer);
#endif

INSTANTIATE_CLASS(DsTripletLossLayer);
REGISTER_LAYER_CLASS(DsTripletLoss);

}  // namespace caffe