#include <algorithm>
#include <vector>

#include "caffe/layers/ds_triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp" 

namespace caffe {

template <typename Dtype>
void DsTripletLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub(  
      count,  
      bottom[2]->gpu_data(),  // f(x_i^n)
      bottom[1]->gpu_data(),  // f(x_i^p)
      diff_np_.mutable_gpu_data());  // f(x_i^n)-f(x_i^p)
  caffe_gpu_sub(  
      count,  
      bottom[0]->gpu_data(),  // f(x_i^a)
      bottom[1]->gpu_data(),  // f(x_i^p)
      diff_ap_.mutable_gpu_data());  // f(x_i^a)-f(x_i^p)
  caffe_gpu_sub(  
      count,  
      bottom[0]->gpu_data(),  // f(x_i^a)
      bottom[2]->gpu_data(),  // f(x_i^n)
      diff_an_.mutable_gpu_data());  // f(x_i^a)-f(x_i^n)
  caffe_gpu_powx(  
      count,  
      diff_ap_.mutable_gpu_data(),  // f(x_i^a)-f(x_i^p)
      Dtype(2),  
      diff_ap_sq_.mutable_gpu_data());  // (f(x_i^a)-f(x_i^p)).^2 
  caffe_gpu_powx(  
      count,  
      diff_an_.mutable_gpu_data(),  // f(x_i^a)-f(x_i^n)
      Dtype(2),  
      diff_an_sq_.mutable_gpu_data());  // (f(x_i^a)-f(x_i^n)).^2 
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),               //alpha
      diff_ap_sq_.gpu_data(),   // A :  (f(x_i^a)-f(x_i^p)).^2 
      summer_vec_.gpu_data(),   // x :  [ 1, 1, 1,...,1 ]
      Dtype(0.0),               // beta
      dist_ap_sq_.mutable_gpu_data());  // y :||f(x_i^a)-f(x_i^p)||^2 
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),               //alpha 
      diff_an_sq_.gpu_data(),   // A :  (f(x_i^a)-f(x_i^n)).^2 
      summer_vec_.gpu_data(),   // x :  [ 1, 1, 1,...,1 ]
      Dtype(0.0),               // beta
      dist_an_sq_.mutable_gpu_data());  // y :||f(x_i^a)-f(x_i^n)||^2 
  Dtype margin = this->layer_param_.ds_triplet_loss_param().margin();
  Dtype loss(0.0);
for (int i = 0; i < bottom[0]->num(); ++i) {  
     loss += std::max(margin +dist_ap_sq_.cpu_data()[i]- dist_an_sq_.cpu_data()[i], Dtype(0.0));  
  }  
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>  
__global__ void CLLBackward(const int count, const int channels,  
    const Dtype margin, const Dtype alpha, const Dtype* diff,
    const Dtype* dist_ap_sq, const Dtype* dist_an_sq,  
    Dtype *bottom_diff) {  
  CUDA_KERNEL_LOOP(i, count) {  
    int n = i / channels;  // the num index, to access dist_ap_sq_ and dist_an_sq_  
    Dtype trip_dist(0.0);  
    trip_dist= margin + dist_ap_sq[n] - dist_an_sq[n];  
    if (trip_dist> 0.0) {  
        bottom_diff[i] = alpha * diff[i];  
    } else {  
        bottom_diff[i] = 0;  
    }  
  }  
}  

template <typename Dtype>
void DsTripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = this->layer_param_.ds_triplet_loss_param().margin(); 
  const int count = bottom[0]->count();  
  const int channels = bottom[0]->channels(); 

  for (int i = 0; i < 3; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 1) ? -2 : 2;
      const Dtype alpha = sign * top[0]->cpu_diff()[0];
      // NOLINT_NEXT_LINE(whitespace/operators)
      if(i==0){     // \frac{\partial(L)}{\partial f(x_i^a)}
          // NOLINT_NEXT_LINE(whitespace/operators)  
          CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(  
              count, channels, margin, alpha,  
              diff_np_.gpu_data(),  //  f(x_i^n)-f(x_i^p)
              dist_ap_sq_.gpu_data(),  // ||f(x_i^a)-f(x_i^p)||^2   
              dist_an_sq_.gpu_data(),  // ||f(x_i^a)-f(x_i^n)||^2 
              bottom[i]->mutable_gpu_diff());  
          CUDA_POST_KERNEL_CHECK;  
      }else if(i==1){    // \frac{\partial(L)}{\partial f(x_i^p)}
          // NOLINT_NEXT_LINE(whitespace/operators)  
          CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(  
              count, channels, margin, alpha,   
              diff_ap_.gpu_data(),  //  f(x_i^a)-f(x_i^p)
              dist_ap_sq_.gpu_data(),  // ||f(x_i^a)-f(x_i^p)||^2   
              dist_an_sq_.gpu_data(),  // ||f(x_i^a)-f(x_i^n)||^2   
              bottom[i]->mutable_gpu_diff());  
          CUDA_POST_KERNEL_CHECK;  
      }else if(i==2){   // \frac{\partial(L)}{\partial f(x_i^n)}
          // NOLINT_NEXT_LINE(whitespace/operators)  
          CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(  
              count, channels, margin, alpha,  
              diff_an_.gpu_data(),  //  f(x_i^a)-f(x_i^n)
              dist_ap_sq_.gpu_data(),  // ||f(x_i^a)-f(x_i^p)||^2   
              dist_an_sq_.gpu_data(),  // ||f(x_i^a)-f(x_i^n)||^2 
              bottom[i]->mutable_gpu_diff());  
          CUDA_POST_KERNEL_CHECK;  
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DsTripletLossLayer);

}  // namespace caffe