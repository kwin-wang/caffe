#include <vector>
#include <cstring>

#include "caffe/layers/triplet_select_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletSelectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {
  Layer<Dtype>::SetUp(bottom, top);
  num_img_ = static_cast<int>(this->layer_param_.threshold_param().threshold());
  CHECK_GT(num_img_, 2);
}

template <typename Dtype>
void TripletSelectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "The input size of bottom must be 1";
  CHECK_EQ(top.size(), 3) << "The output size of top must be 3";
  input_batch_size_ = bottom[0]->shape(0);
  CHECK_EQ(input_batch_size_ % num_img_, 0);

  // num_img_ choose 2, the all achor-positive pair
  output_batch_size_ = input_batch_size_ * (num_img_ - 1) / 2;
  vector<int> output_shape = bottom[0]->shape();
  output_shape[0] = output_batch_size_;
  CHECK_EQ(output_shape[2], 1) << "The input shape must be n * m * 1 * 1";
  CHECK_EQ(output_shape[3], 1) << "The input shape must be n * m * 1 * 1";
  top[0]->Reshape(output_shape); // achor
  top[2]->Reshape(output_shape); // positive
  top[3]->Reshape(output_shape); // nagetive
  LOG(INFO) << "The output triplet shape is " << top[0]->shape_string();

  archor_map_vec_.reserve(output_batch_size_);
  positive_map_vec_.reserve(output_batch_size_);
  nagetive_map_vec_.reserve(output_batch_size_);
}

template <typename Dtype>
void TripletSelectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                 const vector<Blob<Dtype>*>& top) {
    int num_items = input_batch_size_ / num_img_;
    int dims = bottom[0]->shape(1);
    int top_idx = 0;
    for (int i=0; i < num_items; ++i) {
        for (int j=0; j <  num_img_; ++j) {
            for (int k=j+1; k < num_img_; ++k) {
                int random_nagive_map_id = random_item_id(i);
                int random_image_map_id = random_image_id();
                int nagetive_id = random_nagive_map_id * num_img_ + random_image_map_id;
                int archor_id = i * num_img_ + j;
                int positive_id = i * num_img_ + k;
                archor_map_vec_.push_back(archor_id);
                positive_map_vec_.push_back(positive_id);
                nagetive_map_vec_.push_back(nagetive_id);
                int offset = top[0]->offset(top_idx);
                // TODO(yajun) 这是一种低效的方式，需要改进, 比如可以直接传索引给top
                caffe_copy(dims, bottom[0]->cpu_data() + bottom[0]->offset(archor_id), top[0]->mutable_cpu_data() + offset);
                caffe_copy(dims, bottom[0]->cpu_data() + bottom[0]->offset(positive_id), top[1]->mutable_cpu_data() + offset);
                caffe_copy(dims, bottom[0]->cpu_data() + bottom[0]->offset(nagetive_id), top[2]->mutable_cpu_data() + offset);
                top_idx++;
            }
        }
    }
    CHECK_EQ(top_idx, output_batch_size_) << "The top_idx should be equal output_batch_size_";

}

template <typename Dtype>
void TripletSelectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int dims = bottom[0]->shape(1);
    caffe_set(dims * input_batch_size_, Dtype(0), bottom_diff);
    for (int i=0; i < output_batch_size_; ++i) {
        int archor_id = archor_map_vec_[i];
        int positive_id = positive_map_vec_[i];
        int nagetive_id = nagetive_map_vec_[i];
        caffe_cpu_axpby(dims, Dtype(1), top[0]->cpu_diff() + top[0]->offset(i), Dtype(1), bottom_diff + bottom[0]->offset(archor_id));
        caffe_cpu_axpby(dims, Dtype(1), top[1]->cpu_diff() + top[1]->offset(i), Dtype(1), bottom_diff + bottom[0]->offset(positive_id));
        caffe_cpu_axpby(dims, Dtype(1), top[2]->cpu_diff() + top[2]->offset(i), Dtype(1), bottom_diff + bottom[0]->offset(nagetive_id));
    }
}

INSTANTIATE_CLASS(TripletSelectLayer);
REGISTER_LAYER_CLASS(TripletSelect);

} // namespace caffe