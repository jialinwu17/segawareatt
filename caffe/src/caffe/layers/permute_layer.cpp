#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/permute_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PermuteLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      const PermuteParameter& param = this->layer_param_.permute_param();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    channels_ = bottom[0]->channels();
    type_ = param.type();
}

template <typename Dtype>
void PermuteLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    channels_ = bottom[0]->channels();
    if(type_ == 0) top[0]->Reshape(bottom[0]->num(), height_,width_,channels_);
    else top[0]->Reshape(bottom[0]->num(), width_,channels_,height_);
  map_idx_.Reshape(bottom[0]->num(), channels_,height_,width_);
}

template <typename Dtype>
void PermuteLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void PermuteLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(PermuteLayer);
#endif

INSTANTIATE_CLASS(PermuteLayer);
REGISTER_LAYER_CLASS(Permute);

}  // namespace caffe
