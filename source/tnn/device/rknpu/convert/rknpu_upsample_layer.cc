// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <algorithm>

#include "rknpu_base_layer.h"
#include "rknpu_utils.h"

namespace TNN_NS {

DECLARE_RKNPU_LAYER(Upsample, LAYER_UPSAMPLE);

Status RknpuUpsampleLayer::Convert() {
    auto param = dynamic_cast<UpsampleLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    if (param->mode != 2 || param->align_corners != 0) {
        return Status(TNNERR_PARAM_ERR, "upsample only support bilinear mode");
    }

    Status ret = TNN_OK;
    std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;

    // input
    inputs.push_back(input_ops_[0]);

    // output
    ADD_OUTPUT_OP();

    rk::nn::ResizeAttr attr;

    auto output_dims = output_ops_[0]->GetDims();
    attr.sizes.assign(output_dims.end() - 2, output_dims.end());
    attr.mode           = rk::nn::InterpolationMode::BILINEAR;
    attr.transfrom_mode = rk::nn::CoordTransformMode::ALIGN_CORNERS;
    attr.scale = 0.f;

    graph_->AddOperator(rk::nn::OperatorType::RESIZE, inputs, output_ops_, (void *)&attr);

    return ret;
}

REGISTER_RKNPU_LAYER(Upsample, LAYER_UPSAMPLE);

}  // namespace TNN_NS
