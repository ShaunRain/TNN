# Tencent is pleased to support the open source community by making TNN available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from utils import cmd
from utils import checker
from onnx_converter import onnx2tnn
import os


def caffe2onnx(proto_path, model_path, output_path):
    work_dir = "../caffe2onnx/"
    command = "python3 caffe2onnx.py " + proto_path + " " + model_path + " -o " + output_path
    result = cmd.run(command, work_dir=work_dir)
    if result == 0:
        return True
    else:
        return False


def convert(proto_path, model_path, output_dir, version, optimize, half):
    checker.check_file_exist(proto_path)
    checker.check_file_exist(model_path)
    if output_dir is None:
        output_dir = os.path.dirname(proto_path)
    checker.check_file_exist(output_dir)
    proto_name = os.path.basename(proto_path)
    proto_name = proto_name[:-len(".prototxt")]
    onnx_path = os.path.join(output_dir, proto_name + ".onnx")
    if caffe2onnx(proto_path, model_path, onnx_path) is False:
        print("Oh No, caff2onnx failed")
    else:
        print("congratulations! caffe2onnx succeed!")
    if version is None:
        version = "v1.0"
    is_ssd = checker.is_ssd_model(proto_path)
    if is_ssd:
        onnx2tnn.convert(onnx_path, output_dir, version, False, half)
    else:
        onnx2tnn.convert(onnx_path, output_dir, version, optimize, half)