#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from audioop import bias
from platform import node
import sys
import os.path as osp
import onnx_graphsurgeon as gs
import numpy as np
import onnx

op = 'linearCuda'
ROOT = osp.abspath(osp.dirname(__file__))
sys.path.append(ROOT)

# 1. 导入计算图
graph = gs.import_onnx(onnx.load(osp.join(ROOT, "./linear_cuda.onnx")))

nodes = graph.nodes

# 2. 获取所有linearCuda算子
linear_cuda_node_list = [node for node in nodes if node.op == op]

# 3. 修改算子的输入和属性信息
for linear_cuda in linear_cuda_node_list:
    attrs = linear_cuda.attrs
    inputs = linear_cuda.inputs
    weight = inputs.pop(1)
    bias = inputs.pop(1)
    attrs["weight"] = weight
    attrs["bias"] = bias
    np.save("./weight.npy", weight.values)
    np.save("./bias.npy", bias.values)

graph.__module__ = "custom"

# 5. 清理算子
# graph.cleanup().toposort()

# 4. 保存onnx文件
onnx.save(gs.export_onnx(graph), "./linear_cuda_modified.onnx")

# print(nodes)

# # 1. Remove the `b` input of the add node
# first_add = [node for node in graph.nodes if node.op == "Add"][0]
# first_add.inputs = [inp for inp in first_add.inputs if inp.name != "b"]

# # 2. Change the Add to a LeakyRelu
# first_add.op = "LeakyRelu"
# first_add.attrs["alpha"] = 0.02

# # 3. Add an identity after the add node
# identity_out = gs.Variable("identity_out", dtype=np.float32)
# identity = gs.Node(op="Identity", inputs=first_add.outputs, outputs=[identity_out])
# graph.nodes.append(identity)

# # 4. Modify the graph output to be the identity output
# graph.outputs = [identity_out]

# # 5. Remove unused nodes/tensors, and topologically sort the graph
# # ONNX requires nodes to be topologically sorted to be considered valid.
# # Therefore, you should only need to sort the graph when you have added new nodes out-of-order.
# # In this case, the identity node is already in the correct spot (it is the last node,
# # and was appended to the end of the list), but to be on the safer side, we can sort anyway.
# graph.cleanup().toposort()

# onnx.save(gs.export_onnx(graph), "modified.onnx")