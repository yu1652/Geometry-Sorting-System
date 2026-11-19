"""Convert the pinned tray ONNX for TRT8.2, checking bitwise CPU equivalence.

Run on the model-export PC with onnx and onnxruntime installed. Does not train
or modify weights. The unmodified source is retained. Requires fixed 640 input.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort


SOURCE_SHA256 = '73008ff7385e7b31226e66b2428a550e84c657fec6fb979a9c2bc18c08e46f23'
SIZES = {'/model.10/Resize': [1, 256, 40, 40], '/model.13/Resize': [1, 128, 80, 80]}


def convert(source, destination):
    if sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('This conversion is only verified for the pinned 20261107 tray model')
    original = onnx.load(str(source))
    model = onnx.load(str(source))
    assert [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim] == [1, 3, 640, 640]
    nodes, changed = [], []
    for node in model.graph.node:
        if node.name in SIZES:
            name = node.name + '_static_sizes'
            nodes.append(helper.make_node('Constant', [], [name], name=name,
                                          value=numpy_helper.from_array(np.array(SIZES[node.name], np.int64))))
            node.input[3] = name
            changed.append(node.name)
        nodes.append(node)
    assert set(changed) == set(SIZES)
    required = {output.name for output in model.graph.output}
    retained = []
    for node in reversed(nodes):
        if any(output in required for output in node.output):
            retained.append(node)
            required.update(node.input)
    del model.graph.node[:]
    model.graph.node.extend(reversed(retained))
    assert [t.SerializeToString() for t in original.graph.initializer] == [t.SerializeToString() for t in model.graph.initializer]
    onnx.checker.check_model(model)
    onnx.save(model, str(destination))
    options = ort.SessionOptions()
    options.log_severity_level = 4
    before = ort.InferenceSession(str(source), sess_options=options, providers=['CPUExecutionProvider'])
    after = ort.InferenceSession(str(destination), sess_options=options, providers=['CPUExecutionProvider'])
    name = before.get_inputs()[0].name
    rng = np.random.default_rng(12)
    samples = [np.zeros((1, 3, 640, 640), np.float32), np.ones((1, 3, 640, 640), np.float32),
               rng.random((1, 3, 640, 640), dtype=np.float32)]
    errors = []
    for sample in samples:
        a, b = before.run(None, {name: sample})[0], after.run(None, {name: sample})[0]
        np.testing.assert_array_equal(a, b)
        errors.append(float(np.max(np.abs(a - b))))
    return {'source_sha256': SOURCE_SHA256, 'candidate_sha256': sha256(destination.read_bytes()).hexdigest(),
            'resize_sizes': SIZES, 'all_initializers_unchanged': True,
            'cpu_comparison_max_absolute_error': errors}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        raise SystemExit('Choose a new destination; existing files are not overwritten')
    print(json.dumps(convert(args.source, args.destination), indent=2))
