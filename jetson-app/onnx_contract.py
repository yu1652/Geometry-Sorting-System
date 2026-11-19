"""Bounded, standard-library inspection of the app's raw YOLO ONNX contract.

This is not a general ONNX checker. It checks static interface/label metadata
and rejects external tensor data in known ONNX graph, attribute, sparse,
training and function containers. TensorRT must still parse and validate the
complete graph before any engine can be accepted. Unknown protobuf fields are
length-checked and skipped; tensor payload bytes are deliberately opaque.

Field numbers follow https://github.com/onnx/onnx/blob/main/onnx/onnx.proto.
No ONNX/protobuf package, model execution, or external-weight file access.
"""
import ast
import hashlib
import os


MAX_MODEL_BYTES = 512 * 1024 * 1024
MAX_FIELDS = 1000000
MAX_DEPTH = 64
MAX_TEXT_BYTES = 65536
MAX_CLASSES = 1000


class _Reader:
    def __init__(self, data):
        self.data = data
        self.remaining_fields = MAX_FIELDS

    def _varint(self, position, end):
        value = 0
        for index in range(10):
            if position >= end:
                raise ValueError('Truncated protobuf varint')
            byte = self.data[position]
            position += 1
            if index == 9 and byte > 1:
                raise ValueError('Protobuf varint exceeds 64 bits')
            value |= (byte & 127) << (index * 7)
            if byte < 128:
                if index and byte == 0:
                    raise ValueError('Noncanonical protobuf varint')
                return value, position
        raise ValueError('Protobuf varint exceeds 10 bytes')

    def fields(self, bounds, depth):
        if depth > MAX_DEPTH:
            raise ValueError('ONNX protobuf depth limit exceeded')
        position, end = bounds
        while position < end:
            self.remaining_fields -= 1
            if self.remaining_fields < 0:
                raise ValueError('ONNX protobuf field limit exceeded')
            tag, position = self._varint(position, end)
            field, wire = tag >> 3, tag & 7
            if not 1 <= field < (1 << 29):
                raise ValueError('Invalid protobuf field number')
            if wire == 0:
                value, position = self._varint(position, end)
            elif wire in (1, 5):
                stop = position + (8 if wire == 1 else 4)
                if stop > end:
                    raise ValueError('Truncated fixed-width protobuf field')
                value, position = (position, stop), stop
            elif wire == 2:
                length, position = self._varint(position, end)
                stop = position + length
                if stop > end:
                    raise ValueError('Truncated length-delimited protobuf field')
                value, position = (position, stop), stop
            else:
                raise ValueError('Unsupported protobuf wire type')
            yield field, wire, value

    @staticmethod
    def message(wire, value):
        if wire != 2:
            raise ValueError('Expected length-delimited ONNX message')
        return value

    @staticmethod
    def number(wire, value):
        if wire != 0:
            raise ValueError('Expected ONNX integer field')
        return value

    def text(self, wire, value):
        start, end = self.message(wire, value)
        if end - start > MAX_TEXT_BYTES:
            raise ValueError('ONNX metadata text limit exceeded')
        try:
            return self.data[start:end].decode('utf8')
        except UnicodeDecodeError as error:
            raise ValueError('Invalid UTF-8 ONNX metadata') from error

    def strings_entry(self, bounds, depth):
        result = {}
        for field, wire, value in self.fields(bounds, depth):
            if field in (1, 2):
                if field in result:
                    raise ValueError('Duplicate ONNX metadata entry field')
                result[field] = self.text(wire, value)
        if set(result) != {1, 2}:
            raise ValueError('Incomplete ONNX metadata entry')
        return result[1], result[2]

    def value_info(self, bounds, depth):
        result = {}
        for field, wire, value in self.fields(bounds, depth):
            if field in (1, 2):
                if field in result:
                    raise ValueError('Duplicate ONNX value interface field')
                result[field] = (self.text(wire, value) if field == 1 else
                                 self.tensor_type(self.message(wire, value), depth + 1))
        if set(result) != {1, 2} or not result[1]:
            raise ValueError('Missing ONNX value name or tensor type')
        dtype, shape = result[2]
        return result[1], dtype, shape

    def tensor_type(self, bounds, depth):
        tensor = None
        for field, wire, value in self.fields(bounds, depth):
            if field in (1, 4, 5, 7, 8, 9):
                if tensor is not None or field != 1:
                    raise ValueError('Only ordinary ONNX tensor interfaces are supported')
                tensor = self.message(wire, value)
        if tensor is None:
            raise ValueError('Missing ONNX tensor type')
        result = {}
        for field, wire, value in self.fields(tensor, depth + 1):
            if field in (1, 2):
                if field in result:
                    raise ValueError('Duplicate ONNX tensor type field')
                result[field] = (self.number(wire, value) if field == 1 else
                                 self.shape(self.message(wire, value), depth + 2))
        if set(result) != {1, 2} or result[1] != 1:
            raise ValueError('ONNX input and output must be float32 tensors')
        return result[1], result[2]

    def shape(self, bounds, depth):
        dimensions = []
        for field, wire, value in self.fields(bounds, depth):
            if field != 1:
                continue
            dimension = None
            for number, kind, item in self.fields(self.message(wire, value), depth + 1):
                if number == 2:
                    raise ValueError('Dynamic ONNX dimensions are not supported')
                if number == 1:
                    if dimension is not None:
                        raise ValueError('Duplicate ONNX dimension')
                    dimension = self.number(kind, item)
            if dimension is None or not 0 < dimension < (1 << 63):
                raise ValueError('ONNX dimensions must be positive static integers')
            dimensions.append(dimension)
            if len(dimensions) > 8:
                raise ValueError('ONNX tensor rank exceeds interface limit')
        return dimensions

    def scan(self, kind, bounds, depth):
        """Traverse only schema-defined containers; never parse raw weight bytes."""
        children = {
            'graph': {1: 'node', 5: 'tensor', 15: 'sparse'},
            'node': {5: 'attribute'},
            'attribute': {5: 'tensor', 10: 'tensor', 6: 'graph', 11: 'graph',
                          22: 'sparse', 23: 'sparse'},
            'sparse': {1: 'tensor', 2: 'tensor'},
            'training': {1: 'graph', 2: 'graph'},
            'function': {7: 'node', 11: 'attribute'},
            'tensor': {},
        }
        for field, wire, value in self.fields(bounds, depth):
            if kind == 'tensor':
                if field == 13:
                    raise ValueError('External ONNX tensor data is not supported')
                if field == 14 and self.number(wire, value) != 0:
                    raise ValueError('External ONNX tensor data location is not supported')
            child = children[kind].get(field)
            if child:
                self.scan(child, self.message(wire, value), depth + 1)

    def graph(self, bounds, depth):
        inputs, outputs = [], []
        for field, wire, value in self.fields(bounds, depth):
            if field in (11, 12):
                target = inputs if field == 11 else outputs
                target.append(self.value_info(self.message(wire, value), depth + 1))
                if len(target) > 1:
                    raise ValueError('YOLO ONNX requires exactly one input and output')
            elif field in (1, 5, 15):
                self.scan({1: 'node', 5: 'tensor', 15: 'sparse'}[field],
                          self.message(wire, value), depth + 1)
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError('YOLO ONNX requires exactly one input and output')
        return inputs[0], outputs[0]

    def opset(self, bounds, depth):
        result = {}
        for field, wire, value in self.fields(bounds, depth):
            if field in (1, 2):
                if field in result:
                    raise ValueError('Duplicate ONNX opset field')
                result[field] = self.text(wire, value) if field == 1 else self.number(wire, value)
        if not 0 < result.get(2, 0) < (1 << 63):
            raise ValueError('Invalid or missing ONNX opset version')
        return {'domain': result.get(1, ''), 'version': result[2]}


def _class_names(value):
    try:
        expression = ast.parse(value, mode='eval')
        # ast.literal_eval does not execute expressions, but also bound AST
        # complexity and preserve duplicate-key detection before dict creation.
        pending = [(expression, 0)]
        count = 0
        while pending:
            node, depth = pending.pop()
            count += 1
            if count > 5000 or depth > 16:
                raise ValueError('ONNX names literal complexity limit exceeded')
            pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
        if not isinstance(expression.body, ast.Dict):
            raise ValueError('ONNX names must map consecutive class indices')
        mapping = ast.literal_eval(expression)
        if not 1 <= len(mapping) <= MAX_CLASSES:
            raise ValueError('Invalid ONNX class count')
        names = {}
        for key_node, value_node in zip(expression.body.keys, expression.body.values):
            key, name = ast.literal_eval(key_node), ast.literal_eval(value_node)
            if type(key) is int:
                index = key
            elif type(key) is str and key.isdigit() and key == str(int(key)):
                index = int(key)
            else:
                raise ValueError('ONNX class keys must be integer indices')
            if index in names or not isinstance(name, str) or not name.strip():
                raise ValueError('Duplicate class index or invalid ONNX class name')
            names[index] = name
        if set(names) != set(range(len(names))):
            raise ValueError('ONNX class indices must be consecutive from zero')
        ordered = [names[index] for index in range(len(names))]
        if len(set(ordered)) != len(ordered):
            raise ValueError('Duplicate ONNX class name')
        return ordered
    except (SyntaxError, TypeError, OverflowError, RecursionError) as error:
        raise ValueError('Invalid ONNX names literal') from error


def inspect_yolo_onnx(path):
    """Return JSON-safe contract metadata; raise ValueError on unsupported data.

    File access errors remain OSError. The bounded file is inspected in memory;
    external weight paths are never opened and no graph is executed.
    """
    with open(path, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not 0 < before.st_size <= MAX_MODEL_BYTES:
            raise ValueError('ONNX model size limit exceeded or file is empty')
        data = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
    if (len(data) != before.st_size or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns):
        raise ValueError('ONNX file changed during inspection')
    reader = _Reader(data)
    ir_version, graph_info = None, None
    metadata, opsets = {}, []
    for field, wire, value in reader.fields((0, len(data)), 0):
        if field == 1:
            if ir_version is not None:
                raise ValueError('Duplicate ONNX IR version')
            ir_version = reader.number(wire, value)
        elif field == 7:
            if graph_info is not None:
                raise ValueError('Duplicate ONNX graph')
            graph_info = reader.graph(reader.message(wire, value), 1)
        elif field == 8:
            opsets.append(reader.opset(reader.message(wire, value), 1))
        elif field == 14:
            key, item = reader.strings_entry(reader.message(wire, value), 1)
            if key in metadata:
                raise ValueError('Duplicate ONNX metadata key')
            metadata[key] = item
        elif field in (20, 25):
            reader.scan('training' if field == 20 else 'function', reader.message(wire, value), 1)
    if ir_version is None or not 0 < ir_version < (1 << 63) or graph_info is None:
        raise ValueError('Missing or invalid ONNX IR version or graph')
    if not opsets or len({item['domain'] for item in opsets}) != len(opsets):
        raise ValueError('Missing or duplicate ONNX opset domain')
    if metadata.get('task') != 'detect' or 'names' not in metadata:
        raise ValueError('ONNX metadata must declare task detect and class names')
    names = _class_names(metadata['names'])
    (input_name, _, input_shape), (output_name, _, output_shape) = graph_info
    if input_name != 'images' or input_shape != [1, 3, 320, 320]:
        raise ValueError('YOLO ONNX input must be images float32 [1,3,320,320]')
    if output_shape != [1, 4 + len(names), 2100]:
        raise ValueError('YOLO ONNX output must be float32 [1,4+classes,2100]')
    return {'names': names, 'input_name': input_name, 'output_name': output_name,
            'input_shape': input_shape, 'output_shape': output_shape, 'dtype': 'float32',
            'sha256': hashlib.sha256(data).hexdigest(), 'ir_version': ir_version,
            'opsets': opsets, 'size_bytes': len(data)}
