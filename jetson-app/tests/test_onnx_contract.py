"""Bounded import-contract inspection without ONNX/protobuf dependencies."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import onnx_contract as contract
except ImportError:
    contract=None


def varint(value):
    if value<0:value+=(1<<64)
    result=bytearray()
    while value>127:
        result.append((value&127)|128);value>>=7
    result.append(value)
    return bytes(result)


def integer(field,value):return varint(field<<3)+varint(value)
def blob(field,value):return varint((field<<3)|2)+varint(len(value))+value
def text(field,value):return blob(field,value.encode('utf8'))
def entry(key,value):return text(1,key)+text(2,value)


def value_info(name,shape,dtype=1):
    dimensions=b''.join(blob(1,text(2,d) if isinstance(d,str) else integer(1,d)) for d in shape)
    return text(1,name)+blob(2,blob(1,integer(1,dtype)+blob(2,dimensions)))


def graph(inputs=None,outputs=None,extra=b''):
    if inputs is None:inputs=[value_info('images',(1,3,320,320))]
    if outputs is None:outputs=[value_info('output0',(1,6,2100))]
    return extra+b''.join(blob(11,v) for v in inputs)+b''.join(blob(12,v) for v in outputs)


def model(graph_bytes=None,names="{0:'cube',1:'cone'}",task='detect',extra=b'',opsets=None):
    if graph_bytes is None:graph_bytes=graph()
    if opsets is None:opsets=[text(1,'')+integer(2,12)]
    metadata=blob(14,entry('names',names))+blob(14,entry('task',task))
    return integer(1,7)+blob(7,graph_bytes)+b''.join(blob(8,o) for o in opsets)+metadata+extra


def tensor(external=False):
    result=integer(2,1)+text(8,'weight')+blob(9,b'\x00\x00\x00\x00')
    return result+(blob(13,entry('location','weights.bin')) if external else b'')


def node_attribute(attribute):return blob(1,blob(5,attribute))


class OnnxContractTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(contract,'The standard-library ONNX contract inspector is not implemented')
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'model.onnx'

    def inspect(self,data):
        self.path.write_bytes(data)
        return contract.inspect_yolo_onnx(self.path)

    def test_static_fp32_detector_returns_names_shapes_and_exact_file_provenance(self):
        data=model()
        actual=self.inspect(data)
        self.assertEqual(actual,{'names':['cube','cone'],'input_name':'images','output_name':'output0',
            'input_shape':[1,3,320,320],'output_shape':[1,6,2100],'dtype':'float32',
            'sha256':hashlib.sha256(data).hexdigest(),'ir_version':7,
            'opsets':[{'domain':'','version':12}],'size_bytes':len(data)})

    def test_string_and_integer_class_keys_preserve_index_order(self):
        self.assertEqual(self.inspect(model(names="{'1':'cone',0:'cube'}"))['names'],['cube','cone'])

    def test_invalid_names_are_rejected_without_evaluating_code(self):
        cases=("{0:'cube',2:'cone'}","{0:'cube',1:'cube'}","{0:'cube','0':'cone'}",
               "{0:'cube',0:'cone',1:'ball'}","{True:'cube',1:'cone'}","['cube','cone']",
               "{0:'',1:'cone'}","{0:123,1:'cone'}","__import__('os').getcwd()")
        for names in cases:
            with self.subTest(names=names),self.assertRaises(ValueError):self.inspect(model(names=names))

    def test_names_expression_cannot_create_a_file_before_it_is_rejected(self):
        sentinel=Path(self.directory.name)/'evaluated.txt'
        payload="__import__('pathlib').Path(%r).write_text('executed')"%str(sentinel)
        with self.assertRaises(ValueError):self.inspect(model(names=payload))
        self.assertFalse(sentinel.exists(),'Metadata must never execute before validation fails')

    def test_only_detect_task_and_matching_class_count_are_accepted(self):
        for data in (model(task='segment'),model(graph(outputs=[value_info('out',(1,7,2100))])),
                     model(extra=blob(14,entry('names',"{0:'other',1:'name'}")))):
            with self.subTest(data=data),self.assertRaises(ValueError):self.inspect(data)

    def test_single_images_input_and_single_fp32_raw_output_are_required(self):
        values=(graph(inputs=[]),graph(inputs=[value_info('other',(1,3,320,320))]),
                graph(inputs=[value_info('images',(1,3,320,320))]*2),graph(outputs=[]),
                graph(outputs=[value_info('out',(1,6,2100))]*2),
                graph(inputs=[value_info('images',(1,3,320,320),10)]),
                graph(outputs=[value_info('out',(1,6,2100),10)]))
        for value in values:
            with self.subTest(graph=value),self.assertRaises(ValueError):self.inspect(model(value))

    def test_dynamic_wrong_size_rank_or_nonpositive_dimensions_are_rejected(self):
        for shape in ((1,3,640,640),('batch',3,320,320),(0,3,320,320),(-1,3,320,320),(3,320,320)):
            with self.subTest(shape=shape),self.assertRaises(ValueError):
                self.inspect(model(graph(inputs=[value_info('images',shape)])))

    def test_external_initializer_and_external_location_without_map_are_rejected(self):
        for weight in (tensor(True),tensor()+integer(14,1),tensor()+blob(13,b'')):
            with self.subTest(weight=weight),self.assertRaises(ValueError):
                self.inspect(model(graph(extra=blob(5,weight))))

    def test_external_attribute_tensors_and_nested_graphs_are_rejected(self):
        attributes=(blob(5,tensor(True)),blob(10,tensor(True)),
                    blob(6,graph(extra=blob(5,tensor(True)))),blob(11,graph(extra=blob(5,tensor(True)))),
                    blob(22,blob(1,tensor(True))),blob(23,blob(2,tensor(True))))
        for attribute in attributes:
            with self.subTest(attribute=attribute),self.assertRaises(ValueError):
                self.inspect(model(graph(extra=node_attribute(attribute))))

    def test_external_sparse_training_and_function_tensors_are_rejected(self):
        cases=(model(graph(extra=blob(15,blob(1,tensor(True))))),
               model(extra=blob(20,blob(1,graph(extra=blob(5,tensor(True)))))),
               model(extra=blob(25,blob(11,blob(5,tensor(True))))),
               model(extra=blob(25,blob(7,blob(5,blob(5,tensor(True)))))))
        for data in cases:
            with self.subTest(data=data),self.assertRaises(ValueError):self.inspect(data)

    def test_inline_tensor_bytes_are_opaque_and_do_not_resemble_external_metadata(self):
        suspicious=blob(13,entry('location','not-a-real-external-file'))+b'\x80'*20
        result=self.inspect(model(graph(extra=blob(5,integer(2,1)+blob(9,suspicious)))))
        self.assertEqual(result['names'],['cube','cone'])

    def test_malformed_varints_tags_wire_types_and_truncated_lengths_are_rejected(self):
        for invalid in (b'\x80',b'\x80'*11,b'\xff'*9+b'\x02',b'\x00',b'\x0b',b'\x0f',
                        b'\x3a\x85\x01abc',b'\x09\x00',b'\x0d\x00',model()+b'\x12\x05ab'):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):self.inspect(invalid)

    def test_wrong_known_wire_type_and_duplicate_scalar_headers_are_rejected(self):
        for invalid in (model()+integer(7,1),model()+integer(1,8),model(extra=blob(8,integer(2,0))),
                        model(opsets=[]),model(extra=blob(8,integer(2,13)))):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):self.inspect(invalid)

    def test_protobuf_depth_and_field_budgets_are_bounded(self):
        nested=graph()
        for _ in range(30):nested=graph(extra=node_attribute(blob(6,nested)))
        with self.assertRaisesRegex(ValueError,'depth'):self.inspect(model(nested))
        with mock.patch.object(contract,'MAX_FIELDS',4):
            with self.assertRaisesRegex(ValueError,'field'):self.inspect(model())

    def test_file_size_and_names_text_limits_are_checked_before_unbounded_parsing(self):
        self.assertEqual(contract.MAX_MODEL_BYTES,512*1024*1024)
        with mock.patch.object(contract,'MAX_MODEL_BYTES',8):
            with self.assertRaisesRegex(ValueError,'size|large|limit'):self.inspect(model())
        with self.assertRaises(ValueError):self.inspect(model(names=' '*70000))

    def test_missing_ir_graph_or_metadata_and_invalid_utf8_are_rejected(self):
        for data in (b'',integer(1,7),blob(7,graph()),
                     integer(1,7)+blob(7,graph())+blob(8,integer(2,12)),
                     model(extra=blob(14,blob(1,b'\xff')+text(2,'bad')))):
            with self.subTest(data=data),self.assertRaises(ValueError):self.inspect(data)


if __name__=='__main__':unittest.main()
