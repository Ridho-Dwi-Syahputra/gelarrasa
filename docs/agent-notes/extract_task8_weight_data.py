"""Read model tensors as inert pickle data; never execute serialized callables.

This is a symbolic opcode reader, not pickle.load/torch.load/load_learner. All
unknown globals/reducers remain data nodes. Only declared tensor bytes are read.
"""
from dataclasses import dataclass,field
from pathlib import Path
import json
import pickletools
import zipfile
import numpy as np
import torch
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download
from fewshot_train_only_probe import NOTES,sha,dump


@dataclass(frozen=True)
class Symbol:
    name:str


@dataclass(eq=False)
class Node:
    factory:object
    args:object=()
    state:object=None
    sequence:list=field(default_factory=list)
    mapping:dict=field(default_factory=dict)


@dataclass
class Storage:
    kind:str
    key:str
    count:int


@dataclass
class Tensor:
    storage:Storage
    offset:int
    shape:tuple
    stride:tuple


def fields(node):
    if isinstance(node,dict):return node
    if not isinstance(node,Node):return {}
    if isinstance(node.state,dict):return node.state
    if isinstance(node.state,tuple):
        return {key:value for part in node.state if isinstance(part,dict) for key,value in part.items()}
    return node.mapping


def symbolic_read(data):
    stack=[];memo={};marker=object()
    def marked():
        index=len(stack)-1
        while stack[index] is not marker:index-=1
        values=stack[index+1:];del stack[index:];return values
    def put(obj,key,value):
        (obj if isinstance(obj,dict) else obj.mapping)[key]=value
    def extend(obj,values):
        (obj if isinstance(obj,list) else obj.sequence).extend(values)
    def reduce_data(fn,args):
        name=fn.name if isinstance(fn,Symbol) else ''
        if name=='collections.OrderedDict':return dict(args[0]) if args else {}
        if name in ['torch._utils._rebuild_tensor_v2','torch._utils._rebuild_tensor']:
            return Tensor(args[0],int(args[1]),tuple(args[2]),tuple(args[3]))
        if name.startswith('torch._utils._rebuild_parameter'):return args[0]
        if name=='torch._tensor._rebuild_from_type_v2':return reduce_data(args[0],args[2])
        return Node(fn,args)
    for op,arg,pos in pickletools.genops(data):
        name=op.name
        if name in ['PROTO','FRAME']:continue
        if name=='MARK':stack.append(marker)
        elif name=='STOP':return stack[-1]
        elif name=='POP':stack.pop()
        elif name=='POP_MARK':marked()
        elif name=='DUP':stack.append(stack[-1])
        elif name=='NONE':stack.append(None)
        elif name=='NEWTRUE':stack.append(True)
        elif name=='NEWFALSE':stack.append(False)
        elif name in ['INT','BININT','BININT1','BININT2','LONG','LONG1','LONG4','FLOAT','BINFLOAT','STRING','BINSTRING','SHORT_BINSTRING','UNICODE','BINUNICODE','SHORT_BINUNICODE','BINUNICODE8','BINBYTES','SHORT_BINBYTES','BINBYTES8','BYTEARRAY8']:stack.append(arg)
        elif name=='EMPTY_TUPLE':stack.append(())
        elif name=='EMPTY_LIST':stack.append([])
        elif name=='EMPTY_DICT':stack.append({})
        elif name=='EMPTY_SET':stack.append(set())
        elif name=='TUPLE':stack.append(tuple(marked()))
        elif name in ['TUPLE1','TUPLE2','TUPLE3']:
            n=int(name[-1]);args=tuple(stack[-n:]);del stack[-n:];stack.append(args)
        elif name=='LIST':stack.append(marked())
        elif name=='DICT':
            values=marked();stack.append(dict(zip(values[::2],values[1::2])))
        elif name=='APPEND':value=stack.pop();extend(stack[-1],[value])
        elif name=='APPENDS':values=marked();extend(stack[-1],values)
        elif name=='SETITEM':value=stack.pop();key=stack.pop();put(stack[-1],key,value)
        elif name=='SETITEMS':
            values=marked()
            for key,value in zip(values[::2],values[1::2]):put(stack[-1],key,value)
        elif name=='GLOBAL':stack.append(Symbol(arg.replace(' ','.')))
        elif name=='STACK_GLOBAL':attribute=stack.pop();module=stack.pop();stack.append(Symbol(module+'.'+attribute))
        elif name=='REDUCE':args=stack.pop();fn=stack.pop();stack.append(reduce_data(fn,args))
        elif name in ['NEWOBJ','NEWOBJ_EX']:
            if name=='NEWOBJ_EX':kwargs=stack.pop()
            args=stack.pop();cls=stack.pop();stack.append(Node(cls,args))
        elif name=='BUILD':
            state=stack.pop();obj=stack[-1]
            if isinstance(obj,Node):obj.state=state
        elif name in ['BINPUT','LONG_BINPUT','PUT']:memo[int(arg)]=stack[-1]
        elif name=='MEMOIZE':memo[len(memo)]=stack[-1]
        elif name in ['BINGET','LONG_BINGET','GET']:stack.append(memo[int(arg)])
        elif name=='BINPERSID':
            ident=stack.pop();assert ident[0]=='storage' and isinstance(ident[1],Symbol)
            stack.append(Storage(ident[1].name,str(ident[2]),int(ident[4])))
        else:raise ValueError(f'Unsupported inert-data opcode {name} at {pos}')
    raise ValueError('Missing STOP opcode')


def main():
    source=hf_hub_download('C0mpl1cated3/trash_classifier_v2','waste_classifier_v2.pkl',revision='ac54e811e3a5d505b57bf6dc1495ed5cd695c036',local_files_only=True)
    with zipfile.ZipFile(source) as archive:
        assert sum(info.file_size for info in archive.infolist())<1024**3
        pickle_name=next(name for name in archive.namelist() if name.endswith('/data.pkl'));prefix=pickle_name.rsplit('/',1)[0]+'/'
        root=symbolic_read(archive.read(pickle_name));model=fields(root).get('model');assert model is not None
        tensor_specs={}
        def collect(node,path=''):
            data=fields(node)
            for section in ['_parameters','_buffers']:
                for key,value in data.get(section,{}).items():
                    if value is not None:assert isinstance(value,Tensor),(path,key,type(value));tensor_specs[path+key]=value
            for key,child in data.get('_modules',{}).items():
                if child is not None:collect(child,path+key+'.')
        collect(model);assert len(tensor_specs)>200
        cache={};state={}
        dtype={'torch.FloatStorage':'<f4','torch.DoubleStorage':'<f8','torch.HalfStorage':'<f2','torch.LongStorage':'<i8','torch.IntStorage':'<i4','torch.ByteStorage':'u1','torch.BoolStorage':'?'}
        for name,spec in tensor_specs.items():
            store=spec.storage;assert store.kind in dtype
            if store.key not in cache:
                values=np.frombuffer(archive.read(prefix+'data/'+store.key),dtype=np.dtype(dtype[store.kind]));assert len(values)==store.count;cache[store.key]=values
            values=cache[store.key];assert all(v>=0 for v in spec.shape) and all(v>=0 for v in spec.stride)
            largest=spec.offset+sum((size-1)*stride for size,stride in zip(spec.shape,spec.stride))
            assert 0<=spec.offset<=largest<len(values)
            item=np.lib.stride_tricks.as_strided(values[spec.offset:],shape=spec.shape,strides=tuple(s*values.dtype.itemsize for s in spec.stride)).copy()
            state[name]=torch.from_numpy(item)
    candidates=[];transforms=[];seen=set()
    def brief(value,depth=0):
        if value is None or isinstance(value,(str,int,float,bool)):return value
        if depth>=2:return type(value).__name__
        if isinstance(value,Tensor):return {'tensor_shape':value.shape}
        if isinstance(value,Symbol):return value.name
        if isinstance(value,Node):return {'constructor':brief(value.factory,depth+1),'args':brief(value.args,depth+1)}
        if isinstance(value,(list,tuple)):return [brief(x,depth+1) for x in value[:6]]
        return type(value).__name__
    def walk(value):
        if id(value) in seen:return
        seen.add(id(value))
        if isinstance(value,(list,tuple)):
            if len(value)==8 and all(isinstance(x,str) for x in value):candidates.append(list(value))
            for child in value:walk(child)
        elif isinstance(value,dict):
            for child in value.values():walk(child)
        elif isinstance(value,Node):
            if isinstance(value.factory,Symbol) and value.factory.name=='fastai.vision.augment.Resize':
                transforms.append({key:brief(val) for key,val in fields(value).items()})
            walk(value.args);walk(value.state);walk(value.sequence);walk(value.mapping)
    walk(root)
    valid=[v for v in candidates if set(v)=={'cardboard','e-waste','glass','metal','organic','paper','plastic','textile'}]
    assert valid and all(v==valid[0] for v in valid)
    destination=NOTES/'.runtime/task8_weights.safetensors';save_file(state,str(destination))
    report={'repository':'C0mpl1cated3/trash_classifier_v2','revision':'ac54e811e3a5d505b57bf6dc1495ed5cd695c036','original_checkpoint_sha256':sha(source),
            'source_class_order':valid[0],'tensor_count':len(state),'tensor_file_sha256':sha(destination),'serialized_callables_executed':False,'remote_application_executed':False,
            'resize_metadata':transforms,'first_keys':list(state)[:8],'last_keys':list(state)[-18:]}
    dump(NOTES/'json/task8_tensor_extraction.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
