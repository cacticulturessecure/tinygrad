import sys, atexit, functools, pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from tinygrad.ops import GroupOp, UOp, Ops, PatternMatcher, UPat, Variable, can_pad, graph_rewrite, resolve, track_rewrites, view_left, merge_views, graph_rewrite_map
from tinygrad.ops import identity_element, buffers, symbolic_simple, type_verify
from tinygrad.helpers import Context, Metadata, all_int, all_same, colored, diskcache_put, merge_dicts, prod, dedup, getenv, unwrap
from tinygrad.helpers import FUSE_CONV_BW, FUSE_ARANGE, DEBUG, CAPTURE_PROCESS_REPLAY, ContextVar
from tinygrad.dtype import DType, ImageDType, dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.device import Buffer

# creation can recurse a lot
sys.setrecursionlimit(10000)

# **** Tensor UOp spec

tensor_uop_spec = PatternMatcher([
  (UPat(Ops.DEVICE, dtypes.void, (), name="device"), lambda device: isinstance(device.arg, str)),
  (UPat(Ops.BUFFER, src=(UPat(Ops.DEVICE),), name="buf"),
   lambda buf: isinstance(buf.arg, tuple) and len(buf.arg) == 2 and all_int(buf.arg) and isinstance(buf.dtype, (DType, ImageDType))),

  (UPat(GroupOp.Movement, name="mv", src=(UPat.var("x"),)),
   # naturally correct
   lambda mv,x: (isinstance(mv.arg, tuple) and mv.dtype == x.dtype) or
   # "make things that can't be images not images" can change the buffer dtype
   # this is fine as long as it's a realized buffer and base dtypes match.
   ((isinstance(mv.dtype, ImageDType) or isinstance(x.dtype, ImageDType)) and x.dtype.base == mv.dtype.base and x.is_realized)),

  # Tensor variable bindings
  (UPat(Ops.BIND, dtypes.int, (UPat(Ops.DEFINE_VAR), UPat.cvar(dtype=dtypes.int)), arg=None), lambda: True),
  (UPat(Ops.DEFINE_VAR, src=(UPat(Ops.VIEW, arg=ShapeTracker.from_shape(()))), arg=None), lambda: True),

  # Tensor const has an unmasked ShapeTracker of stride 0 and a device
  (UPat(Ops.CONST, src=(UPat(Ops.VIEW, name="st", src=(UPat(Ops.DEVICE),)),)),
   lambda st: len(st.st.views) == 1 and all(s == 0 for s in st.st.views[0].strides) and st.st.views[0].mask is None),

  # DETACH and CONTIGUOUS change how we interpret the source UOp
  # CONTIGUOUS ensures the source UOp realizes
  (UPat((Ops.DETACH, Ops.CONTIGUOUS), name="root", src=(UPat.var("x"),), arg=None), lambda root,x: root.dtype == x.dtype),

  # COPY
  # NOTE: the arg here specifies clone=True, which prevents folding same device copy
  (UPat(Ops.COPY, name="copy", src=(UPat(Ops.DEVICE), UPat.var("x"))), lambda copy,x: isinstance(copy.arg, bool) and copy.dtype == x.dtype),

  # VIEW(BUFFER) applies a ShapeTracker on top of the underlying device buffer
  # NOTE: VIEW size exactly matches the underlying BUFFER, tensor doesn't apply movement ops to the VIEW
  (UPat(Ops.VIEW, name="view", src=(UPat(Ops.BUFFER, name="buf"),)),
   lambda view,buf: view.dtype == buf.dtype and view.size == buf.size and view.st.contiguous),

  # ASSIGN changes the value of a realized buffer
  (UPat(Ops.ASSIGN, name="assign", src=(UPat.var("target"), UPat.var("new_val"))),
   lambda assign,target,new_val: (target.op is Ops.BUFFER or target.is_realized) and (assign.dtype == target.dtype == new_val.dtype)),
])

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...]

def view_const(mv:UOp, x:UOp):
  new_st = unwrap(x.st)+unwrap(mv.st)
  if new_st.views[0].mask is None: return x.replace(src=(x.src[0].replace(arg=new_st),))

remove_movement_ops = PatternMatcher([
  (UPat(GroupOp.Movement, name="mov", src=(UPat.var("x"),)), lambda x,mov: x.view(mov.st)),
  (UPat(Ops.VIEW, name="mv", src=(UPat.cvar("x"),)), view_const),
])

def collapse_const_reduce(root:UOp, x:UOp):
  if not all_int(x.shape): return None
  # remove root on unmasked const
  prshape = prod(unwrap(x.st).shape[i] for i in root.arg[1])
  ret = x.const_arg
  match root.arg[0]:
    case Ops.ADD: ret *= prshape
    case Ops.MUL: ret **= prshape
    case Ops.MAX: pass # NOTE: Ops.MAX is passthrough
    case _: return None
  return root.const_like(ret)

sym = symbolic_simple+PatternMatcher([
  (UPat(set(Ops)-{Ops.SINK}, name="root"), lambda root: root.const_like(0) if root.base.st is not None and root.size == 0 \
      and not (root.base.op is Ops.CONST and root.base.arg == 0) else None),
  (UPat(Ops.DETACH, name="root"), lambda root: root.src[0]),
  (UPat(Ops.REDUCE_AXIS, name="root", src=(UPat(Ops.CONST, arg=0),)), lambda root: root.const_like(identity_element(root.arg[0], root.dtype))),
  (UPat(Ops.REDUCE_AXIS, name="root", src=(UPat.cvar("x"),)), collapse_const_reduce),
  (UPat(Ops.CONTIGUOUS, src=(UPat(Ops.VIEW, name="view", src=(UPat(set(Ops)-{Ops.CONST}, name="base"),)),)),
   lambda base,view: base.contiguous().view(unwrap(view.st)) if view.st.contiguous and view.size == base.size else None),
  (UPat(Ops.CONTIGUOUS, src=(UPat(Ops.BUFFER, name="buf"),)), lambda buf: buf),
  (UPat(Ops.SINK, name="root"),
   lambda root: root.replace(src=a) if (a:=tuple(dedup(x.base for x in root.src if x.base.op is not Ops.CONST))) != root.src else None),
])

def realize(ctx:dict[UOp, None], root:UOp):
  ctx[root.buf_uop] = None

def do_sink(ctx:dict[UOp, UOp], root:UOp):
  for x in root.src: realize(ctx, x)

def check_view(ctx:dict[UOp, UOp], root:UOp):
  if root.size <= root.buf_uop.size and all(v.mask is None for v in unwrap(root.st).views): return None
  realize(ctx, root)

class UPatBufferized(UPat):
  def __init__(self, *args, **kwargs): super().__init__(Ops.VIEW, name="root", src=(UPat(Ops.BUFFER), UPat(*args, **kwargs)))

do_realize = PatternMatcher([
  (UPat(Ops.SINK, name="root"), do_sink),
  (UPatBufferized((Ops.COPY, Ops.CONTIGUOUS)), realize),
  (UPatBufferized(Ops.REDUCE_AXIS), realize),
  (UPat(Ops.COPY, src=(UPat(), UPatBufferized(set(Ops)),)), realize),
  (UPatBufferized(set(Ops)), check_view),
])

@dataclass(frozen=True)
class SchedulerContext:
  realizes: frozenset[UOp]
  var_vals: dict[Variable, int]
  stores: dict[UOp, None]

def store_or_fuse(ctx:SchedulerContext, buf:UOp, root:UOp, st:UOp):
  if buf not in ctx.realizes: return root.view(unwrap(st.st))
  ctx.stores[buf] = UOp(Ops.STORE, dtypes.void, (buf, ShapeTracker.from_shape(root.shape).to_uop(), root))
  return UOp(Ops.LOAD, root.dtype, (buf, unwrap(st.st).to_uop()))

def load_realized(ctx:SchedulerContext, buf:UOp, st:UOp):
  return UOp(Ops.LOAD, buf.dtype, (buf, unwrap(st.st).to_uop()))

def make_valid(st:UOp, x:UOp):
  unmasked_st = ShapeTracker.from_shape(()).reshape((1,)*len(st.shape)).expand(st.shape)
  return UOp(Ops.VALID, dtypes.bool, (unwrap(st.st).to_uop(),)).where(x.replace(src=(x.src[0].replace(arg=unmasked_st),)), 0)

break_sched = PatternMatcher([
  (UPat(Ops.VIEW, name="st", src=(UPat(Ops.BUFFER, name="buf"),)), load_realized),
  (UPat(Ops.VIEW, name="st", src=(UPat(Ops.BUFFER, name="buf"), UPat.var("root"))), store_or_fuse),
  (UPat(Ops.VIEW, name="st", src=(UPat(Ops.CONST, name="x"))), make_valid),
])

def load_buffer(ctx:list[UOp], buf:UOp):
  ctx.append(buf)
  glbl = UOp(Ops.DEFINE_GLOBAL, buf.dtype.ptr(buf.size), (), len(ctx)-1)
  return UOp(Ops.LOAD, buf.dtype, (glbl, unwrap(buf.st).to_uop()))

to_ast = PatternMatcher([
  (UPat(Ops.BUFFER, name="buf"), load_buffer),
  (UPat(Ops.STORE, name="root", src=(UPat(Ops.LOAD, src=(UPat.var("glbl"), UPat())), UPat.var("st"), UPat.var("v"))), lambda root,glbl,st,v: root.replace(src=(glbl, st, v))),
  (UPat(Ops.LOAD, name="root", src=(UPat(Ops.LOAD, src=(UPat.var("glbl"), UPat())), UPat.var("st"))), lambda root,glbl,st: root.replace(src=(glbl, st))),
  (UPat(Ops.SINK, src=(UPat.store(UPat(), UPat(), UPat(Ops.COPY, name="copy")))), lambda copy:copy),
  (UPat((Ops.CONTIGUOUS, Ops.ASSIGN), name="root"), lambda root: root.src[-1]),
])

view_right = merge_views+PatternMatcher([
])

def add_buffer(u:UOp, buffer_map:dict[UOp, UOp], cache:dict[UOp, UOp]):
  if (cret:=cache.get(u)) is not None: return cret
  if u is not u.base: return add_buffer(u.base, buffer_map, cache).view(unwrap(u.st))
  if u.is_realized or u.op in {Ops.CONST, Ops.DEVICE}: return u
  src = tuple(add_buffer(x, buffer_map, cache) for x in u.src)
  if u.op is Ops.SINK: return u.replace(src=src)
  buffer_map[buf_uop:=UOp.new_buffer(u.device, u.size, u.dtype)] = u
  cache[u] = ret = UOp(Ops.VIEW, u.dtype, (buf_uop, u.replace(src=tuple(src))), ShapeTracker.from_shape(u.shape))
  return ret

@track_rewrites(named=True)
def create_schedule_with_vars(outs:list[UOp], skip_check:bool=not __debug__) -> tuple[list[ScheduleItem], dict[Variable, int], dict[UOp, UOp]]:
  sink = UOp.sink(*outs)
  if not skip_check: type_verify(list(sink.toposort), tensor_uop_spec)
  tensor_map = graph_rewrite_map(sink, merge_views+remove_movement_ops+sym)
  buffer_map: dict[UOp, UOp] = {}
  sink = add_buffer(tensor_map[sink], buffer_map, cache={})
  realizes: dict[UOp, UOp] = {}
  sink = graph_rewrite(sink, merge_views+do_realize, realizes)
  graph_rewrite(sink, merge_views+remove_movement_ops+break_sched, ctx:=SchedulerContext(frozenset(realizes), stores={}, var_vals={}))

  schedule: list[ScheduleItem] = []
  becomes_map: dict[UOp, UOp] = {}
  for buf_uop, store in ctx.stores.items():
    si_bufs: list[UOp] = []
    ast = graph_rewrite(store.sink(), view_left+to_ast, si_bufs)
    ast = graph_rewrite(ast, view_right, si_bufs)
    schedule.append(si:=ScheduleItem(ast, tuple(x.buffer for x in si_bufs), ()))
    for buf in si.bufs: buf.ref(1)
    tensor_uops = [k for k,v in tensor_map.items() if v is buffer_map[buf_uop]]
    for t in tensor_uops:
      becomes_map[t] = buf_uop.view(unwrap(t.st))

  return schedule, {}, becomes_map
