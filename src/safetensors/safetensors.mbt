///|
/// Where one tensor lives inside a safetensors file, and in what layout.
///
/// `begin` and `end` are absolute file offsets, already shifted past the 8-byte
/// length prefix and the JSON header, so they can be handed to the mapping
/// without further arithmetic.
pub struct TensorInfo {
  name : String
  dtype : String
  shape : Array[Int]
  begin : Int64
  end : Int64
} derive(Debug, Eq)

///|
/// Number of scalars in the tensor, from its shape.
pub fn TensorInfo::elements(self : TensorInfo) -> Int64 {
  self.shape.fold(init=1L, (acc, dim) => acc * dim.to_int64())
}

///|
/// Bytes one scalar of this tensor occupies.
pub fn TensorInfo::item_size(self : TensorInfo) -> Int raise @error.LayaError {
  match self.dtype {
    "F16" | "BF16" => 2
    "F32" => 4
    "F64" => 8
    other =>
      raise @error.LayaError(
        "tensor \{self.name} has unsupported dtype \{other}; laya.mbt reads F16, BF16, F32 and F64",
      )
  }
}

///|
/// An opened safetensors file: the reader plus its parsed header.
pub struct File {
  reader : @tensor.Reader
  tensors : Map[String, TensorInfo]
}

///|
/// Open and parse a safetensors file. Tensor payloads are not touched here.
pub fn File::open(path : String) -> File raise @error.LayaError {
  let reader = @tensor.Reader::open(path)
  let header = @utf8.decode_lossy(reader.header())
  let json = @json.parse(header) catch {
    _ =>
      raise @error.LayaError("\{path} has a malformed safetensors JSON header")
  }
  guard json is Object(fields) else {
    raise @error.LayaError("\{path} safetensors header is not a JSON object")
  }
  let data_start = reader.data_start()
  let tensors = Map([])
  for name, value in fields {
    // Producers may append arbitrary string metadata under this key.
    if name == "__metadata__" {
      continue
    }
    tensors[name] = parse_entry(path, name, value, data_start)
  }
  { reader, tensors, }
}

///|
fn parse_entry(
  path : String,
  name : String,
  value : Json,
  data_start : Int64,
) -> TensorInfo raise @error.LayaError {
  guard value
    is Object(
      {
        "dtype": String(dtype),
        "shape": Array(shape),
        "data_offsets": Array([Number(begin, ..), Number(end, ..)]),
        ..
      }
    ) else {
    raise @error.LayaError(
      "\{path} header entry \{name} is missing dtype, shape or data_offsets",
    )
  }
  let dims = Array(capacity=shape.length())
  for dim in shape {
    guard dim is Number(dim, ..) else {
      raise @error.LayaError("tensor \{name} has a non-numeric shape entry")
    }
    dims.push(dim.to_int())
  }
  {
    name,
    dtype,
    shape: dims,
    begin: data_start + begin.to_int64(),
    end: data_start + end.to_int64(),
  }
}

///|
/// Names of every tensor in the file, in header order.
pub fn File::names(self : File) -> Array[String] {
  self.tensors.keys().collect()
}

///|
/// Number of tensors in the file.
pub fn File::length(self : File) -> Int {
  self.tensors.length()
}

///|
/// Look up a tensor, failing with the name that was missing.
pub fn File::info(
  self : File,
  name : String,
) -> TensorInfo raise @error.LayaError {
  match self.tensors.get(name) {
    Some(info) => info
    None => raise @error.LayaError("checkpoint has no tensor named \{name}")
  }
}

///|
/// Whether the file contains `name`.
pub fn File::contains(self : File, name : String) -> Bool {
  self.tensors.contains(name)
}

///|
/// Look up a tensor and check its shape, which catches a checkpoint that was
/// built for a different configuration before any arithmetic happens.
pub fn File::info_shaped(
  self : File,
  name : String,
  shape : Array[Int],
) -> TensorInfo raise @error.LayaError {
  let info = self.info(name)
  if info.shape != shape {
    raise @error.LayaError(
      "tensor \{name} has shape \{Repr(info.shape)} but the configuration implies \{Repr(shape)}",
    )
  }
  info
}

///|
/// Decode a tensor into `arena` at element offset `dst`, converting to float32.
pub fn File::read_into_arena(
  self : File,
  info : TensorInfo,
  arena : @tensor.Arena,
  dst : Int64,
) -> Unit raise @error.LayaError {
  let count = info.elements()
  self.check_span(info, count)
  match info.dtype {
    "F16" => self.reader.read_f16(info.begin, count, arena, dst)
    "F32" => self.reader.read_f32(info.begin, count, arena, dst)
    other =>
      raise @error.LayaError(
        "tensor \{info.name} is \{other}; only F16 and F32 can be loaded into the arena",
      )
  }
}

///|
/// Decode a tensor into a MoonBit array, which must match its element count.
pub fn File::read_into_array(
  self : File,
  info : TensorInfo,
  out : FixedArray[Float],
) -> Unit raise @error.LayaError {
  let count = info.elements()
  @error.expect(
    out.length().to_int64() == count,
    "tensor \{info.name} has \{count} elements but the destination holds \{out.length()}",
  )
  self.check_span(info, count)
  match info.dtype {
    "F16" => self.reader.read_f16_into(info.begin, out)
    other =>
      raise @error.LayaError(
        "tensor \{info.name} is \{other}; only F16 can be read into an array",
      )
  }
}

///|
/// Gather rows of a float16 matrix straight out of the mapping.
///
/// Used for the token embedding, which is by far the largest tensor in a Laya
/// checkpoint and is only ever indexed, never multiplied.
pub fn File::gather_rows(
  self : File,
  info : TensorInfo,
  rows : FixedArray[Int],
  out : FixedArray[Float],
) -> Unit raise @error.LayaError {
  @error.expect(
    info.shape.length() == 2,
    "tensor \{info.name} is not a matrix, cannot gather rows",
  )
  @error.expect(
    info.dtype == "F16",
    "row gather needs an F16 tensor, \{info.name} is \{info.dtype}",
  )
  self.reader.gather_rows_f16(
    info.begin,
    rows,
    info.shape[1],
    info.shape[0],
    out,
  )
}

///|
fn File::check_span(
  self : File,
  info : TensorInfo,
  count : Int64,
) -> Unit raise @error.LayaError {
  let expected = count * info.item_size().to_int64()
  @error.expect(
    info.end - info.begin == expected,
    "tensor \{info.name} spans \{info.end - info.begin} bytes but its shape and dtype imply \{expected}",
  )
  @error.expect(
    info.end <= self.reader.size(),
    "tensor \{info.name} extends past the end of the file",
  )
}
