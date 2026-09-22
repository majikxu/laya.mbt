///|
/// Minimal stdin/stderr access for the CLI.
///
/// `read` and `write` are bound directly rather than through a stub because
/// they are plain libc symbols — unlike `stderr`, which is a macro on some
/// platforms — and the CLI only needs to slurp stdin and put one diagnostic
/// line on fd 2.
#borrow(buffer)
extern "c" fn posix_read(
  fd : Int,
  buffer : FixedArray[Byte],
  count : Int64,
) -> Int64 = "read"

///|
#borrow(buffer)
extern "c" fn posix_write(fd : Int, buffer : Bytes, count : Int64) -> Int64 = "write"

///|
/// 64 KiB per `read`, which is one syscall for any realistic question file.
const CHUNK : Int = 65536

///|
/// Read all of standard input as UTF-8 text.
fn read_stdin() -> String raise @error.LayaError {
  let collected = Buffer()
  let chunk = FixedArray::make(CHUNK, b'\x00')
  for ;; {
    let read = posix_read(0, chunk, CHUNK.to_int64())
    if read < 0L {
      raise @error.LayaError("cannot read standard input")
    }
    if read == 0L {
      break
    }
    for index in 0..<read.to_int() {
      collected.write_byte(chunk[index])
    }
  }
  @utf8.decode_lossy(collected.to_bytes())
}

///|
/// Write one line to standard error.
fn eprintln(line : String) -> Unit {
  let bytes = @utf8.encode(line + "\n")
  // Diagnostics are best-effort: a closed stderr must not fail the run.
  ignore(posix_write(2, bytes, bytes.length().to_int64()))
}
