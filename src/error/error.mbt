///|
/// The single failure type used across `laya.mbt`.
///
/// Loading a checkpoint touches a lot of external data — JSON configs written by
/// another toolchain, a memory-mapped tensor file, a `tokenizer.json` — so almost
/// every entry point can fail for reasons that are worth reporting verbatim
/// rather than collapsing into a variant.
pub(all) suberror LayaError {
  LayaError(String)
} derive(Eq)

///|
pub impl Show for LayaError with fn output(self, logger) {
  let LayaError(message) = self
  logger.write_string(message)
}

///|
/// Raise a `LayaError` when `condition` does not hold.
pub fn expect(condition : Bool, message : String) -> Unit raise LayaError {
  if !condition {
    raise LayaError(message)
  }
}
