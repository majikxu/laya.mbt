///|
/// Guard against the header parser silently accepting a checkpoint it cannot
/// read. `models/` is not in version control, so the tests that need real
/// weights live in `src/model` behind a fixture check.
test "opening a missing file reports the path" {
  try @safetensors.File::open("/nonexistent/model.safetensors") catch {
    err => inspect(err, content="cannot open /nonexistent/model.safetensors")
  } noraise {
    _ => fail("expected the open to fail")
  }
}
