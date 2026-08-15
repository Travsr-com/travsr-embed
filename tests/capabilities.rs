// Issue #6: the `--capabilities` handshake is a cross-process contract with the
// travsr CLI, so it is tested across a real process boundary. The unit tests in
// src/backend/mod.rs cover what the payload *says*; these cover that the binary
// prints it on stdout, exits 0, and leaves the older `--version` probe and the
// unknown-argument rejection intact — the two behaviours the CLI's capability
// negotiation is built on.

use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_travsr-embed"))
}

#[test]
fn capabilities_exits_zero_with_json_on_stdout() {
    let out = bin()
        .arg("--capabilities")
        .output()
        .expect("run travsr-embed --capabilities");
    assert!(
        out.status.success(),
        "expected exit 0, got {:?}; stderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8(out.stdout).expect("stdout is utf-8");
    let v: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout is not valid JSON ({e}): {stdout:?}"));

    // Every key the CLI reads. Asserted by name because dropping or renaming one
    // silently breaks a consumer in the other repo.
    assert_eq!(v["capabilities_version"], 1);
    assert_eq!(v["plugin_version"], env!("CARGO_PKG_VERSION"));
    assert!(v["engines"].is_array(), "{v}");
    assert!(v["families"].is_array(), "{v}");
    assert!(v["universal_onnx"].is_boolean(), "{v}");
    assert!(v["accelerated_compiled"].is_boolean(), "{v}");

    // tract is compiled into every build, so these hold regardless of features.
    let engines = v["engines"].as_array().unwrap();
    assert!(
        engines.iter().any(|e| e == "tract"),
        "tract must always be registered: {v}"
    );
    let families = v["families"].as_array().unwrap();
    assert!(families.iter().any(|f| f == "bert"), "{v}");
}

/// The handshake must not disturb the WS3-era probe that came before it.
#[test]
fn version_probe_still_works() {
    let out = bin().arg("--version").output().expect("run --version");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.starts_with("travsr-embed "),
        "unexpected --version output: {stdout:?}"
    );
}

/// How a NEW CLI detects an OLD sidecar: the unknown flag exits non-zero. If
/// travsr-embed ever started tolerating unknown arguments, that probe would
/// read a silent success and assume capabilities it cannot deliver.
#[test]
fn unknown_flag_still_exits_nonzero() {
    let out = bin()
        .arg("--definitely-not-a-real-flag")
        .output()
        .expect("run unknown flag");
    assert!(
        !out.status.success(),
        "unknown arguments must be rejected so version probing stays meaningful"
    );
}
