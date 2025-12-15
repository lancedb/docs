// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

use std::path::PathBuf;

use lancedb::connect;

// --8<-- [start:connect]
async fn connect_example(uri: &str) {
    let db = connect(uri).execute().await.unwrap();
    let _ = db;
}
// --8<-- [end:connect]

#[tokio::main]
async fn main() {
    let temp_dir = tempfile::tempdir().unwrap();
    let uri = temp_dir.path().join("ex_lancedb");
    connect_example(uri.to_str().unwrap()).await;

    // Keep the cloud snippet in this file, but don't run it in CI.
    let _ = connect_cloud_config();
}

fn connect_cloud_config() -> (String, String, String) {
    // --8<-- [start:connect_cloud]
    let uri = "db://your-database-uri";
    let api_key = "your-api-key";
    let region = "us-east-1";
    // --8<-- [end:connect_cloud]

    (uri.to_string(), api_key.to_string(), region.to_string())
}

#[allow(dead_code)]
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}
