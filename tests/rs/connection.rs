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
    let _ = connect_object_storage_config();
}

fn connect_cloud_config() -> (String, String, String) {
    // --8<-- [start:connect_cloud]
    let uri = "db://your-database-uri";
    let api_key = "your-api-key";
    let region = "us-east-1";
    // --8<-- [end:connect_cloud]

    (uri.to_string(), api_key.to_string(), region.to_string())
}

fn connect_object_storage_config() -> &'static str {
    // --8<-- [start:connect_object_storage]
    let uri = "s3://your-bucket/path";
    // You can also use "gs://your-bucket/path" or "az://your-container/path".
    // --8<-- [end:connect_object_storage]

    uri
}

async fn namespace_table_ops_example(uri: &str) -> lancedb::Result<()> {
    // --8<-- [start:namespace_table_ops]
    let conn = connect(uri).execute().await?;
    let namespace = vec!["prod".to_string(), "search".to_string()];

    let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("id", arrow_schema::DataType::Int64, false),
    ]));

    conn.create_empty_table("users", schema)
        .namespace(namespace.clone())
        .execute()
        .await?;

    let _table = conn
        .open_table("users")
        .namespace(namespace.clone())
        .execute()
        .await?;
    let _table_names = conn
        .table_names()
        .namespace(namespace.clone())
        .execute()
        .await?;

    conn.drop_table("users", &namespace).await?;
    // drop_all_tables is namespace-aware as well:
    // conn.drop_all_tables(&namespace).await?;
    // --8<-- [end:namespace_table_ops]
    Ok(())
}

#[allow(dead_code)]
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}
