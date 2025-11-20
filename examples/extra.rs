// --8<-- [start:install]
cargo add lancedb
// --8<-- [end:install]

// --8<-- [start:connect]
#[tokio::main]
async fn main() -> Result<()> {
    let uri = "data/sample-lancedb";
    let db = connect(uri).execute().await?;
}
// --8<-- [end:connect]

// --8<-- [start:install_preview]
[dependencies]
lancedb = { git = "https://github.com/lancedb/lancedb.git", tag = "vX.Y.Z-beta.N" }
// --8<-- [end:install_preview]