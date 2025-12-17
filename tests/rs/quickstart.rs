// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, LargeStringArray, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use lancedb::arrow::IntoPolars;
use lancedb::database::CreateTableMode;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::{connect, table::Table};
use polars::prelude::DataFrame;
use serde::{Deserialize, Serialize};

// --8<-- [start:quickstart_define_struct]
// Define a struct representing the data schema
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Adventurer {
    id: String,
    text: String,
    vector: [f32; 3],
}
// --8<-- [end:quickstart_define_struct]

fn adventurers_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::LargeUtf8, false),
        Field::new("text", DataType::LargeUtf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3),
            false,
        ),
    ]))
}

type BatchIter = RecordBatchIterator<
    std::vec::IntoIter<std::result::Result<RecordBatch, arrow_schema::ArrowError>>,
>;

fn adventurers_to_reader(schema: Arc<Schema>, rows: &[Adventurer]) -> BatchIter {
    let ids = LargeStringArray::from_iter_values(rows.iter().map(|row| row.id.as_str()));
    let texts = LargeStringArray::from_iter_values(rows.iter().map(|row| row.text.as_str()));
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        rows.iter()
            .map(|row| Some(row.vector.iter().copied().map(Some).collect::<Vec<_>>())),
        3,
    );

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(ids), Arc::new(texts), Arc::new(vectors)],
    )
    .unwrap();

    RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema)
}

#[tokio::main]
async fn main() {
    let temp_dir = tempfile::tempdir().unwrap();
    let uri = temp_dir.path().to_str().unwrap();
    let db = connect(uri).execute().await.unwrap();

    // --8<-- [start:quickstart_create_table]
    // Define an arrow schema named adventurers_schema beforehand (omitted here for brevity)
    let schema = adventurers_schema();
    let data = vec![
        Adventurer {
            id: "1".to_string(),
            text: "knight".to_string(),
            vector: [0.9, 0.4, 0.8],
        },
        Adventurer {
            id: "2".to_string(),
            text: "ranger".to_string(),
            vector: [0.8, 0.4, 0.7],
        },
        Adventurer {
            id: "9".to_string(),
            text: "priest".to_string(),
            vector: [0.6, 0.2, 0.6],
        },
        Adventurer {
            id: "4".to_string(),
            text: "rogue".to_string(),
            vector: [0.7, 0.4, 0.7],
        },
    ];
    // Create a new table with the data, overwriting if it already exists
    let mut table = db
        .create_table("adventurers", adventurers_to_reader(schema.clone(), &data))
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();
    // --8<-- [end:quickstart_create_table]
    assert_eq!(table.count_rows(None).await.unwrap(), 4);
    db.drop_table("adventurers", &[]).await.unwrap();

    // --8<-- [start:quickstart_create_table_no_overwrite]
    table = db
        .create_table("adventurers", adventurers_to_reader(schema.clone(), &data))
        .execute()
        .await
        .unwrap();
    // --8<-- [end:quickstart_create_table_no_overwrite]
    assert_eq!(table.count_rows(None).await.unwrap(), 4);

    // --8<-- [start:quickstart_vector_search_1]
    // Let's search for vectors similar to "warrior"
    let query_vector = [0.8, 0.3, 0.8];

    let result: DataFrame = table
        .query()
        .nearest_to(&query_vector)
        .unwrap()
        .limit(2)
        .select(Select::Columns(vec!["text".to_string()]))
        .execute()
        .await
        .unwrap()
        .into_polars()
        .await
        .unwrap();
    println!("{result:?}");
    // --8<-- [end:quickstart_vector_search_1]
    let text_col = result.column("text").unwrap().str().unwrap();
    assert_eq!(text_col.get(0).unwrap(), "knight");

    // --8<-- [start:quickstart_output_array]
    let result: DataFrame = table
        .query()
        .nearest_to(&query_vector)
        .unwrap()
        .limit(2)
        .select(Select::Columns(vec!["text".to_string()]))
        .execute()
        .await
        .unwrap()
        .into_polars()
        .await
        .unwrap();
    println!("{result:?}");
    let text_col = result.column("text").unwrap().str().unwrap();
    let top_two = vec![
        text_col.get(0).unwrap().to_string(),
        text_col.get(1).unwrap().to_string(),
    ];
    // --8<-- [end:quickstart_output_array]
    assert_eq!(top_two[0], "knight");

    // --8<-- [start:quickstart_open_table]
    let table: Table = db.open_table("adventurers").execute().await.unwrap();
    // --8<-- [end:quickstart_open_table]

    // --8<-- [start:quickstart_add_data]
    let more_data = vec![
        Adventurer {
            id: "7".to_string(),
            text: "mage".to_string(),
            vector: [0.6, 0.3, 0.4],
        },
        Adventurer {
            id: "8".to_string(),
            text: "bard".to_string(),
            vector: [0.3, 0.8, 0.4],
        },
    ];

    // Add data to table
    table
        .add(adventurers_to_reader(schema.clone(), &more_data))
        .execute()
        .await
        .unwrap();
    // --8<-- [end:quickstart_add_data]
    assert_eq!(table.count_rows(None).await.unwrap(), 6);

    // --8<-- [start:quickstart_vector_search_2]
    // Let's search for vectors similar to "wizard"
    let query_vector = [0.7, 0.3, 0.5];

    let result: DataFrame = table
        .query()
        .nearest_to(&query_vector)
        .unwrap()
        .limit(2)
        .select(Select::Columns(vec!["text".to_string()]))
        .execute()
        .await
        .unwrap()
        .into_polars()
        .await
        .unwrap();
    println!("{result:?}");
    let text_col = result.column("text").unwrap().str().unwrap();
    let top_two = vec![
        text_col.get(0).unwrap().to_string(),
        text_col.get(1).unwrap().to_string(),
    ];
    // --8<-- [end:quickstart_vector_search_2]
    assert_eq!(top_two[0], "mage");
}
