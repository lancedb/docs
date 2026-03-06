// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{
    FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use lancedb::connect;
use lancedb::database::CreateTableMode;

#[tokio::main]
async fn main() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db = connect(temp_dir.path().to_str().unwrap())
        .execute()
        .await
        .unwrap();

    // --8<-- [start:create_table_from_dicts]
    struct Location {
        vector: [f32; 2],
        lat: f32,
        long: f32,
    }

    let data = vec![
        Location {
            vector: [1.1, 1.2],
            lat: 45.5,
            long: -122.7,
        },
        Location {
            vector: [0.2, 1.8],
            lat: 40.1,
            long: -74.1,
        },
    ];

    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            false,
        ),
        Field::new("lat", DataType::Float32, false),
        Field::new("long", DataType::Float32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    data.iter()
                        .map(|row| Some(row.vector.iter().copied().map(Some).collect::<Vec<_>>())),
                    2,
                ),
            ),
            Arc::new(Float32Array::from_iter_values(
                data.iter().map(|row| row.lat),
            )),
            Arc::new(Float32Array::from_iter_values(
                data.iter().map(|row| row.long),
            )),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema.clone());
    let table = db
        .create_table("test_table", reader)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();
    // --8<-- [end:create_table_from_dicts]
    assert_eq!(table.count_rows(None).await.unwrap(), 2);

    // --8<-- [start:create_table_custom_schema]
    let custom_schema = Arc::new(Schema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            false,
        ),
        Field::new("lat", DataType::Float32, false),
        Field::new("long", DataType::Float32, false),
    ]));

    let custom_batch = RecordBatch::try_new(
        custom_schema.clone(),
        vec![
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    vec![
                        Some(vec![Some(1.1), Some(1.2), Some(1.3), Some(1.4)]),
                        Some(vec![Some(0.2), Some(1.8), Some(0.4), Some(3.6)]),
                    ],
                    4,
                ),
            ),
            Arc::new(Float32Array::from(vec![45.5, 40.1])),
            Arc::new(Float32Array::from(vec![-122.7, -74.1])),
        ],
    )
    .unwrap();
    let custom_reader =
        RecordBatchIterator::new(vec![Ok(custom_batch)].into_iter(), custom_schema.clone());
    let custom_table = db
        .create_table("my_table_custom_schema", custom_reader)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();
    // --8<-- [end:create_table_custom_schema]
    assert_eq!(custom_table.count_rows(None).await.unwrap(), 2);

    // --8<-- [start:create_table_from_arrow]
    let arrow_schema = Arc::new(Schema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 16),
            false,
        ),
        Field::new("text", DataType::Utf8, false),
    ]));

    let arrow_batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    vec![Some(vec![Some(0.1); 16]), Some(vec![Some(0.2); 16])],
                    16,
                ),
            ),
            Arc::new(StringArray::from(vec!["foo", "bar"])),
        ],
    )
    .unwrap();
    let arrow_reader =
        RecordBatchIterator::new(vec![Ok(arrow_batch)].into_iter(), arrow_schema.clone());
    let arrow_table = db
        .create_table("arrow_table_example", arrow_reader)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();
    // --8<-- [end:create_table_from_arrow]
    assert_eq!(arrow_table.count_rows(None).await.unwrap(), 2);

    // --8<-- [start:create_table_from_iterator]
    let batch_schema = Arc::new(Schema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            false,
        ),
        Field::new("item", DataType::Utf8, false),
        Field::new("price", DataType::Float32, false),
    ]));

    let batches = (0..5)
        .map(|i| {
            RecordBatch::try_new(
                batch_schema.clone(),
                vec![
                    Arc::new(
                        FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                            vec![
                                Some(vec![Some(3.1 + i as f32), Some(4.1), Some(5.1), Some(6.1)]),
                                Some(vec![
                                    Some(5.9),
                                    Some(26.5 + i as f32),
                                    Some(4.7),
                                    Some(32.8),
                                ]),
                            ],
                            4,
                        ),
                    ),
                    Arc::new(StringArray::from(vec![
                        format!("item{}", i * 2 + 1),
                        format!("item{}", i * 2 + 2),
                    ])),
                    Arc::new(Float32Array::from(vec![
                        ((i * 2 + 1) * 10) as f32,
                        ((i * 2 + 2) * 10) as f32,
                    ])),
                ],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    let batch_reader = RecordBatchIterator::new(batches.into_iter().map(Ok), batch_schema.clone());
    let batch_table = db
        .create_table("batched_table", batch_reader)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();
    // --8<-- [end:create_table_from_iterator]
    assert_eq!(batch_table.count_rows(None).await.unwrap(), 10);

    // --8<-- [start:open_existing_table]
    let open_schema = Arc::new(Schema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            false,
        ),
        Field::new("lat", DataType::Float32, false),
        Field::new("long", DataType::Float32, false),
    ]));
    let open_batch = RecordBatch::try_new(
        open_schema.clone(),
        vec![
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    vec![Some(vec![Some(1.1), Some(1.2)])],
                    2,
                ),
            ),
            Arc::new(Float32Array::from(vec![45.5])),
            Arc::new(Float32Array::from(vec![-122.7])),
        ],
    )
    .unwrap();
    let open_reader =
        RecordBatchIterator::new(vec![Ok(open_batch)].into_iter(), open_schema.clone());
    db.create_table("test_table", open_reader)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();

    println!("{:?}", db.table_names().execute().await.unwrap());

    let opened_table = db.open_table("test_table").execute().await.unwrap();
    // --8<-- [end:open_existing_table]
    assert_eq!(opened_table.count_rows(None).await.unwrap(), 1);

    // --8<-- [start:create_empty_table]
    let empty_schema = Arc::new(Schema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            false,
        ),
        Field::new("item", DataType::Utf8, false),
        Field::new("price", DataType::Float32, false),
    ]));
    let empty_table = db
        .create_empty_table("test_empty_table", empty_schema)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();
    // --8<-- [end:create_empty_table]
    assert_eq!(empty_table.count_rows(None).await.unwrap(), 0);

    // --8<-- [start:drop_table]
    let drop_schema = Arc::new(Schema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            false,
        ),
        Field::new("lat", DataType::Float32, false),
    ]));
    let drop_batch = RecordBatch::try_new(
        drop_schema.clone(),
        vec![
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    vec![Some(vec![Some(1.1), Some(1.2)])],
                    2,
                ),
            ),
            Arc::new(Float32Array::from(vec![45.5])),
        ],
    )
    .unwrap();
    let drop_reader =
        RecordBatchIterator::new(vec![Ok(drop_batch)].into_iter(), drop_schema.clone());
    db.create_table("my_table", drop_reader)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();

    db.drop_table("my_table", &[]).await.unwrap();
    // --8<-- [end:drop_table]
    assert!(
        !db.table_names()
            .execute()
            .await
            .unwrap()
            .contains(&"my_table".to_string())
    );
}
