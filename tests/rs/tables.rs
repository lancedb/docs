// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
    RecordBatchIterator, RecordBatchReader, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use lancedb::connect;
use lancedb::database::CreateTableMode;
use lancedb::table::{ColumnAlteration, Duration, NewColumnTransform, OptimizeAction};

// --8<-- [start:update_make_users_reader]
fn make_users_reader(
    ids: Vec<i64>,
    names: Vec<&str>,
    login_counts: Option<Vec<i64>>,
) -> Box<dyn RecordBatchReader + Send> {
    let mut fields = vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ];
    let mut columns: Vec<Arc<dyn Array>> =
        vec![Arc::new(Int64Array::from(ids)), Arc::new(StringArray::from(names))];

    if let Some(login_counts) = login_counts {
        fields.push(Field::new("login_count", DataType::Int64, true));
        columns.push(Arc::new(Int64Array::from(login_counts)));
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
    Box::new(reader)
}
// --8<-- [end:update_make_users_reader]

#[allow(dead_code)]
async fn update_connect_cloud_example() {
    // --8<-- [start:update_connect_cloud]
    let db = connect("db://your-project-slug")
        .api_key("your-api-key")
        .region("us-east-1")
        .execute()
        .await
        .unwrap();
    // --8<-- [end:update_connect_cloud]
    let _ = db;
}

#[allow(dead_code)]
async fn update_connect_local_example() {
    // --8<-- [start:update_connect_local]
    let db = connect("./data").execute().await.unwrap();
    // --8<-- [end:update_connect_local]
    let _ = db;
}

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

    // --8<-- [start:schema_add_setup]
    let schema_add_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 128),
            false,
        ),
    ]));
    let schema_add_batch = RecordBatch::try_new(
        schema_add_schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Laptop", "Smartphone", "Headphones"])),
            Arc::new(Float64Array::from(vec![1200.0, 800.0, 150.0])),
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    vec![
                        Some(vec![Some(0.1_f32); 128]),
                        Some(vec![Some(0.2_f32); 128]),
                        Some(vec![Some(0.3_f32); 128]),
                    ],
                    128,
                ),
            ),
        ],
    )
    .unwrap();
    let schema_add_reader = RecordBatchIterator::new(
        vec![Ok(schema_add_batch)].into_iter(),
        schema_add_schema.clone(),
    );
    let schema_add_table = db
        .create_table("schema_evolution_add_example", schema_add_reader)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();
    // --8<-- [end:schema_add_setup]
    assert_eq!(schema_add_table.count_rows(None).await.unwrap(), 3);

    // --8<-- [start:add_columns_calculated]
    // Add a discounted price column (10% discount)
    schema_add_table
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![(
                "discounted_price".to_string(),
                "cast((price * 0.9) as float)".to_string(),
            )]),
            None,
        )
        .await
        .unwrap();
    // --8<-- [end:add_columns_calculated]

    // --8<-- [start:add_columns_default_values]
    // Add a stock status column with default value
    schema_add_table
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![(
                "in_stock".to_string(),
                "cast(true as boolean)".to_string(),
            )]),
            None,
        )
        .await
        .unwrap();
    // --8<-- [end:add_columns_default_values]

    // --8<-- [start:add_columns_nullable]
    // Add a nullable timestamp column
    schema_add_table
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![(
                "last_ordered".to_string(),
                "cast(NULL as timestamp)".to_string(),
            )]),
            None,
        )
        .await
        .unwrap();
    // --8<-- [end:add_columns_nullable]

    // --8<-- [start:schema_alter_setup]
    let schema_alter_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("price", DataType::Int32, false),
        Field::new("discount_price", DataType::Float64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 128),
            false,
        ),
    ]));
    let schema_alter_batch = RecordBatch::try_new(
        schema_alter_schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["Laptop", "Smartphone"])),
            Arc::new(Int32Array::from(vec![1200, 800])),
            Arc::new(Float64Array::from(vec![1080.0, 720.0])),
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    vec![Some(vec![Some(0.1_f32); 128]), Some(vec![Some(0.2_f32); 128])],
                    128,
                ),
            ),
        ],
    )
    .unwrap();
    let schema_alter_reader = RecordBatchIterator::new(
        vec![Ok(schema_alter_batch)].into_iter(),
        schema_alter_schema.clone(),
    );
    let schema_alter_table = db
        .create_table("schema_evolution_alter_example", schema_alter_reader)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();
    // --8<-- [end:schema_alter_setup]
    assert_eq!(schema_alter_table.count_rows(None).await.unwrap(), 2);

    // --8<-- [start:alter_columns_rename]
    // Rename discount_price to sale_price
    schema_alter_table
        .alter_columns(&[ColumnAlteration::new("discount_price".to_string())
            .rename("sale_price".to_string())])
        .await
        .unwrap();
    // --8<-- [end:alter_columns_rename]

    // --8<-- [start:alter_columns_data_type]
    // Change price from int32 to int64 for larger numbers
    schema_alter_table
        .alter_columns(&[ColumnAlteration::new("price".to_string()).cast_to(DataType::Int64)])
        .await
        .unwrap();
    // --8<-- [end:alter_columns_data_type]

    // --8<-- [start:alter_columns_nullable]
    // Make the name column nullable
    schema_alter_table
        .alter_columns(&[ColumnAlteration::new("name".to_string()).set_nullable(true)])
        .await
        .unwrap();
    // --8<-- [end:alter_columns_nullable]

    // --8<-- [start:alter_columns_multiple]
    // Rename, change type, and make nullable in one operation
    schema_alter_table
        .alter_columns(&[ColumnAlteration::new("sale_price".to_string())
            .rename("final_price".to_string())
            .cast_to(DataType::Float64)
            .set_nullable(true)])
        .await
        .unwrap();
    // --8<-- [end:alter_columns_multiple]

    // --8<-- [start:alter_columns_with_expression]
    // For custom transforms, create a new column from a SQL expression.
    let expression_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("price_text", DataType::Utf8, false),
    ]));
    let expression_batch = RecordBatch::try_new(
        expression_schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(StringArray::from(vec!["$100"])),
        ],
    )
    .unwrap();
    let expression_reader = RecordBatchIterator::new(
        vec![Ok(expression_batch)].into_iter(),
        expression_schema.clone(),
    );
    let expression_table = db
        .create_table("schema_evolution_expression_example", expression_reader)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();

    expression_table
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![(
                "price_numeric".to_string(),
                "cast(replace(price_text, '$', '') as int)".to_string(),
            )]),
            None,
        )
        .await
        .unwrap();
    expression_table.drop_columns(&["price_text"]).await.unwrap();
    expression_table
        .alter_columns(&[ColumnAlteration::new("price_numeric".to_string())
            .rename("price".to_string())])
        .await
        .unwrap();
    // --8<-- [end:alter_columns_with_expression]
    assert_eq!(expression_table.count_rows(None).await.unwrap(), 1);

    // --8<-- [start:schema_drop_setup]
    let schema_drop_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
        Field::new("temp_col1", DataType::Utf8, false),
        Field::new("temp_col2", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 128),
            false,
        ),
    ]));
    let schema_drop_batch = RecordBatch::try_new(
        schema_drop_schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Laptop", "Smartphone", "Headphones"])),
            Arc::new(Float64Array::from(vec![1200.0, 800.0, 150.0])),
            Arc::new(StringArray::from(vec!["X", "Y", "Z"])),
            Arc::new(Int32Array::from(vec![100, 200, 300])),
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    vec![
                        Some(vec![Some(0.1_f32); 128]),
                        Some(vec![Some(0.2_f32); 128]),
                        Some(vec![Some(0.3_f32); 128]),
                    ],
                    128,
                ),
            ),
        ],
    )
    .unwrap();
    let schema_drop_reader = RecordBatchIterator::new(
        vec![Ok(schema_drop_batch)].into_iter(),
        schema_drop_schema.clone(),
    );
    let schema_drop_table = db
        .create_table("schema_evolution_drop_example", schema_drop_reader)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();
    // --8<-- [end:schema_drop_setup]
    assert_eq!(schema_drop_table.count_rows(None).await.unwrap(), 3);

    // --8<-- [start:drop_columns_single]
    // Remove the first temporary column
    schema_drop_table.drop_columns(&["temp_col1"]).await.unwrap();
    // --8<-- [end:drop_columns_single]

    // --8<-- [start:drop_columns_multiple]
    // Remove the second temporary column
    schema_drop_table.drop_columns(&["temp_col2"]).await.unwrap();
    // --8<-- [end:drop_columns_multiple]

    // --8<-- [start:alter_vector_column]
    let old_dim = 384;
    let new_dim = 1024;
    let vector_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                old_dim,
            ),
            true,
        ),
    ]));
    let vector_batch = RecordBatch::try_new(
        vector_schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    vec![Some(vec![Some(0.1_f32); old_dim as usize])],
                    old_dim,
                ),
            ),
        ],
    )
    .unwrap();
    let vector_reader =
        RecordBatchIterator::new(vec![Ok(vector_batch)].into_iter(), vector_schema.clone());
    let vector_table = db
        .create_table("vector_alter_example", vector_reader)
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();

    // Changing FixedSizeList dimensions (384 -> 1024) is not supported via alter_columns.
    // Use add_columns + drop_columns + alter_columns(rename) to replace the column.
    vector_table
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![(
                "embedding_v2".to_string(),
                format!("arrow_cast(NULL, 'FixedSizeList({}, Float32)')", new_dim),
            )]),
            None,
        )
        .await
        .unwrap();
    vector_table.drop_columns(&["embedding"]).await.unwrap();
    vector_table
        .alter_columns(&[ColumnAlteration::new("embedding_v2".to_string())
            .rename("embedding".to_string())])
        .await
        .unwrap();
    // --8<-- [end:alter_vector_column]
    assert_eq!(vector_table.count_rows(None).await.unwrap(), 1);

    // --8<-- [start:update_example_table_setup]
    let table = db
        .create_table(
            "users_example",
            make_users_reader(vec![1, 2], vec!["Alice", "Bob"], Some(vec![10, 20])),
        )
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();
    // --8<-- [end:update_example_table_setup]
    let _ = table;

    // --8<-- [start:update_operation]
    let table = db
        .create_table(
            "users_example",
            make_users_reader(vec![1, 2], vec!["Alice", "Bob"], Some(vec![10, 20])),
        )
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();
    table
        .update()
        .only_if("id = 2")
        .column("name", "'Bobby'")
        .execute()
        .await
        .unwrap();
    // --8<-- [end:update_operation]

    // --8<-- [start:update_using_sql]
    let table = db
        .create_table(
            "users_example",
            make_users_reader(vec![1, 2], vec!["Alice", "Bob"], Some(vec![10, 20])),
        )
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();
    table
        .update()
        .only_if("id = 2")
        .column("login_count", "login_count + 1")
        .execute()
        .await
        .unwrap();
    // --8<-- [end:update_using_sql]

    // --8<-- [start:merge_matched_update_only]
    let table = db
        .create_table(
            "users_example",
            make_users_reader(vec![1, 2], vec!["Alice", "Bob"], Some(vec![10, 20])),
        )
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();

    let mut merge_insert = table.merge_insert(&["id"]);
    merge_insert.when_matched_update_all(None);
    merge_insert
        .execute(make_users_reader(
            vec![2, 3],
            vec!["Bobby", "Charlie"],
            Some(vec![21, 5]),
        ))
        .await
        .unwrap();
    // --8<-- [end:merge_matched_update_only]

    // --8<-- [start:insert_if_not_exists]
    let table = db
        .create_table(
            "users_example",
            make_users_reader(vec![1, 2], vec!["Alice", "Bob"], Some(vec![10, 20])),
        )
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();

    let mut merge_insert = table.merge_insert(&["id"]);
    merge_insert.when_not_matched_insert_all();
    merge_insert
        .execute(make_users_reader(
            vec![2, 3],
            vec!["Bobby", "Charlie"],
            Some(vec![21, 5]),
        ))
        .await
        .unwrap();
    // --8<-- [end:insert_if_not_exists]

    // --8<-- [start:merge_update_insert]
    let table = db
        .create_table(
            "users_example",
            make_users_reader(vec![1, 2], vec!["Alice", "Bob"], Some(vec![10, 20])),
        )
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();

    let mut merge_insert = table.merge_insert(&["id"]);
    merge_insert
        .when_matched_update_all(None)
        .when_not_matched_insert_all();
    merge_insert
        .execute(make_users_reader(
            vec![2, 3],
            vec!["Bobby", "Charlie"],
            Some(vec![21, 5]),
        ))
        .await
        .unwrap();
    // --8<-- [end:merge_update_insert]

    // --8<-- [start:merge_delete_missing_by_source]
    let table = db
        .create_table(
            "users_example",
            make_users_reader(
                vec![1, 2, 3],
                vec!["Alice", "Bob", "Charlie"],
                Some(vec![10, 20, 5]),
            ),
        )
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();

    let mut merge_insert = table.merge_insert(&["id"]);
    merge_insert
        .when_matched_update_all(None)
        .when_not_matched_insert_all()
        .when_not_matched_by_source_delete(None);
    merge_insert
        .execute(make_users_reader(
            vec![2, 3],
            vec!["Bobby", "Charlie"],
            Some(vec![21, 5]),
        ))
        .await
        .unwrap();
    // --8<-- [end:merge_delete_missing_by_source]

    // --8<-- [start:merge_partial_columns]
    let table = db
        .create_table(
            "users_example",
            make_users_reader(vec![1, 2], vec!["Alice", "Bob"], Some(vec![10, 20])),
        )
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();

    let mut merge_insert = table.merge_insert(&["id"]);
    merge_insert
        .when_matched_update_all(None)
        .when_not_matched_insert_all();
    merge_insert
        .execute(make_users_reader(vec![2, 3], vec!["Bobby", "Charlie"], None))
        .await
        .unwrap();
    // --8<-- [end:merge_partial_columns]

    let table = db
        .create_table(
            "users_example",
            make_users_reader(
                vec![1, 2, 3],
                vec!["Alice", "Bob", "Charlie"],
                Some(vec![10, 20, 5]),
            ),
        )
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();

    // --8<-- [start:delete_operation]
    // delete data
    let predicate = "id = 3";
    table.delete(predicate).await.unwrap();
    // --8<-- [end:delete_operation]

    let table = db
        .create_table(
            "users_cleanup_example",
            make_users_reader(
                vec![1, 2, 3],
                vec!["Alice", "Bob", "Charlie"],
                Some(vec![10, 20, 5]),
            ),
        )
        .mode(CreateTableMode::Overwrite)
        .execute()
        .await
        .unwrap();

    // --8<-- [start:update_optimize_cleanup]
    table
        .optimize(OptimizeAction::Prune {
            older_than: Some(Duration::days(1)),
            delete_unverified: None,
            error_if_tagged_old_versions: None,
        })
        .await
        .unwrap();
    // --8<-- [end:update_optimize_cleanup]
}
