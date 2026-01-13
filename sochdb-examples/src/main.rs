use sochdb::Database;
use anyhow::Result;

fn main() -> Result<()> {
    println!("🗄️  SochDB Rust SDK Examples v0.4.0");
    println!("=====================================\n");

    // Open database
    let db = Database::open("./example_db")?;
    println!("✅ Database opened\n");

    // Run examples
    basic_kv_operations(&db)?;
    path_operations(&db)?;
    stats_operations(&db)?;

    println!("\n✅ All examples completed successfully!");
    Ok(())
}

fn basic_kv_operations(db: &Database) -> Result<()> {
    println!("1. Basic Key-Value Operations:");
    println!("   ----------------------------");
    
    // Put operations
    db.put(b"user:1", br#"{"name": "Alice", "age": 30}"#)?;
    db.put(b"user:2", br#"{"name": "Bob", "age": 25}"#)?;
    println!("   ✅ Put: user:1, user:2");
    
    // Get operation
    if let Some(value) = db.get(b"user:1")? {
        let data = String::from_utf8_lossy(&value);
        println!("   ✅ Get: user:1 -> {}", data);
    }
    
    // Delete operation
    db.delete(b"user:2")?;
    println!("   ✅ Delete: user:2\n");
    
    Ok(())
}

fn path_operations(db: &Database) -> Result<()> {
    println!("2. Path-Based Operations:");
    println!("   ----------------------");
    
    // Put with path (using string path format)
    db.put_path("users/alice/email", b"alice@example.com")?;
    db.put_path("users/alice/age", b"30")?;
    db.put_path("users/bob/name", b"Bob Jones")?;
    println!("   ✅ PutPath: users/alice/email, users/bob/name");
    
    // Get with path
    if let Some(value) = db.get_path("users/alice/email")? {
        println!("   ✅ GetPath: users/alice/email -> {}", String::from_utf8_lossy(&value));
    }
    
    // Demonstrate hierarchical keys with regular put/get
    for i in 1..=5 {
        let key = format!("product:{}", i);
        let value = format!("Item {}", i);
        db.put(key.as_bytes(), value.as_bytes())?;
    }
    println!("   ✅ Inserted 5 hierarchical keys (product:1-5)");
    
    println!();
    Ok(())
}

fn stats_operations(db: &Database) -> Result<()> {
    println!("3. Database Statistics:");
    println!("   --------------------");
    
    let stats = db.stats();
    println!("   📊 Queries executed: {}", stats.queries_executed);
    println!("   📊 Tables registered: {}", stats.tables_registered);
    
    Ok(())
}
