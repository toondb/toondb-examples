use toondb::Database;

fn main() {
    println!("Rust SDK Test - KV Operations");
    println!("==============================");
    
    let db = Database::open("./rust_kv_db").expect("Failed to open db");
    
    // Basic KV operations
    db.put(b"key1", b"value1").expect("put failed");
    println!("Put: key1 -> value1");
    
    let val = db.get(b"key1").expect("get failed");
    println!("Get: key1 = {:?}", val.map(|v| String::from_utf8_lossy(&v).to_string()));
    
    println!("\nNote: Rust SDK uses direct KV operations, no SQL execute method found");
}
