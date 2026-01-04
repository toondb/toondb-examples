const { Database } = require('@sushanth/toondb');

async function main() {
  const db = await Database.open('./sql_test_db');
  
  try {
    // Create table
    let result = await db.execute('CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT)');
    console.log('CREATE TABLE:', result);
    
    // Insert
    result = await db.execute("INSERT INTO products (id, name, price) VALUES (1, 'Laptop', 999.99)");
    console.log('INSERT 1:', result);
    result = await db.execute("INSERT INTO products (id, name, price) VALUES (2, 'Mouse', 29.99)");
    console.log('INSERT 2:', result);
    
    // Select
    result = await db.execute('SELECT * FROM products');
    console.log('SELECT *:', result.rows);
    
    // Select with WHERE
    result = await db.execute('SELECT name, price FROM products WHERE price > 50');
    console.log('SELECT WHERE price > 50:', result.rows);
  } finally {
    await db.close();
  }
}

main().catch(console.error);
