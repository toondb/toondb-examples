package main

import (
	"fmt"
	"log"

	"github.com/toondb/toondb-go"
)

func main() {
	fmt.Println("Go SDK Test")
	fmt.Println("============")

	// Open database
	db, err := toondb.Open("./test_go_db")
	if err != nil {
		log.Fatal("Failed to open database:", err)
	}
	defer db.Close()
	fmt.Println("✅ Database opened")

	// Test Put
	err = db.Put([]byte("test_key"), []byte("test_value"))
	if err != nil {
		log.Fatal("Failed to put:", err)
	}
	fmt.Println("✅ Put: test_key -> test_value")

	// Test Get
	value, err := db.Get([]byte("test_key"))
	if err != nil {
		log.Fatal("Failed to get:", err)
	}
	fmt.Printf("✅ Get: test_key = %s\n", value)

	// Test Path operations
	err = db.PutPath("users/alice/email", []byte("alice@example.com"))
	if err != nil {
		log.Fatal("Failed to put path:", err)
	}
	email, err := db.GetPath("users/alice/email")
	if err != nil {
		log.Fatal("Failed to get path:", err)
	}
	fmt.Printf("✅ Path: users/alice/email = %s\n", email)

	// Test Scan
	db.Put([]byte("tenants/acme/user1"), []byte(`{"name":"Alice"}`))
	db.Put([]byte("tenants/acme/user2"), []byte(`{"name":"Bob"}`))
	results, err := db.Scan("tenants/acme/")
	if err != nil {
		log.Fatal("Failed to scan:", err)
	}
	fmt.Printf("✅ Scan: Found %d items with prefix 'tenants/acme/'\n", len(results))

	fmt.Println("\n🎉 All Go SDK tests passed!")
}
