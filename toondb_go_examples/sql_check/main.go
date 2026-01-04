package main

import (
	"fmt"
	"log"

	"github.com/toondb/toondb-go"
)

func main() {
	fmt.Println("Go SQL Test")
	fmt.Println("===========")

	db, err := toondb.Open("./sql_go_db")
	if err != nil {
		log.Fatal("Failed to open:", err)
	}
	defer db.Close()

	// Test SQL Execute
	result, err := db.Execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, salary FLOAT)")
	if err != nil {
		fmt.Println("CREATE TABLE error:", err)
	} else {
		fmt.Println("CREATE TABLE:", result)
	}

	result, err = db.Execute("INSERT INTO employees (id, name, salary) VALUES (1, 'Alice', 75000)")
	if err != nil {
		fmt.Println("INSERT error:", err)
	} else {
		fmt.Println("INSERT:", result)
	}

	result, err = db.Execute("SELECT * FROM employees")
	if err != nil {
		fmt.Println("SELECT error:", err)
	} else {
		fmt.Println("SELECT *:", result)
	}
}
