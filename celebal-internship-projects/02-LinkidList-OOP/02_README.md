# Singly Linked List in Python

This project implements a basic **Singly Linked List** using Object-Oriented Programming principles.

> 📌 Created by Ritika Choudhary  
> 🏢 Internship Project – Celebal Technologies  
> 📅 June 2025

## 💡 Description

The program creates a linked list using custom `Node` and `LinkedList` classes, and allows the user to:

- Add elements to the end of the list
- Print the current list
- Delete the nth node using a 1-based index

## 🔧 Features

- Object-Oriented Design (`Node` and `LinkedList` classes)
- User input-based list creation
- Exception Handling:
  - Trying to delete from an empty list
  - Index out of range or invalid index
- Automatic stop when the list is empty
- One built-in test case (before user input) to demonstrate functionality

## 🧪 Sample Test Case Output

```python
# Sample Input
list = ["apple", "banana", "cherry"]
delete 2nd node

# Output
apple -> banana -> cherry -> None
Deleted: banana
apple -> cherry -> None
