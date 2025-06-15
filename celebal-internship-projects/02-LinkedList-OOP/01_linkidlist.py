# Singly Linked List implementation using OOP in Python.
# Includes methods to add, print, and delete nodes with exception handling.
# Tested with a sample list to demonstrate functionality.
class Node:
    """
    Represents a node in the singly linked list.
    Each node contains data and a pointer to the next node.
    """
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """
    A LinkedList class to manage the nodes.
    Includes methods to:
    - Add a node to the end of the list
    - Print the list
    - Delete the nth node (1-based index)
    Handles edge cases using exception handling.
    """

    def __init__(self):
        self.head = None

    def add_node(self, data):
        """
        Add a node to the end of the list.
        """
        try:
            new_node = Node(data)
            if self.head is None:
                self.head = new_node
            else:
                current = self.head
                while current.next:
                    current = current.next
                current.next = new_node
        except Exception as e:
            print("Error while adding node:", e)

    def print_list(self):
        """
        Print the linked list elements in order.
        """
        try:
            if self.head is None:
                print("The list is currently empty.")
                return

            current = self.head
            while current:
                print(current.data, end=" -> ")
                current = current.next
            print("None")
        except Exception as e:
            print("Error while printing list:", e)

    def delete_nth_node(self, n):
        """
        Delete the nth node in the list (1-based index).
        Includes exception handling:
        - Empty list
        - Invalid index
        - Out of range
        """
        try:
            if self.head is None:
                raise Exception("Cannot delete from an empty list.")

            if not isinstance(n, int):
                raise TypeError("Index must be an integer.")

            if n <= 0:
                raise ValueError("Index must be 1 or greater.")

            if n == 1:
                deleted = self.head.data
                self.head = self.head.next
                print(f"Deleted: {deleted}")
                return

            current = self.head
            count = 1

            while current and count < n - 1:
                current = current.next
                count += 1

            if current is None or current.next is None:
                raise IndexError("Index is out of range.")

            deleted = current.next.data
            current.next = current.next.next
            print(f"Deleted: {deleted}")

        except (ValueError, TypeError, IndexError, Exception) as err:
            print("Error:", err)


# -------------------- TEST CASE ---------------------
# Test your implementation with a sample list

print("Test Case: Creating a sample list and deleting the 2nd node")
test_list = LinkedList()
test_list.add_node("apple")
test_list.add_node("banana")
test_list.add_node("cherry")
test_list.print_list()
test_list.delete_nth_node(2)  # should delete "banana"
test_list.print_list()
print("-" * 40)
# -------------------- END TEST CASE ------------------


# -------------------- MAIN INTERACTIVE PROGRAM --------------------

if __name__ == "__main__":
    print("Singly Linked List Demo by Ritika Choudhary")

    my_list = LinkedList()

    try:
        total_nodes = int(input("Enter the number of nodes you want to add: "))
        for i in range(total_nodes):
            value = input(f"Enter value for node {i + 1}: ")
            my_list.add_node(value)

        print("\nYour Linked List:")
        my_list.print_list()

        choice = input("\nDo you want to delete a node? (yes/no): ").lower()
        while choice == "yes":
            if my_list.head is None:
                print("The list is now empty. No more deletions possible.")
                break

            try:
                position = int(input("Enter the 1-based index of the node to delete: "))
                my_list.delete_nth_node(position)
                print("Updated Linked List:")
                my_list.print_list()
            except ValueError:
                print("Please enter a valid integer index.")

            if my_list.head is None:
                print("The list is now empty. No more deletions possible.")
                break

            choice = input("\nDo you want to delete another node? (yes/no): ").lower()

        print("\nFinal Linked List:")
        my_list.print_list()

    except ValueError:
        print("Invalid input. Please enter numeric values only.")
    except Exception as e:
        print("An unexpected error occurred:", e)
