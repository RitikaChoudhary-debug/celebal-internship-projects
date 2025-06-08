# Patterns with Stars by Ritika Choudhary
# This script prints lower triangular, upper triangular, and pyramid patterns
# Using the '*' character. Customize the size by entering your own value!

def print_lower_triangle(rows: int) -> None:
    """Prints a left-aligned lower triangular pattern of stars."""
    for n in range(1, rows + 1):
        print("* " * n)


def print_upper_triangle(rows: int) -> None:
    """Prints a right-aligned upper triangular pattern of stars."""
    for n in range(rows, 0, -1):
        print("  " * (rows - n) + "* " * n)


def print_pyramid(rows: int) -> None:
    """Prints a symmetric pyramid pattern of stars."""
    for n in range(1, rows + 1):
        print(" " * (rows - n) + "* " * n)


def main():
    print("Welcome to Ritika's Pattern Generator 🎨")
    try:
        rows = int(input("Enter the number of rows for your patterns (e.g., 5): "))
    except ValueError:
        print("Invalid input. Using default size 5.")
        rows = 5

    print("\n🔻 Lower Triangular Pattern:")
    print_lower_triangle(rows)

    print("\n🔺 Upper Triangular Pattern:")
    print_upper_triangle(rows)

    print("\n⛰️  Pyramid Pattern:")
    print_pyramid(rows)


if __name__ == "__main__":
    main()
