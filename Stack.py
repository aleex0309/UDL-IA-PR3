
class Stack:
    """
    LIFO data structure
    """

    def __init__(self):
        self.stack = []

    def push(self, item):
        """
        Appends item to stack
        """
        self.stack.append(item)

    def pop(self):
        """
        "Pop" lastly introduced item
        """
        return self.stack.pop()

    def is_empty(self):
        """
        Returns true if stack is empty
        """
        return len(self.stack) == 0
