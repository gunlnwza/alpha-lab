class IntegerDescriptor:
    def __init__(self, default_value=None):
        self.default_value = default_value

    def __setattr__(self, name, value):
        if name == 'my_attribute':
            if not isinstance(value, int):
                raise ValueError("Attribute 'my_attribute' must be an integer")
        # Use object.__setattr__ for safe internal assignment
        object.__setattr__(self, name, value)

    def __init__(self, value=0):
        # This will trigger our custom __setattr__
        self.my_attribute = value

# Usage:
obj = IntegerDescriptor(10)
print(obj.my_attribute)  # Output: 10

try:
    obj.my_attribute = "a string"
except ValueError as e:
    print(e) # Output: Attribute 'my_attribute' must be an integer
