import inspect

def print_public_interface(cls):
    """
    Prints the public interface of a Python class, including methods and attributes
    that are intended for external use (i.e., do not start with an underscore).

    Args:
        cls: The class object to inspect.
    """
    print(f"Public Interface of class: {cls.__name__}")
    print("-" * 40)

    public_methods = []
    public_attributes = []

    for name, member in inspect.getmembers(cls):
        if name == '__init__':
            print("\nConstructor:")
            signature = inspect.signature(member)
            docstring = inspect.getdoc(member) or "No docstring available."
            print(f"  - {cls.__name__}{signature}")
            print(f"    Docstring: {docstring.replace('\n', '\n      ')}\n")

        if name.startswith('_'):
            continue  # Skip members starting with underscore (considered non-public)

        if inspect.isfunction(member) or inspect.ismethod(member):
            public_methods.append(name)
        elif not inspect.isroutine(member): # Avoid properties being listed as routines
            public_attributes.append(name)

    if public_methods:
        print("\nPublic Methods:")
        for method_name in sorted(public_methods):
            method = getattr(cls, method_name)
            signature = inspect.signature(method)
            docstring = inspect.getdoc(method) or "No docstring available."
            print(f"  - {method_name}{signature}")
            print(f"    Docstring: {docstring.replace('\n', '\n      ')}\n")

    if public_attributes:
        print("\nPublic Attributes:")
        for attr_name in sorted(public_attributes):
            attribute = getattr(cls, attr_name)
            docstring = inspect.getdoc(attribute) or "No docstring available." # Try to get docstring for attributes too, if possible.
            print(f"  - {attr_name}")
            if docstring:
                print(f"    Docstring: {docstring}\n")
            else:
                print()

    print("-" * 40)
    print("\n\n")

# Example Usage (using the TrainingAgent class from your code):
if __name__ == '__main__':
    # Assuming your TrainingAgent class is defined in this file or imported
    # from where it's defined.
    from model.training_agent import TrainingAgent
    print_public_interface(TrainingAgent)

    # Example with the Transaction class (assuming it's also defined or imported)
    from model.portfolio import Transaction # Assuming Transaction is in portfolio.py
    print_public_interface(Transaction)