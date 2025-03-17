print("Hello, World!")

def add(a, b):
    return a + b

print(add(1, 2))

def test_add():
    assert add(1, 2) == 3
    assert add(0, 0) == 0
    assert add(-1, 1) == 0
    print("All add tests passed!")

test_add()
