class Customer:
    """
    Customer information
    """

    def __init__(self, id: int, x: float, y: float, demand: int):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
