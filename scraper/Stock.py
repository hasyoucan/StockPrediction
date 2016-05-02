class Stock:
    date = ''
    start = 0
    high = 0
    low = 0
    end = 0
    volume = 0
    adj_end = 0

    def __init__(self, date, start, high, low, end, volume, adj_end):
        self.date = date
        self.start = start
        self.high = high
        self.low = low
        self.end = end
        self.volume = volume
        self.adj_end = adj_end
