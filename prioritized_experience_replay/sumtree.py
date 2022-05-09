import numpy as np

class SumTree:
    items=0
    def __init__(self,capacity):
        self.capacity=capacity
        self.tree=np.zeros(2*capacity-1)
        self.data=np.zeros(capacity,dtype=object)
        self.number_of_enteries=0

    def getchangestoroot(self,indx,change_value):
        parent=(indx-1)//2
        self.tree[parent]+=change_value
        if parent!=0:
            self.getchangestoroot(parent,change_value)

    # find sample on leaf node
    def retrieve(self, indx, s):
        left = 2 * indx + 1
        right = left + 1
        if left >= len(self.tree):
            return indx
        if s <= self.tree[left]:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.items+ self.capacity - 1
        data=list(data)
        self.data[self.items] = data
        self.update(idx, p)

        self.items += 1
        if self.items >= self.capacity:
            self.items = 0

        if self.number_of_enteries < self.capacity:
            self.number_of_enteries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self.getchangestoroot(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self.retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])