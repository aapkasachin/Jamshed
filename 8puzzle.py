import copy
class Node:
    def __init__(self, state, level, f):
        self.state = state
        self.level = level
        self.f = f
    
    def findBlank(self):
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == '_':
                    return (i, j)

    def generateChildren(self):
        pos = self.findBlank()
        moves = [ (0, -1), (0, 1), (-1, 0), (1, 0)]
        children = []
        for move in moves:
            if pos[0]+move[0] > 2 or pos[0]+move[0] < 0 or pos[1]+move[1]>2 or pos[1]+move[1]<0:
                continue
            newState = copy.deepcopy(self.state)
            newState[pos[0]][pos[1]] = self.state[pos[0]+move[0]][pos[1]+move[1]]
            newState[pos[0]+move[0]][pos[1]+move[1]] = self.state[pos[0]][pos[1]]
            children.append(newState)
        return children
    
class Puzzle:
    def __init__(self):
        self.open = []
        self.closed = []

    def h(self, current, goal):
        h = 0
        for i in range(3):
            for j in range(3):
                h += (0 if current.state[i][j]==goal.state[i][j] else 1)
        return h
    
    def f(self, current, goal):
        return current.level + self.h(current, goal)
    
    def print_state(self, current):
        for row in current.state:
            print(' '.join(row))

    def main(self):
        print('Enter start state')
        startState = []
        for i in range(3):
            startState.append(input().split(' '))
        print('Enter goal state')
        goalState = []
        for i in range(3):
            goalState.append(input().split(' '))
        goal = Node(goalState, 0, 0)
        start = Node(startState, 0, 0)
        start.f = self.f(start, goal)
        self.open.append(start)
        print('-'*5)
        while True:
            current = self.open[0]
            self.print_state(current)
            print('-'*5)
            print()
            if( self.h(current, goal) == 0 ):
                break
            for child in current.generateChildren():
                childNode = Node(child, current.level+1, 0)
                for closedNode in self.closed:
                    if(self.h(closedNode, childNode) == 0):
                        continue
                childNode.f = self.f(childNode, goal)
                self.open.append(childNode)
            self.closed.append(self.open[0])
            del self.open[0]
            self.open.sort(key= lambda x: x.f, reverse=False)

if __name__=='__main__':
    puzzle = Puzzle()
    puzzle.main()