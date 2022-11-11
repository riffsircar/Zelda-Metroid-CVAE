'''

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import pathfinding

def makeIsSolid(solids):
    def isSolid(tile):
        return tile in solids
    return isSolid

def makeIsPassable(passables, hazards):
	def isPassable(tile):
		return tile in passables and tile not in hazards
	return isPassable

def makeIsHazard(hazards):
	def isHazard(tile):
		return tile in hazards
	return isHazard

def makeIsClimbable(climbables):
	def isClimbable(tile):
		return tile in climbables
	return isClimbable

#Make sure we are getting the proper neighbors and that all checks are happending appropriately
def makeGetNeighborsFull(jumps,levelStr,visited,isSolid,isPassable,isClimbable,isHazard,level_wrap):
	maxX = len(levelStr[0])
	maxY = len(levelStr)-1
	jumpDiffs = []
	for jump in jumps:
		jumpDiff = [jump[0]]
		for ii in range(1,len(jump)):
			jumpDiff.append((jump[ii][0]-jump[ii-1][0],jump[ii][1]-jump[ii-1][1]))
		jumpDiffs.append(jumpDiff)
	jumps = jumpDiffs

	def getNeighbors(pos):
		dist = pos[0]-pos[2] 
		pos = pos[1] 
		visited.add((pos[0],pos[1])) 
		below = (pos[0],pos[1]+1) 

		neighbors = []
		#if the player falls to the bottom of the level
		if below[1] > maxY or isHazard(levelStr[below[1]][below[0]]):
			return []
		if pos[2] != -1:
			ii = pos[3] +1
			jump = pos[2]

			if ii < len(jumps[jump]):

				if level_wrap:
					if  (pos[1]+jumps[jump][ii][1] >= 0) and (not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+pos[4]*jumps[jump][ii][0])%maxX]) or isPassable((levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+pos[4]*jumps[jump][ii][0])%maxX]))):
						neighbors.append([dist+1,((pos[0]+pos[4]*jumps[jump][ii][0])%maxX,pos[1]+jumps[jump][ii][1],jump,ii,pos[4])])
						#print("mid jump")
					if 	pos[1]+jumps[jump][ii][1] < 0 and (not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+pos[4]*jumps[jump][ii][0])%maxX]) or isPassable(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+pos[4]*jumps[jump][ii][0])%maxX])):
						neighbors.append([dist+1,((pos[0]+pos[4]*jumps[jump][ii][0])%maxX,0,jump,ii,pos[4])])
						#print("mid fall")
				else:
					if  pos[1]+jumps[jump][ii][1] >= 0 and (pos[0]+pos[4]*jumps[jump][ii][0]) < maxX and (pos[0]+pos[4]*jumps[jump][ii][0]) >= 0 and (not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+pos[4]*jumps[jump][ii][0])]) or  isPassable((levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+pos[4]*jumps[jump][ii][0])]))):
						neighbors.append([dist+1,((pos[0]+pos[4]*jumps[jump][ii][0]),pos[1]+jumps[jump][ii][1],jump,ii,pos[4])])
						#print("mid jump")
					if 	pos[1]+jumps[jump][ii][1] < 0 and (pos[0]+pos[4]*jumps[jump][ii][0]) < maxX and (pos[0]+pos[4]*jumps[jump][ii][0]) >= 0 and (not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][pos[0]+pos[4]*jumps[jump][ii][0]]) or isPassable(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+pos[4]*jumps[jump][ii][0])])):
						neighbors.append([dist+1,(pos[0]+pos[4]*jumps[jump][ii][0],0,jump,ii,pos[4])])
						#print("mid fall")
				
		if isSolid(levelStr[below[1]][below[0]]) and not isHazard(levelStr[below[1]][below[0]]):
			if level_wrap:
				if not isSolid(levelStr[pos[1]][(pos[0]+1)%maxX]):
					neighbors.append([dist+1,((pos[0]+1)%maxX,pos[1],-1)])
					#print("move right")
				if not isSolid(levelStr[pos[1]][(pos[0]-1)%maxX]):
					neighbors.append([dist+1,((pos[0]-1)%maxX,pos[1],-1)])
					#print("move left")

				for jump in range(len(jumps)):
					ii = 0
					if pos[1] >= 0 and (not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+jumps[jump][ii][0])%maxX]) or isPassable(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]+jumps[jump][ii][0])%maxX])):
						neighbors.append([dist+ii+1,((pos[0]+jumps[jump][ii][0])%maxX,pos[1]+jumps[jump][ii][1],jump,ii,1)])
						#print("start jump right")

					if pos[1] >= 0 and (not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]-jumps[jump][ii][0])%maxX]) or isPassable(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]-jumps[jump][ii][0])%maxX])):
						neighbors.append([dist+ii+1,((pos[0]-jumps[jump][ii][0])%maxX,pos[1]+jumps[jump][ii][1],jump,ii,-1)])
						#print("start jump left")
			else:
				if pos[0]+1 < maxX and not isSolid(levelStr[pos[1]][(pos[0]+1)]):
					neighbors.append([dist+1,((pos[0]+1),pos[1],-1)])
					#print("move right")
				if pos[0]-1 >= 0 and not isSolid(levelStr[pos[1]][(pos[0]-1)]):
					neighbors.append([dist+1,((pos[0]-1),pos[1],-1)])
					#print("move left")

				for jump in range(len(jumps)):
					ii = 0

					#print(pos[1]+jumps[jump][ii][1], pos[0]-jumps[jump][ii][0], len(levelStr), len(levelStr[0]))
					#print(levelStr[pos[1]+jumps[jump][ii][1]])
					#print(isSolid(levelStr[pos[1]+jumps[jump][ii][1]][pos[0]-jumps[jump][ii][0]]))
					#print(isPassable(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]-jumps[jump][ii][0])]))

					if (pos[1] >= 0 and (pos[0]+jumps[jump][ii][0]) < maxX and (pos[0]+jumps[jump][ii][0]) >=0) and (not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][pos[0]+jumps[jump][ii][0]]) or isPassable(levelStr[pos[1]+jumps[jump][ii][1]][pos[0]+jumps[jump][ii][0]])):
						neighbors.append([dist+ii+1,(pos[0]+jumps[jump][ii][0],pos[1]+jumps[jump][ii][1],jump,ii,1)])
						#print("start jump right")

					if (pos[1] >= 0 and (pos[0]-jumps[jump][ii][0]) < maxX and (pos[0]-jumps[jump][ii][0]) >= 0) and (not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][pos[0]-jumps[jump][ii][0]]) or isPassable(levelStr[pos[1]+jumps[jump][ii][1]][(pos[0]-jumps[jump][ii][0])])):
						neighbors.append([dist+ii+1,(pos[0]-jumps[jump][ii][0],pos[1]+jumps[jump][ii][1],jump,ii,-1)])

		if not isSolid(levelStr[below[1]][below[0]]) or isPassable(levelStr[below[1]][below[0]]):
			neighbors.append([dist+1,(pos[0],pos[1]+1,-1)])
			if level_wrap:
				if pos[1]+1 <= maxY:
					if not isSolid(levelStr[pos[1]+1][(pos[0]+1)%maxX]) or isPassable(levelStr[pos[1]+1][(pos[0]+1)%maxX]):
						neighbors.append([dist+1.4,((pos[0]+1)%maxX,pos[1]+1,-1)])
						#print("falling right")
					if not isSolid(levelStr[pos[1]+1][(pos[0]-1)%maxX]) or isPassable(levelStr[pos[1]+1][(pos[0]-1)%maxX]):
						neighbors.append([dist+1.4,((pos[0]-1)%maxX,pos[1]+1,-1)])
						#print("falling left")
					if not isSolid(levelStr[pos[1]+1][pos[0]]) or isPassable(levelStr[pos[1]+1][pos[0]]):
						neighbors.append([dist+1,(pos[0],pos[1]+1,-1)])
						#falling straight down
				if pos[1]+2 <= maxY:
					if (not isSolid(levelStr[pos[1]+2][(pos[0]+1)%maxX]) or isPassable(levelStr[pos[1]+2][(pos[0]+1)%maxX])) and (not isSolid(levelStr[pos[1]+1][(pos[0]+1)%maxX]) or isPassable(levelStr[pos[1]+1][(pos[0]+1)%maxX])):
						neighbors.append([dist+2,((pos[0]+1)%maxX,pos[1]+2,-1)])
						#print("falling right fast")
					if (not isSolid(levelStr[pos[1]+2][(pos[0]-1)%maxX]) or isPassable(levelStr[pos[1]+2][(pos[0]-1)%maxX])) and (not isSolid(levelStr[pos[1]+1][(pos[0]-1)%maxX]) or isPassable(levelStr[pos[1]+1][(pos[0]-1)%maxX])):
						neighbors.append([dist+2,((pos[0]-1)%maxX,pos[1]+2,-1)])
						#print("falling left fast")
				#	if (not isSolid(levelStr[pos[1]+2][pos[0]]) or isPassable(levelStr[pos[1]+2][pos[0]])) and (not isSolid(levelStr[pos[1]+1][pos[0]]) or isPassable(levelStr[pos[1]+1][pos[0]])):
				#		neighbors.append([dist+2,(pos[0],pos[1]+2,-1)])
						#falling straight down fast
			else:
				if pos[1]+1 <= maxY:
					if pos[0]+1 < maxX and (not isSolid(levelStr[pos[1]+1][pos[0]+1]) or isPassable(levelStr[pos[1]+1][pos[0]+1])):
						neighbors.append([dist+1.4,(pos[0]+1,pos[1]+1,-1)])
						#print("falling right")
					if pos[0]-1 >= 0 and (not isSolid(levelStr[pos[1]+1][pos[0]-1]) or isPassable(levelStr[pos[1]+1][pos[0]-1])):
						neighbors.append([dist+1.4,(pos[0]-1,pos[1]+1,-1)])
						#print("falling left")
					if not isSolid(levelStr[pos[1]+1][pos[0]]) or isPassable(levelStr[pos[1]+1][pos[0]]):
						neighbors.append([dist+1,(pos[0],pos[1]+1,-1)])
						#falling straight down
				if pos[1]+2 <= maxY:
					if pos[0]+1 < maxX and (not isSolid(levelStr[pos[1]+2][pos[0]+1]) or isPassable(levelStr[pos[1]+2][pos[0]+1])) and (not isSolid(levelStr[pos[1]+1][pos[0]+1]) or isPassable(levelStr[pos[1]+1][pos[0]+1])):
						neighbors.append([dist+2,(pos[0]+1,pos[1]+2,-1)])
						#print("falling right fast")
					if pos[0]-1 >= 0 and (not isSolid(levelStr[pos[1]+2][pos[0]-1]) or isPassable(levelStr[pos[1]+2][pos[0]-1])) and (not isSolid(levelStr[pos[1]+1][pos[0]-1]) or isPassable(levelStr[pos[1]+1][pos[0]-1])):
						neighbors.append([dist+2,(pos[0]-1,pos[1]+2,-1)])
						#print("falling left fast")
					#if (not isSolid(levelStr[pos[1]+2][pos[0]]) or isPassable(levelStr[pos[1]+2][pos[0]])) and (not isSolid(levelStr[pos[1]+1][pos[0]]) or isPassable(levelStr[pos[1]+1][pos[0]])):
					#	neighbors.append([dist+2,(pos[0],pos[1]+2,-1)])
						#falling straight down fast
		
		#if currently on a climbable tile, see if we can climb in any direction
		if isClimbable(levelStr[pos[1]][pos[0]]):
			
			#up
			if pos[1]+1 <=maxY and (isClimbable(levelStr[below[1]][below[0]]) or not isSolid(levelStr[below[1]][below[0]]) or isPassable(levelStr[below[1]][below[0]])):
				neighbors.append([dist+1, (pos[0], pos[1]+1,-1)])
			#down
			if pos[1]-1 >= 0 and (isClimbable(levelStr[pos[1]-1][pos[0]]) or not isSolid(levelStr[pos[1]-1][pos[0]]) or isPassable(levelStr[pos[1]-1][pos[0]])):
				neighbors.append([dist+1, (pos[0], pos[1]-1,-1)])

			if level_wrap:
				#left
				if isClimbable(levelStr[pos[1]][(pos[0]-1)%maxX]) or (not isSolid(levelStr[pos[1]][(pos[0]-1)%maxX]) or isPassable(levelStr[pos[1]][(pos[0]-1)%maxX])):
					neighbors.append([dist+1, ((pos[0]-1)%maxX, pos[1],-1)])
				#right
				if isClimbable(levelStr[pos[1]][(pos[0]+1)%maxX]) or (not isSolid(levelStr[pos[1]][(pos[0]+1)%maxX]) or isPassable(levelStr[pos[1]][(pos[0]+1)%maxX])):
					neighbors.append([dist+1, ((pos[0]+1)%maxX, pos[1],-1)])
			else:
				#left
				if pos[0]-1 >= 0 and (isClimbable(levelStr[pos[1]][pos[0]-1]) or not isSolid(levelStr[pos[1]][pos[0]-1]) or isPassable(levelStr[pos[1]][pos[0]-1])):
					neighbors.append([dist+1, (pos[0]-1, pos[1],-1)])
				#right
				if pos[0]+1 < maxX and (isClimbable(levelStr[pos[1]][pos[0]+1]) or not isSolid(levelStr[pos[1]][pos[0]+1]) or isPassable(levelStr[pos[1]][pos[0]+1])):
					neighbors.append([dist+1, (pos[0]+1, pos[1],-1)])


		return neighbors
	return getNeighbors

def goalReached(current_width, current_height, goal_width, goal_height):
	#If no goal specified
	if goal_height == None and goal_width == None:
		print("No goal specified; so we reached it!")
		return True
	#If only a horizontal goal
	elif goal_height == None and goal_width != None:
		if current_width == goal_width:
			print("horizontal goal reached!")
			return True
		else:
			return False
	#If only a vertical goal
	elif goal_height != None and goal_width == None:
		if current_height == goal_height:
			print("vertical goal reached!")
			return True
		else:
			return False
	#If an exact positional goal
	elif goal_height != None and goal_width != None:
		if current_height == goal_height and current_width == goal_width:
			print("positional goal reached!")
			return True
		else:
			return False

def tileDistance(X_1, Y_1, X_2, Y_2):
	if X_2 == None:
		return abs(Y_2 - Y_1)
	elif Y_2 == None:
		return abs(X_2 - X_1)
	else:
		return abs(X_2 - X_1) + abs(Y_2 - Y_1)

def makeGetNeighbors(jumps,levelStr,visited,isSolid):
    maxX = len(levelStr[0])-1
    maxY = len(levelStr)-1
    jumpDiffs = []
    for jump in jumps:
        jumpDiff = [jump[0]]
        for ii in range(1,len(jump)):
            jumpDiff.append((jump[ii][0]-jump[ii-1][0],jump[ii][1]-jump[ii-1][1]))
        jumpDiffs.append(jumpDiff)
    jumps = jumpDiffs
    def getNeighbors(pos):
        #print('Input pos: ', pos)
        dist = pos[0]-pos[2]
        #print('dist:',dist)
        pos = pos[1]
        #print('pos:',pos)
        visited.add((pos[0],pos[1]))
        #print('visited:',visited)
        below = (pos[0],pos[1]+1)
        neighbors = []
        if below[1] > maxY:
            return []
        if pos[2] != -1:
            ii = pos[3] +1
            jump = pos[2]
            if ii < len(jumps[jump]):
                if  not (pos[0]+pos[4]*jumps[jump][ii][0] > maxX or pos[0]+pos[4]*jumps[jump][ii][0] < 0 or pos[1]+jumps[jump][ii][1] < 0) and not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][pos[0]+pos[4]*jumps[jump][ii][0]]):
                    neighbors.append([dist+1,(pos[0]+pos[4]*jumps[jump][ii][0],pos[1]+jumps[jump][ii][1],jump,ii,pos[4])])
                if pos[1]+jumps[jump][ii][1] < 0 and not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][pos[0]+pos[4]*jumps[jump][ii][0]]):
                    neighbors.append([dist+1,(pos[0]+pos[4]*jumps[jump][ii][0],0,jump,ii,pos[4])])
                
        if isSolid(levelStr[below[1]][below[0]]):
            if pos[0]+1 <= maxX and not isSolid(levelStr[pos[1]][pos[0]+1]):
                neighbors.append([dist+1,(pos[0]+1,pos[1],-1)])
            if pos[0]-1 >= 0 and not isSolid(levelStr[pos[1]][pos[0]-1]):
                neighbors.append([dist+1,(pos[0]-1,pos[1],-1)])

            for jump in range(len(jumps)):
                ii = 0
                if not (pos[0]+jumps[jump][ii][0] > maxX or pos[1] < 0) and not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][pos[0]+jumps[jump][ii][0]]):
                    neighbors.append([dist+ii+1,(pos[0]+jumps[jump][ii][0],pos[1]+jumps[jump][ii][1],jump,ii,1)])

                if not (pos[0]-jumps[jump][ii][0] < 0 or pos[1] < 0) and not isSolid(levelStr[pos[1]+jumps[jump][ii][1]][pos[0]-jumps[jump][ii][0]]):
                    neighbors.append([dist+ii+1,(pos[0]-jumps[jump][ii][0],pos[1]+jumps[jump][ii][1],jump,ii,-1)])

        else:
            neighbors.append([dist+1,(pos[0],pos[1]+1,-1)])
            if pos[1]+1 <= maxY:
                #print('pos: ',pos)
                #print(pos[1]+1, pos[0]+1)
                if not isSolid(levelStr[pos[1]+1][pos[0]+1]):
                    neighbors.append([dist+1.4,(pos[0]+1,pos[1]+1,-1)])
                if not isSolid(levelStr[pos[1]+1][pos[0]-1]):
                    neighbors.append([dist+1.4,(pos[0]-1,pos[1]+1,-1)])
            if pos[1]+2 <= maxY:
                if not isSolid(levelStr[pos[1]+2][pos[0]+1]):
                    neighbors.append([dist+2,(pos[0]+1,pos[1]+2,-1)])
                if not isSolid(levelStr[pos[1]+2][pos[0]-1]):
                    neighbors.append([dist+2,(pos[0]-1,pos[1]+2,-1)])
        return neighbors
    return getNeighbors

def findPaths(solids,jumps,levelStr,is_vertical=False):
	visited = set()
	isSolid = makeIsSolid(solids)
	print('jumps:',jumps)
	print('solids:',solids)
	getNeighbors = makeGetNeighbors(jumps,levelStr,visited,isSolid)
	maxX = len(levelStr[0]) - 1
	maxY = len(levelStr) - 1
	print(maxX,maxY)
	if is_vertical:
		startX = None
		startY = maxY
		while startX == None:
			startY -= 1
			for x in range(1, maxX):
				if not isSolid(levelStr[startY][x]) and isSolid(levelStr[startY + 1][x]):
					startX = x
					break

		goalX = None
		goalY = 0
		while goalX == None:
			goalY += 1
			for x in range(1, maxX):
				if not isSolid(levelStr[goalY][x]) and isSolid(levelStr[goalY + 1][x]):
					goalX = x
					break

	else:
		startX = -1
		startY = None
		while startY == None:
			startX += 1
			if startX == 16:
				break
			for y in range(maxY-1, 0, -1):
				if not isSolid(levelStr[y][startX]) and isSolid(levelStr[y + 1][startX]):
					startY = y
					break

		goalX = maxX+1
		goalY = None
		while goalY == None:
			goalX -= 1
			if goalX < 0:
				break
			for y in range(maxY-1, 0, -1):
				if not isSolid(levelStr[y][goalX]) and isSolid(levelStr[y + 1][goalX]):
					goalY = y
					break

	#level[startY][startX] = '{'
	#level[goalY][goalX] = '}'
	#paths = pathfinding.astar_shortest_path( (2,2,-1), lambda pos: pos[0] == maxX, getNeighbors, subOptimal, lambda pos: 0)#lambda pos: abs(maxX-pos[0]))
	if None in [startX, startY, goalX, goalY]:
		print('start goal error')
		print(startX, startY, goalX, goalY)
		return
	path, node = pathfinding.astar_shortest_path( (startX,startY,-1), lambda pos: pos[0] == goalX and pos[1] == goalY, getNeighbors, lambda pos: 0)#lambda pos: abs(maxX-pos[0]))
	print(node)
	dist = node[1][0]
	if not path:
		return None, dist
	return [ (p[0],p[1]) for p in path], dist

def findPathsFull(game_desc, levelStr, game):
	visited = set()
	solids, passables, climbables, hazards, jumps = game_desc['solid'],game_desc['passable'],game_desc['climbable'],game_desc['hazard'],game_desc['jumps']
	isSolid = makeIsSolid(solids)
	isPassable = makeIsPassable(passables, hazards)
	isHazard = makeIsHazard(hazards)
	isClimbable = makeIsClimbable(climbables)

	#maxX = len(levelStr[0])-1
	maxX = len(levelStr[0]) - 1
	maxY = len(levelStr) - 1
	starts_h, goals_h = set(), set()
	starts_v, goals_v = set(), set()
	sg = set()
	if game == 'ki' or game == 'mm' or game == 'ng':
		startX = None
		startY = maxY-1
		while startX == None and startY > maxY-1:
			startY -= 1
			for x in range(0, maxX):
				if not isSolid(levelStr[startY][x]) and isSolid(levelStr[startY + 1][x]):
					startX = x
					starts_v.add((startX,startY))
					if game == 'mm':  # MM can go both ways vertically
						goals_v.add((startX,startY))
					break
		
		startX = None
		startY = maxY-1
		while startX == None and startY > maxY-1:
			startY -= 1
			for x in range(maxX-1, 0, -1):
				if not isSolid(levelStr[startY][x]) and isSolid(levelStr[startY + 1][x]):
					startX = x
					starts_v.add((startX,startY))
					if game == 'mm':
						goals_v.add((startX,startY))
					break

		goalX = None
		goalY = -1
		while goalX == None and goalY < 0:
			goalY += 1
			for x in range(0, maxX):
				if not isSolid(levelStr[goalY][x]) and isSolid(levelStr[goalY + 1][x]):
					goalX = x
					goals_v.add((goalX,goalY))
					if game == 'mm':
						starts_v.add((goalX,goalY))
					break

		goalX = None
		goalY = -1
		while goalX == None and goalY < 0:	
			goalY += 1
			for x in range(maxX-1, 0, -1):
				if not isSolid(levelStr[goalY][x]) and isSolid(levelStr[goalY + 1][x]):
					goalX = x
					goals_v.add((goalX,goalY))
					if game == 'mm':
						starts_v.add((goalX,goalY))
					break
	if game == 'smb' or game == 'mm' or game == 'cv' or game == 'ng':
		startX = -1
		startY_bottom, startY_top = None, None
		while startY_bottom == None and startX < 0:
			startX += 1
			for y in range(maxY-1, 0, -1):
				if not isSolid(levelStr[y][startX]) and isSolid(levelStr[y + 1][startX]):
					startY_bottom = y
					starts_h.add((startX,startY_bottom))
					break
		
		startX = -1
		while startY_bottom == None and startX < 0:
			startX += 1
			for y in range(0, maxY-1):
				if not isSolid(levelStr[y][startX]) and isSolid(levelStr[y + 1][startX]):
					startY_top = y
					starts_h.add((startX,startY_top))
					break
		
		goalX = maxX+1
		goalY_bottom, goalY_top = None, None
		while goalY_bottom == None and goalX > maxX:
			goalX -= 1
			for y in range(maxY-1, 0, -1):
				if not isSolid(levelStr[y][goalX]) and isSolid(levelStr[y + 1][goalX]):
					goalY_bottom = y
					goals_h.add((goalX, goalY_bottom))
					break

		goalX = maxX+1
		while goalY_bottom == None and goalX > maxX:
			goalX -= 1
			for y in range(0, maxY-1):
				if not isSolid(levelStr[y][goalX]) and isSolid(levelStr[y + 1][goalX]):
					goalY_top = y
					goals_h.add((goalX, goalY_top))
					break
	for start in starts_h:
		for goal in goals_h:
			sg.add((start,goal))
	
	for start in starts_v:
		for goal in goals_v:
			sg.add((start,goal))

		"""
		startX_p = -1
		startY_p = None
		while startY_p == None and startX_p < maxX:
			startX_p += 1
			for y in range(maxY-1, 0, -1):
				#print(y,startX_p)
				if levelStr[y][startX_p] == 'P':
					startY_p = y
					#starts.append((startX_p,startY_p))
					break
		goalX_p = maxX+1
		goalY_p = None
		while goalY_p == None and goalX_p > 0:
			goalX_p -= 1
			for y in range(maxY-1, 0, -1):
				if levelStr[y][goalX_p] == 'P':
					goalY_p = y
					#goals.append((goalX_p,goalY_p))
					break
		"""
		#if None not in [startX_p,startY_p,goalX_p,goalY_p]:
		#	sg.add(((startX_p,startY_p),(goalX_p,goalY_p)))

	#level[startY][startX] = '{'
	#level[goalY][goalX] = '}'
	#print('start: ', startX, startY)
	#print('goal: ', goalX, goalY)
	#if None in [startX, startY, goalX, goalY]:
	#	print('start goal error')
	#	print(startX, startY, goalX, goalY)
	#	return None, 0

	if len(sg) == 0:
		return None, 0
	getNeighbors = makeGetNeighborsFull(jumps,levelStr,visited,isSolid,isPassable,isClimbable,isHazard,game)
	#paths = pathfinding.astar_shortest_path( (start_X, start_Y,-1), lambda pos: pos[0] == maxX, getNeighbors, subOptimal, lambda pos: 0)
	paths, best_dist, best_path = [], 0, None
	for start, goal in sg:
		startX, startY = start
		goalX, goalY = goal
		path, node = pathfinding.astar_shortest_path( (startX, startY,-1), lambda pos: pos[0] == goalX and pos[1] == goalY, getNeighbors, lambda pos: 0)
		#path, node = pathfinding.astar_shortest_path( (startX, startY,-1), lambda pos: pos[1] == 0 if is_vertical else pos[0] == maxX, getNeighbors, lambda pos: 0)
		if path:
			first, last = path[0], path[-1]
			#print('First:',first,' Last:',last)
			if last[0] == goalX and last[1] == goalY:
				dist = 16
		else:
			final_x, final_y = node[1][0], node[1][1]
			dist_x = abs(final_x - startX)
			dist_y = abs(final_y - startY)
			dist = max(dist_x, dist_y)
		#dist = node[1][0]+1
		if dist > best_dist:
			best_dist = dist
			best_path = path
	#print(node)
	
	best_dist /= 16
	if not best_path:
		return None, best_dist
	return [ (p[0],p[1]) for p in best_path], best_dist