import os, sys, random
from util import pad_zelda

for file in os.listdir('zelda_rooms_new/'):
	#print(file)
	level = file[file.index('_')+1]
	data = open('zelda_rooms_new/' + file,'r').read().splitlines()
	data = [line.replace('\r\n','') for line in data]
	padded = pad_zelda(data)
	outfile = open('zelda_padded/' + file,'w')
	for i, line in enumerate(padded):
		outfile.write(line)
		if i < len(padded)-1:
			outfile.write('\n')
	outfile.close()
	#sys.exit()