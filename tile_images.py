from PIL import Image

# {'#': 0, '(': 1, ')': 2, '+': 3, '-': 4, 'B': 5, 'D': 6, 'E': 7, 'P': 8, '[': 9, ']': 10, '^': 11, 'v': 12}
met_images = {
    "#":Image.open('tiles/Met_X.png'),  # solid
    "(":Image.open('tiles/0.png'),  # beam around door (ignore using background)
    ")":Image.open('tiles/0.png'),  # beam around door (ignore using background)
    "+":Image.open('tiles/Met_+.png'),  # powerup
    "-":Image.open('tiles/0.png'),   # background
    "B":Image.open('tiles/Met_B.png'),  # breakable
    "D":Image.open('tiles/Met_D.png'),  # door
    "E":Image.open('tiles/Met_E.png'),  # enemy
    "P":Image.open('tiles/0.png'),   # path
    "[":Image.open('tiles/Met_[.png'),  # ??
    "]":Image.open('tiles/Met_].png'),  # ??
    "^":Image.open('tiles/Met_^2.png'),  # lava
    "v":Image.open('tiles/0.png')  # ??
}

mm_images = {
    "#":Image.open('tiles/MM_X2.png'),
    "*":Image.open('tiles/MM_star.png'),
    "+":Image.open('tiles/MM_+.png'),
    "-":Image.open('tiles/-.png'),
    "B":Image.open('tiles/MM_B2.png'),
    "C":Image.open('tiles/CMM.png'),
    "D":Image.open('tiles/DMM.png'),
    "H":Image.open('tiles/HMM.png'),
    "L":Image.open('tiles/MM_L.png'),
    "M":Image.open('tiles/MMM.png'),
    "P":Image.open('tiles/-.png'),
    "U":Image.open('tiles/MM_U.png'),
    "W":Image.open('tiles/MM_w.png'),
    "l":Image.open('tiles/MM_L.png'),
    "t":Image.open('tiles/TMM.png'),
    "w":Image.open('tiles/MM_w.png'),
    "|":Image.open('tiles/LMM.png')
}

# {'B': 0, 'D': 1, 'F': 2, 'I': 3, 'M': 4, 'O': 5, 'P': 6, 'S': 7, 'W': 8}
zelda_images = {
   "B":Image.open('tiles/Z_B.png'),
   "D":Image.open('tiles/DMM.png'),
   "F":Image.open('tiles/Z_F.png'),
   "I":Image.open('tiles/Z_I.png'),
   "M":Image.open('tiles/Z_M.png'),
   "O":Image.open('tiles/Z_O.png'),
   "P":Image.open('tiles/Z_P.png'),
   "S":Image.open('tiles/Z_S.png'),
   "W":Image.open('tiles/Z_W.png')
}
# {'-': 0, 'B': 1, 'E': 2, 'G': 3, 'L': 4, 'M': 5, 'R': 6, 'b': 7}
lode_images = {
    "-":Image.open('tiles/0.png'),  # background
    'B':Image.open('tiles/LR_bb.png'),  # solid ground   # # for blend
    'b':Image.open('tiles/LR_b.png'),  # solid ground diggable
    'E':Image.open('tiles/LR_E.png'),  # enemy
    'R':Image.open('tiles/LR_R.png'), # rope
    'L':Image.open('tiles/LR_L.png'), # ladder
    'G':Image.open('tiles/LR_G.png'), # gold
    'M':Image.open('tiles/LR_M.png'), # spawn point   # N for blend
}

bmm_images = {
    "#":Image.open('tiles/Met_X.png'),  # solid
    "+":Image.open('tiles/Met_+.png'),  # powerup
    "-":Image.open('tiles/0.png'),   # background
    "B":Image.open('tiles/Met_B.png'),  # breakable
    "D":Image.open('tiles/Met_D.png'),  # door
    "E":Image.open('tiles/Met_E.png'),  # enemy
    "^":Image.open('tiles/Met_^2.png'),  # lava

    #"#":Image.open('tiles/MM_X2.png'),
    "*":Image.open('tiles/MM_star.png'),
    #"+":Image.open('tiles/MM_+.png'),
    #"-":Image.open('tiles/-.png'),
    #"B":Image.open('tiles/MM_B2.png'),
    "C":Image.open('tiles/CMM.png'),
    "D":Image.open('tiles/DMM.png'),
    "H":Image.open('tiles/HMM.png'),
    "L":Image.open('tiles/MM_L.png'),
    "M":Image.open('tiles/MMM.png'),
    "U":Image.open('tiles/MM_U.png'),
    "W":Image.open('tiles/MM_w.png'),
    "l":Image.open('tiles/MM_L.png'),
    "t":Image.open('tiles/TMM.png'),
    "w":Image.open('tiles/MM_w.png'),
    "|":Image.open('tiles/LMM.png')
}

bzl_images = {
    "B":Image.open('tiles/Z_B.png'),
    "D":Image.open('tiles/DMM.png'),
    "F":Image.open('tiles/Z_F.png'),
    "I":Image.open('tiles/Z_I.png'),
    "M":Image.open('tiles/Z_M.png'),
    "O":Image.open('tiles/Z_O.png'),
    "P":Image.open('tiles/Z_P.png'),
    "S":Image.open('tiles/Z_S.png'),
    "W":Image.open('tiles/Z_W.png'),

    "-":Image.open('tiles/0.png'),  # background
    '#':Image.open('tiles/LR_bb.png'),  # solid ground   # # for blend
    'b':Image.open('tiles/LR_b.png'),  # solid ground diggable
    'E':Image.open('tiles/LR_E.png'),  # enemy
    'R':Image.open('tiles/LR_R.png'), # rope
    'L':Image.open('tiles/LR_L.png'), # ladder
    'G':Image.open('tiles/LR_G.png'), # gold
    'N':Image.open('tiles/LR_M.png'), # spawn point   # N for blend
}

def common(g1,g2):
    for k in g1:
        if k in g2:
            print(k)

common(mm_images, zelda_images)