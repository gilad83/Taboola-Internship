import os


root= r'Data/Single servers/AM/40 cores 187.35 GB'
dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
dirlist = [['/'+item,item] for item in dirlist]
print (dirlist)