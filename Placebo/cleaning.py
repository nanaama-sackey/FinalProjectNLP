import re

# Cleaning the data so that i have a uniform looking data
# this is because some values are numbered and others are not


# # Cleaning the questions file
# def cleanfile(dirtyfile, cleanfile):

# 	infile = open(dirtyfile, "r", encoding ='utf-8')
# 	outfile = open(cleanfile, "w", encoding = 'utf-8')

# 	fileToClean = infile.readlines()

# 	for line in fileToClean:
# 		line = re.sub('[(0-9\t)\uFFFD]', '', line)

# 		outfile.write(line)

# cleanfile("Questions.txt", "que.txt")


# # Cleaning the Topics file
# def cleanfile(dirtyfile, cleanfile):

# 	infile = open(dirtyfile, "r", encoding ='utf-8')
# 	outfile = open(cleanfile, "w", encoding = 'utf-8')

# 	fileToClean = infile.readlines()

# 	for line in fileToClean:
# 		line = re.sub('[(0-9\t)\uFFFD\.]', '', line)

# 		outfile.write(line)

# cleanfile("Topics.txt", "top.txt")


# # Cleaning the Answers file
# def cleanfile(dirtyfile, cleanfile):

# 	infile = open(dirtyfile, "r", encoding ='utf-8')
# 	outfile = open(cleanfile, "w", encoding = 'utf-8')

# 	fileToClean = infile.readlines()

# 	for line in fileToClean:
# 		line = re.sub('[\t\uFFFD]', '', line)

# 		outfile.write(line)

# cleanfile("Answers.txt", "ans.txt")