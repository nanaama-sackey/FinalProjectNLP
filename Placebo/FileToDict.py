def dictCreate(Questions, Answers):
	keys = read("Questions.txt")
	values = read("Answers.txt")
	dictionary = dict(zip(keys, values))
	print(dictionary)


def read(fname):
	with open(fname) as f:
	    content = f.readlines()
	content = [x.strip() for x in content]

