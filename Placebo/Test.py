import re
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, LSTM, Dropout
from keras.layers import add, dot, concatenate

def tokenize(text):
	'''
	Return the tokenized form of a string 
	>>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']

	'''
	return [x.strip() for x in re.split('(\W+)?', text) if x.strip()]


def parse_questions_answers(ques_lines, answer_lines):
	'''
	Parsing the question and answers.
	'''
	data = []
	for i in range(len(ques_lines)):
		ques = tokenize(ques_lines[i])
		ans = tokenize(answer_lines[i])
		data.append((ques, ans))
	return data


def get_quesNans(ques_file, answer_file):
	'''
	Given a question and answer file names, returns parsed data pairs of them.
	'''
	data = parse_questions_answers(ques_file.readlines(), answer_file.readlines())
	return data


def parse_test_questions(ques_lines):
	'''
	Parsing the questions.
	'''
	data = []
	for i in range(len(ques_lines)):
		ques = tokenize(ques_lines[i])
		data.append((ques))
	return data


def get_test_ques(ques_file):
	'''
	Given a question file names, returns parsed data of them.
	'''
	data = parse_questions_answers(ques_file.readlines())
	return data


def vectorize_ques(data, word_idx, ques_maxlen):
	'''
	Return a vectorized form the data
	'''
	X = []
	Y = []

	for ques, ans in data:
		x = [word_idx[w] for w in ques]

		# index 0 is reserved		
		y = np.zeros(len(word_idx) + 1)
		y[word_idx[ans[0]]] = 1
		X.append(x)
		Y.append(y)

	return (pad_sequences(X, maxlen=ques_maxlen), np.array(Y))


try:
	ques = open('tester.txt', "r")
	ans = open('testerans.txt', "r")
	test_ques = open('tester.txt', "r")
	test_ans = open('testerans.txt', "r")

	print('Extracting questions and answers for training')
	training_qNa = get_quesNans(ques,ans)
	test_qNa = get_quesNans(test_ques,test_ans)

	vocab = set()
	for ques, ans in training_qNa:
		vocab |= set(ques + ans)
	vocab = sorted(vocab)

	# Reserve 0 for masking via pad_sequences
	vocab_size = len(vocab) + 1
	ques_maxlen = max(map(len, (x for x, _ in training_qNa)))
	test_maxlen = max(map(len, (x for x, _ in test_qNa)))

	print('-')
	print('Vocab size:', vocab_size, 'unique words.')
	print('Question max length:', ques_maxlen, 'words.')
	print('Number of training questions and answers:', len(training_qNa))
	print('Number of test questions:', len(test_qNa))
	print('-')
	print('Vectorizing the word sequences...')

	word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
	training_inputs, training_answers = vectorize_ques(training_qNa,
														word_idx,
														ques_maxlen)

	test_inputs, test_answers = vectorize_ques(test_qNa,
												word_idx,
												test_maxlen)

	print('-')
	print('Compiling...')

	#placeholders
	question_placeholder = Input((ques_maxlen,))

	#encoders
	#embed the questions into a sequesnce of vectors
	question_model = Sequential()
	question_model.add(Embedding(input_dim=vocab_size,
									output_dim=vocab_size,
									input_length=ques_maxlen))
	question_model.add(LSTM(100))
	question_model.add(Dropout(0.3))
	question_model.add(Dense(vocab_size, activation='sigmoid'))
	# print(question_model.summary())


	# Train and evaluate model
	question_model.compile(loss='binary_crossentropy',
							optimizer='adam',
							metrics=['accuracy'])
	# model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
 #              metrics=['accuracy'])

	# Actul training which will take time
	batch_size = 1
	num_epochs = 500

	question_model.fit(training_inputs, training_answers, batch_size = batch_size,
						epochs=num_epochs, 
						validation_data=(test_inputs, test_answers))

	# # saving the model as a pickle file
	# model_path = 'questionsmodel.h5'
	# with open(model_path, 'wb') as handle:
	# 	pickle.dump(question_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# # loading model from pickle file
	# with open(model_path, 'rb') as handle:
	# 	question_model = pickle.load(handle)

	# question_model.load_weights(model_path)

	prediction_results = question_model.predict(test_inputs)

	# Display results of test inputs
	print('-')
	print('Test results')
	print('-')

	for i in range(len(test_inputs)):

		# # Generating test question and answer
		# question = test_qNa[i][0]
		# question_sentence = ' '.join(word for word in question)
		# print("Question: ", question_sentence)

		# print('-')

		# answer = test_qNa[i][1]
		# answer_sentence = ' '.join(word for word in answer)
		# print("Actual answer: ", answer_sentence)

		# print('-')

		# Generating predicted anser
		val_max = np.argmax(prediction_results[i])
		print(prediction_results[i])

		# for key, val in word_idx.items():
		# 	print(key, val)
		    # if val == val_max:
      #   		k = key

		# print("Machine answer is: ", k)
		# print("I am ", prediction_results[i][val_max], "certain of it")



except:
	print("Error finding the question and answer files.")
	raise
