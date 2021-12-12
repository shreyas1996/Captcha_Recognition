import os
import shutil

PATH = 'extracted_letter_images/'
a = os.listdir(PATH)
for x in a:
	source = PATH + x
	destination = 'extracted_letters_test/' + x
	if not os.path.exists(destination):
		os.mkdir(destination)

	files = os.listdir(source)
	length = int(0.25*len(files))

	captcha = files[:length]

	for i in captcha:
		shutil.move(source + '/' + i, destination)
