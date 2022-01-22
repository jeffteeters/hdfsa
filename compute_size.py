# script to compute sizes (# dimensions in bind and # rows in sdm)

import sys

num_items = 100 + 10  # 1000 states and 10 actions

def bind_len(storage):
	# given storage in bytes, compute vector length to use that storage
	global num_items
	length = round((storage * 8) / (num_items + 1))
	return length


def sdm_len(storage):
	# given storage in bytes, compute # rows in SDM memory to use that storage
	global num_items
	sdm_word_length = 512  # num bites in sdm address and memory
	length = round((storage - (num_items * sdm_word_length / 8)) / (sdm_word_length + (sdm_word_length/8)))
	return length

def main():
	# if len(sys.argv) != 2:
	# 	sys.exit("Usage %s <storage (in bytes)>" % sys.argv[0])
	# storage = int(sys.argv[1])
	print("storage\tbind_len\tsdm_len\tparameters")
	k = 1000  # assume 1000 items stored
	for i in range(1, 11):
		storage = i * 100000
		bind_l = bind_len(storage)
		sdm_l = sdm_len(storage)
		nact = round(sdm_l/(2 * sdm_l * k)**(1/3))
		old_nact = round(sdm_l / 100)
		parameters = "-b %s -m %s -a %s # old_nact=%s" % (bind_l, sdm_l, nact, old_nact)
		print("%s\t%s\t%s" % (storage, bind_len(storage), sdm_len(storage)), parameters)

main()
