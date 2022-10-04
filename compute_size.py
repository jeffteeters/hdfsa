# script to compute sizes (# dimensions in bind and # rows in sdm)

import sys

def bind_len(storage, num_items):
	# given storage in bytes, compute vector length to use that storage
	# global num_items
	length = round((storage * 8) / (num_items + 1))
	return length


def sdm_len(storage, num_items, bits_per_counter):
	# given storage in bytes, compute # rows in SDM memory to use that storage
	# global num_items
	sdm_word_length = 512  # num bites in sdm address and memory
	bytes_in_counter_row = sdm_word_length * bits_per_counter / 8
	bytes_in_address = sdm_word_length/8 if num_items > 0 else 0  # assume address can be made dynamically if item_memory can
	length = round((storage - (num_items * sdm_word_length / 8)) / (bytes_in_counter_row + bytes_in_address))
	return length

def calculate_sizes(num_items, bits_per_counter=8):
	# if len(sys.argv) != 2:
	# 	sys.exit("Usage %s <storage (in bytes)>" % sys.argv[0])
	# storage = int(sys.argv[1])
	print("Sizes if %s items in item memory:" % num_items)
	print("storage\tbind_len\tsdm_len\tparameters")
	k = 1000  # assume 1000 items stored
	for i in range(1, 11):
		storage = i * 100000
		bind_l = bind_len(storage, num_items)
		sdm_l = sdm_len(storage, num_items, bits_per_counter)
		nact = round(sdm_l/(2 * sdm_l * k)**(1/3))
		old_nact = round(sdm_l / 100)
		parameters = "-b %s -m %s -a %s # old_nact=%s" % (bind_l, sdm_l, nact, old_nact)
		# print("%s\t%s\t%s" % (storage, bind_len(storage), sdm_len(storage)), parameters)
		print("%s\t%s\t%s" % (storage, bind_l, sdm_l), parameters)


def main_old():
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

def main():
	num_items = 100 + 10  # 1000 states and 10 actions
	calculate_sizes(num_items)
	print("\n")
	calculate_sizes(0, bits_per_counter=1)

main()
