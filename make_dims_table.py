# script to generate table of dimensions for Frontiers paper
# sizes are given in file mdie.py

from mdie import mdie

def get_size(short_name, error_rate):
	for mi in mdie:
		if mi["short_name"] == short_name:
			dim = mi["dims"][error_rate - 1]
			assert dim[0] == error_rate
			if short_name[0] == "S":
				size = "%s" % dim[1]
			else:
				# include nact in size
				size = "%s,%s" % (dim[1], dim[2])
			return size
	sys.exit("%s-%s not found" % (short_name, error_rate))


def main():
	for ie in range(1,10):
		sizes = " & ".join([ get_size(short_name, ie) for short_name in ("S1","S2","A1","A2","A3","A4")])
		print("%s & %s \\\\" % (ie, sizes))

main()


