# compare one long bundle to multiple smaller bundles and also to
# binaraized sdm.  Shows multiple smaller bundles a little bit
# more efficient than a large bundle and binarized sdm with nact==1
# requires a lot more space.  Output needs to be cleaned up.


from scipy import special
import math

def bunlen(k, per):
	# calculated bundle length needed to store k items with accuracy per
	# This calculates the mean distance to the matching vector using
	# approximation on page 3 in Pentti's paper (referenced above)
	# then calls dplen to calculate the vector length.
	return dplen(0.5 - 0.4 / math.sqrt(k - 0.44), per)


def dplen(mm, per):
	# calculate vector length requred to store bundle at per accuracy
	# mm - mean of match distribution (single bit error rate, 0< mm < 0.5)
	# per - desired probability of error on recall (e.g. 0.000001)
	# This derived from equations in Pentti's 1997 paper, "Fully Distributed
	# Representation", page 3, solving per = probability random vector has
	# smaller hamming distance than hamming to matching vector, (difference
	# in normal distributions) is less than zero; solve for length (N)
	# in terms of per and mean distance to match (mm, denoted as delta in the
	# paper.)
	n = (-2*(-0.25 - mm + mm**2)*special.erfinv(-1 + 2*per)**2)/(0.5 - mm)**2
	return round(n) + 1

binarized_sdm_num_rows = [
		(1, 50),
		(2, 97),
		(3, 161),
		(4, 254),
		(5, 396),
		(6, 619),
		(7, 984),
		(8, 1599),
		(9,2665)]

def compare_bundles(k,m):
	# k - number of items stored
	# m - number of groups items split into for storing in smaller bundles
	print("Storing k=%s items, spliting bundle into m=%s parts" % (k, m))
	print("err - desired error in 10**(-err)")
	print("len1 - length of superposition vector to acheive desire error, assuming match to item and one random distractor")
	print("lenm - length of bundle needed to store k / m items at desired error; then times m (to compare with bun1)")
	print("pdif - percent difference, round(100*(lenm - len1)/len1,2)")
	print("rows512 - number of rows in sdm to store len1 bits, e.g. ound(len1 / 512)")
	print("rows_512 - number rows required to store 1000 items in sdm using binarized sdm, activaction count == 1")
	print("factor - number rows required, found empirically / num rows predicted; e.g. round(bsd_rows / rows_512, 2)")
	print("err\tlen1\tlenm\t%diff\trows_512\tbsd_rows\tfactor")
	for i in range(1,10):
		# import pdb; pdb.set_trace()
		err = 10**(-i)
		len1 =  bunlen(k, err)
		lenm = bunlen(int(k/m), err) * m
		pdif = round(100*(lenm - len1)/len1,2)
		rows_512 = round(len1 / 512)
		bsd_err, bsd_rows = binarized_sdm_num_rows[i-1]
		assert bsd_err == i
		factor = round(bsd_rows / rows_512, 2)
		print("%s\t%s\t%s\t%s\t%s\t%s\t%s" % (i, len1, lenm, pdif, rows_512, bsd_rows, factor))


def main():
	compare_bundles(1000, 10)


if __name__ == "__main__":
	# test sdm and bundle
	main()
