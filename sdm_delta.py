import pprint
import re
import sys
import matplotlib.pyplot as plt
pp = pprint.PrettyPrinter(indent=4)
import overlap as ov


# script to test equations and empiricial output of overlap routine matches
# found SDM delta when running "python itmx.py -p f'.  The output from itmx.py
# is included in this script since that takes fairly long to run.

# Feb 2, 2022: This is just started, not complete.

sdm_deltas = [
{
	"per,cbs": "10,36",
	"Num items / sdm (nrows, cols)":
	['5/(6, 52)', '11/(9, 59)', '21/(12, 70)', '51/(40, 49)', '101/(94, 39)', '250/(214, 39)'],
		'sdm_delta_found': [   0.14682692307692308,
						   0.20467796610169492,
						   0.2346285714285714,
						   0.2520408163265306,
						   0.27499999999999997,
						   0.371],
	'sdm_delta_predicted': [   0.24845292378238398,
							   0.2612156424807822,
							   0.2790923247113286,
							   0.2406607189039176,
							   0.21581731923793607,
							   0.2139892353103407],
	'sdm_dimensions': [   (6, 52, 2, 586),
						  (9, 59, 2, 858),
						  (12, 70, 2, 1256),
						  (40, 49, 3, 2404),
						  (94, 39, 4, 4319),
						  (214, 39, 5, 9462)],
},

{
	"per,cbs":"10,100",
	"Num items / sdm (nrows, cols)":
	['5/(9, 54)', '11/(12, 63)', '21/(16, 73)', '51/(40, 61)', '101/(94, 49)', '250/(214, 48)'],

'sdm_delta_found': [   0.10335185185185186,
						   0.15022222222222223,
						   0.21749315068493152,
						   0.2139344262295082,
						   0.24022448979591837,
						   0.29931250000000004],
	'sdm_delta_predicted': [   0.2266273523768682,
							   0.2442111583112968,
							   0.2609196951668077,
							   0.2406607189039176,
							   0.21581731923793607,
							   0.2139892353103407],
	'sdm_dimensions': [   (9, 54, 2, 1215),
						  (12, 63, 2, 1630),
						  (16, 73, 2, 2230),
						  (40, 61, 3, 3491),
						  (94, 49, 4, 5793),
						  (214, 48, 5, 12217)],
},

{
	"per,cbs":"1.0,36",
	"Num items / sdm (nrows, cols)":
		['5/(6, 82)', '11/(9, 92)', '21/(12, 110)', '51/(40, 76)', '101/(94, 62)', '250/(214, 61)'],
		'sdm_delta_found': [   0.14930487804878048,
						   0.19293478260869565,
						   0.23372727272727273,
						   0.21676315789473685,
						   0.25612903225806455,
						   0.372016393442623],
	'sdm_delta_predicted': [   0.24845292378238398,
							   0.2612156424807822,
							   0.2790923247113286,
							   0.2406607189039176,
							   0.21581731923793607,
							   0.2139892353103407],
	'sdm_dimensions': [   (6, 82, 2, 921),
						  (9, 92, 2, 1347),
						  (12, 110, 2, 1973),
						  (40, 76, 3, 3777),
						  (94, 62, 4, 6787),
						  (214, 61, 5, 14867)],
 },
{	"per,cbs":"1.0,100",
	"Num items / sdm (nrows, cols)":
		['5/(9, 79)', '11/(12, 92)', '21/(16, 107)', '51/(40, 89)', '101/(94, 72)', '250/(214, 71)'],
		'sdm_delta_found': [   0.08877215189873418,
						   0.15678260869565216,
						   0.20622429906542056,
						   0.22919101123595506,
						   0.17995833333333333,
						   0.2996901408450704],
	'sdm_delta_predicted': [   0.2266273523768682,
							   0.2442111583112968,
							   0.2609196951668077,
							   0.2406607189039176,
							   0.21581731923793607,
							   0.2139892353103407],
	'sdm_dimensions': [   (9, 79, 2, 1777),
						  (12, 92, 2, 2384),
						  (16, 107, 2, 3262),
						  (40, 89, 3, 5107),
						  (94, 72, 4, 8473),
						  (214, 71, 5, 17870)],
},
{
	"per,cbs":"0.1,36",
	"Num items / sdm (nrows, cols)":
		['5/(6, 112)', '11/(9, 126)', '21/(12, 150)', '51/(40, 104)', '101/(94, 84)', '250/(214, 83)'],
	'sdm_delta_found': [   0.14208928571428572,
						   0.19080952380952382,
						   0.23731333333333335,
						   0.19971153846153847,
						   0.2510595238095238,
						   0.35106024096385546],
	'sdm_delta_predicted': [   0.24845292378238398,
							   0.2612156424807822,
							   0.2790923247113286,
							   0.2406607189039176,
							   0.21581731923793607,
							   0.2139892353103407],
	'sdm_dimensions': [   (6, 112, 2, 1257),
						  (9, 126, 2, 1840),
						  (12, 150, 2, 2695),
						  (40, 104, 3, 5158),
						  (94, 84, 4, 9268),
						  (214, 83, 5, 20304)],
},
{
	"per,cbs":"0.1,100",
	"Num items / sdm (nrows, cols)":
		['5/(9, 103)', '11/(12, 121)', '21/(16, 141)', '51/(40, 117)', '101/(94, 94)', '250/(214, 93)'],
 'sdm_delta_found': [   0.09240776699029127,
						   0.16528099173553717,
						   0.20009929078014183,
						   0.2072991452991453,
						   0.18801063829787232,
						   0.3020430107526882],
	'sdm_delta_predicted': [   0.2266273523768682,
							   0.2442111583112968,
							   0.2609196951668077,
							   0.2406607189039176,
							   0.21581731923793607,
							   0.2139892353103407],
	'sdm_dimensions': [   (9, 103, 2, 2339),
						  (12, 121, 2, 3138),
						  (16, 141, 2, 4293),
						  (40, 117, 3, 6722),
						  (94, 94, 4, 11153),
						  (214, 93, 5, 23524)],
}
]
def test_deltas():
	global sdm_deltas
	for tr in sdm_deltas:
		per, cbs = tr["per,cbs"].split(",")  # desired percent error, coodbook size
		per = float(per)
		cbs = int(cbs)
		isdm = tr["Num items / sdm (nrows, cols)"]
		sdm_d = tr['sdm_dimensions']
		ki = []
		deltaEmpirical = []
		for i in range(len(isdm)):
			x = isdm[i]
			m = re.match(r'(\d+)/\((\d+), (\d+)\)', x)
			if m is None:
				sys.exit("Could not match: %s" % x)  # 5/(6, 52)
			k = int(m.group(1))
			nrows = int(m.group(2))
			ncols = int(m.group(3))
			assert nrows == sdm_d[i][0]
			assert ncols == sdm_d[i][1]
			nact = sdm_d[i][2]
			ki.append((k, nrows, ncols))
			deltaEmpirical.append(ov.sdm_delta_empirical(nrows, k, nact, num_trials=1000))
		sdm_d = tr['sdm_dimensions']
		assert len(sdm_d) == len(ki)
		xvals = range(len(ki))
		dfound = tr["sdm_delta_found"]
		dpred = tr["sdm_delta_predicted"]
		xaxis_labels = ["%s/(%s,%s)" % ki[i] for i in range(len(ki)) ]
		title = "Predicted vs found delta for desired error=%s%%, codebook size=%s" % (per, cbs)
		fig = plt.figure()
		plt.errorbar(xvals, dfound, yerr=None, label="itmx found", fmt="-o")
		plt.errorbar(xvals, dpred, yerr=None, label="Predicted", fmt="-o")
		plt.errorbar(xvals, deltaEmpirical, yerr=None, label="Empirical", fmt="-o")	
		plt.xticks(xvals,xaxis_labels)
		plt.title(title)
		plt.xlabel("number items (k) / (nrows, ncols)")
		plt.ylabel("delta")
		plt.legend(loc='upper right')
		plt.grid()
		plt.show()


test_deltas()







