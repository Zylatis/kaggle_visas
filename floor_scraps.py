# File to keep code sippets I don't end up using but thought worth having


def kl_divergence( P, Q ):
	# Compute D_KL(P||Q)
	# Assumes they are evaluated on same x-grid
	# (could check in assert by forcing a 2D version to be fed in but is okay for now)
	assert len(P) == len(Q)

	Dkl = (np.array(P)*np.array(np.log(P/Q))).sum()

	return Dkl


# FROM THE PLOTTING ORDINAL FUNCTION
# This was to try look at similarity between histograms to see if certified and denied diff manifested
# in ordinals. Bit of overkill, just used ANOVA for mean for now
certified_density = ss.gaussian_kde(certified_data)
certified_density.covariance_factor = lambda : certified_kernel_factor
certified_density._compute_covariance()

denied_density = ss.gaussian_kde(denied_data)
denied_density.covariance_factor = lambda : denied_kernel_factor
denied_density._compute_covariance()

xs = np.arange(0,x_max,x_max/100.)
ys_certified = certified_density(xs)
ys_denied = denied_density(xs)

Dkl =  kl_divergence(ys_certified,ys_denied)

plt.plot(xs,ys_certified,antialiased=True, linewidth=2)
plt.plot(xs,ys_denied,antialiased=True, linewidth=2)

plt.xlim(right = x_max)
plt.xlim(left = 0) # assumes we won't be plottin negative stuff for now
plt.title(col + " split for certified (red) and denied (blue) - D_KL = " + str(round(Dkl,2)), fontdict = {'fontsize':10,'weight': 'bold'})
plt.xlabel(col,fontsize = 11)
plt.ylabel("PDF",fontsize = 11)
plt.savefig( img_folder + col + "_split_gaussian_kernel.png",dpi = 400)
plt.clf()



# ATTEMPT AT ITERATIVE PARALLEL CRAMERS CORRELATION TO FIX MEM USAGE ISSUE
# TL;DR IT SUUUUUCKS. Super slow, unclear why, possibly constant slicing of dataframes which is naughty.
# For now will focus on mem reduction on other case, though it's kind of useless really as just for one plot
def cramers_corr_pair( df_slice ):

	col1, col2 = df_slice.columns
	print col1, col2
	if col1 == col2:
		return 1.
	n = len(df_slice)
		
	vals1 = set(df_slice[col1].values)
	vals2 = set(df_slice[col2].values)

	chi2 = 0.
	k = len(vals1)
	r = len(vals2)
	assert n>0
	for v1 in vals1:
		di = df_slice[df_slice[col1] == v1]
		ni =1.*len(di)
		for v2 in vals2:
			
			dj = df_slice[df_slice[col2]==v2]
			dij = di[di[col2] == v2]
			nj = 1.*len(dj)
			nij = 1.*len(dij)
			
			frac = ni*nj/n
			assert ni>0
			assert nj>0
			assert n>0
			chi2 += ((nij - frac)**2)/frac
	phi2 = chi2/n
	phi2corr = max(0., phi2 - ((k-1.)*(r-1.))/(n-1.))    
	
	rcorr = r - ((r-1.)**2)/(n-1.)
	kcorr = k - ((k-1.)**2)/(n-1.)
	
	return np.float32(np.sqrt(phi2corr / ( min( (kcorr-1), (rcorr-1)) + eps)  ))


def cramers2( df ):
	cols = df.columns

	all_pairs = list(itertools.product(cols,cols))
	all_pair_frames = [ df[list(pair)] for pair in all_pairs ]
	p = mp.Pool(4)

	vals = p.map( cramers_corr_pair, all_pair_frames )
	return 0




# At one point was going to use this to do all correlations at once using label, onehot, and leavig it alone.
# However, Cramers recovers correlation between city and state which this didn't (even full one-hot, i think, recheck)
# So, given the fact we only have two numeric features we will just consider the correlations amongst categoricals
# and look at the rest separately
def mixed_encode(data, label_encode_all = False):
	# If label_encode_all = True then apply_one_hot ignored (see below for commentary on why this is dumb and should be fixed)
	encoded_dat = copy.deepcopy(data)
	one_hot_cat = []

	if not label_encode_all:
		label_cat = []
		for col in encoded_dat.columns:
			# Get list of values
			bit = list(set(data[col]))
			# Print number of different values of this feature
			print( col, len(bit) )

			# Only looking at cases where the value type is a string
			if type(data[col][0]) == str:
				bit = list(set(data[col]))
				
				# If there are more than n = 5 values for given feature, make it label encoded
				if(len(bit) > 5 ):
					label_cat.append(col)
				# if  <=5 then we can reasonably one-hot encode it using
				else:
					one_hot_cat.append( col )
	else:
		label_cat = data.columns

	for col in label_cat:
		enc = LabelEncoder()
		enc.fit( encoded_dat[col])
		encoded_dat[col] = enc.transform(encoded_dat[col])

	# Seems weird to have allow_one_hot and include catagoricals
	# The purpose is to make this general purpose.
	# When fitting we want to do the mixed encoding using one hot for certain catagoricals and label encoding for the rest

	# For correlation matrix, however, we want to leave the catagoricals alone in this function and do something else with them, 
	# namely the Cramers correlation. For now it's easier to not have that toggle here.
	# Possibly better to have the calcualtion of cols done externally and then just have this encoder take in those cols and apply
	# Add to todo list!

	encoded_dat = pd.get_dummies(encoded_dat, columns=one_hot_cat, drop_first=True)

	return encoded_dat

